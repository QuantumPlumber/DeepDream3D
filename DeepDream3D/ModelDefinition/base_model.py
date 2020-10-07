'''
Base class for defining common  model architectures.



'''

import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import mcubes
import cv2

from .utils import *


# pytorch 1.2.0 implementation


class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.linear_1 = nn.Linear(self.z_dim + self.point_dim, self.gf_dim * 8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z, is_training=False):
        zs = z.view(-1, 1, self.z_dim).repeat(1, points.size()[1], 1)
        pointz = torch.cat([points, zs], 2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        # l7 = torch.clamp(l7, min=0, max=1)
        l7 = torch.max(torch.min(l7, l7 * 0.01 + 0.99), l7 * 0.01)

        return l7


class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs, is_training=False):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.z_dim)
        d_5 = torch.sigmoid(d_5)

        return d_5


class resnet_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(resnet_block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_1 = nn.BatchNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(self.dim_out)
            nn.init.xavier_uniform_(self.conv_1.weight)
            nn.init.xavier_uniform_(self.conv_2.weight)
        else:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
            self.bn_1 = nn.BatchNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(self.dim_out)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
            self.bn_s = nn.BatchNorm2d(self.dim_out)
            nn.init.xavier_uniform_(self.conv_1.weight)
            nn.init.xavier_uniform_(self.conv_2.weight)
            nn.init.xavier_uniform_(self.conv_s.weight)

    def forward(self, input, is_training=False):
        if self.dim_in == self.dim_out:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            output = output + input
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        else:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            input_ = self.bn_s(self.conv_s(input))
            output = output + input_
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        return output


class img_encoder(nn.Module):
    def __init__(self, img_ef_dim, z_dim):
        super(img_encoder, self).__init__()
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.bn_0 = nn.BatchNorm2d(self.img_ef_dim)
        self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim * 2)
        self.res_4 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 2)
        self.res_5 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 4)
        self.res_6 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 4)
        self.res_7 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 8)
        self.res_8 = resnet_block(self.img_ef_dim * 8, self.img_ef_dim * 8)
        self.conv_9 = nn.Conv2d(self.img_ef_dim * 8, self.img_ef_dim * 8, 4, stride=2, padding=1, bias=False)
        self.bn_9 = nn.BatchNorm2d(self.img_ef_dim * 8)
        self.conv_10 = nn.Conv2d(self.img_ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_9.weight)
        nn.init.xavier_uniform_(self.conv_10.weight)

    def forward(self, view, is_training=False):
        layer_0 = self.bn_0(self.conv_0(1 - view))
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.02, inplace=True)

        layer_1 = self.res_1(layer_0, is_training=is_training)
        layer_2 = self.res_2(layer_1, is_training=is_training)

        layer_3 = self.res_3(layer_2, is_training=is_training)
        layer_4 = self.res_4(layer_3, is_training=is_training)

        layer_5 = self.res_5(layer_4, is_training=is_training)
        layer_6 = self.res_6(layer_5, is_training=is_training)

        layer_7 = self.res_7(layer_6, is_training=is_training)
        layer_8 = self.res_8(layer_7, is_training=is_training)

        layer_9 = self.bn_9(self.conv_9(layer_8))
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.02, inplace=True)

        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1, self.z_dim)
        layer_10 = torch.sigmoid(layer_10)

        return layer_10


class BaseModel(object):
    def __init__(self, config):

        #keep track of if the last checkpoint was loaded.
        self.checkpoint_loaded = False

        # progressive training
        # 1-- (16, 16*16*16)
        # 2-- (32, 16*16*16)
        # 3-- (64, 16*16*16*4)
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size == 16:
            self.load_point_batch_size = 16 * 16 * 16
            self.point_batch_size = 16 * 16 * 16
            self.shape_batch_size = 32
        elif self.sample_vox_size == 32:
            self.load_point_batch_size = 16 * 16 * 16
            self.point_batch_size = 16 * 16 * 16
            self.shape_batch_size = 32
        elif self.sample_vox_size == 64:
            self.load_point_batch_size = 16 * 16 * 16 * 4
            self.point_batch_size = 16 * 16 * 16
            self.shape_batch_size = 32
        self.input_size = 64  # input voxel grid size

        self.gf_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # keep everything a power of 2
        self.cell_grid_size = 4
        self.frame_grid_size = 64
        self.real_size = self.cell_grid_size * self.frame_grid_size  # =256, output point-value voxel grid size in testing
        # TODO revert back to 32
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.test_point_batch_size = self.test_size * self.test_size * self.test_size  # do not change

        self.build_training_coords()
        self.build_sampling_coords()

        self.sampling_threshold = 0.5  # final marching cubes threshold

    def build_training_coords(self):
        # get coords for training
        # These are the coordinates used to sample the grid
        dima = self.test_size
        dim = self.frame_grid_size
        self.aux_x = np.zeros([dima, dima, dima], np.uint8)
        self.aux_y = np.zeros([dima, dima, dima], np.uint8)
        self.aux_z = np.zeros([dima, dima, dima], np.uint8)
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i, j, k] = i * multiplier
                    self.aux_y[i, j, k] = j * multiplier
                    self.aux_z[i, j, k] = k * multiplier
        self.coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        self.coords = (self.coords.astype(np.float32) + 0.5) / dim - 0.5
        self.coords = np.reshape(self.coords, [multiplier3, self.test_point_batch_size, 3])
        self.coords = torch.from_numpy(self.coords)
        self.coords = self.coords.to(self.device)

    def build_sampling_coords(self):
        # get coords for testing
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k
        self.cell_coords = (self.cell_coords.astype(np.float32) + 0.5) / self.real_size - 0.5
        self.cell_coords = np.reshape(self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])

    def cv2_image_transform(self, img):
        '''
        Basic image transform used as input to IM_SVR

        :param img:
        :return:
        '''
        imgo = img[:, :, :3]
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        imga = (img[:, :, 3]) / 255.0
        img_out = imgo * imga + 255 * (1 - imga)
        img_out = np.round(img_out).astype(np.uint8)
        return img_out
