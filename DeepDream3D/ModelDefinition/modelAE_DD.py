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

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

import mcubes

from .utils import *
from .modelAE import IM_AE


class IM_AE_DD(IM_AE):
    def __init__(self, config):
        super().__init__(config)

    def get_activation(self, output_list):
        '''
        A wrapper function to establish the forward hook

        :param out:
        :return:
        '''

        def hook(model, input, output):
            output_list[0] = output

        return hook

    def get_zvec(self, z_num):
        if z_num < len(self.data_voxels):
            batch_voxels = self.data_voxels[z_num:z_num + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            z_vec_, _ = self.im_network(batch_voxels, None, None, is_training=False)
            z_vec = z_vec_.detach().cpu().numpy()

            return (z_vec)

        else:
            print("z_num not a valid number")

    # output shape as ply
    def create_model_mesh(self, model_z, name: str):
        # TODO: uncomment load checkpoint
        # load previous checkpoint
        self.load_checkpoint()

        self.im_network.eval()
        model_float = self.z2voxel(model_z)

        print('model_float shape')
        print(model_float.shape)

        # This transform necessary to accommodate coordinate transform induced in marching cubes
        model_float = np.flip(np.transpose(model_float, (2, 1, 0)), 0)

        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
        # vertices = self.optimize_mesh(vertices,model_z)
        full_path = self.result_dir + "/" + name + "_vox.ply"
        write_ply_triangle(full_path, vertices, triangles)

        print("created .ply for image {}", name)

        return full_path

    def interpolate_z(self, config):
        '''
        A method to create the meshes from latent z vectors linearly interpolated between two vectors.

        :param config:
        :return:
        '''

        # load previous checkpoint
        self.load_checkpoint()

        z1 = int(config.interpol_z1)
        z2 = int(config.interpol_z2)
        interpol_steps = int(config.interpol_steps)
        result_base_directory = config.interpol_directory
        self.result_dir_name = 'interpol_' + str(z1) + '_' + str(z2)
        self.result_dir = result_base_directory + '/' + self.result_dir_name
        print(self.result_dir)

        # Create output directory
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
            print('creating directory ' + self.result_dir)

        # get latent vectors from hdf5
        # hdf5_path = self.checkpoint_dir + '/' + self.model_dir + '/' + self.dataset_name + '_train_z.hdf5'
        # hdf5_file = h5py.File(hdf5_path, mode='r')
        # num_z = hdf5_file["zs"].shape[0]

        # get the z vectors via forward pass through encoder
        z1_vec = self.get_zvec(z1)
        z2_vec = self.get_zvec(z2)

        # compute linear interpolation between vectors
        fraction = np.linspace(0, 1, interpol_steps)
        interpolated_z = np.multiply.outer(np.ones_like(fraction), z1_vec) + np.multiply.outer(fraction,
                                                                                               z2_vec - z1_vec)
        interpolated_z = interpolated_z.astype(np.float64)

        self.out_filenames = []
        for z_index in np.arange(interpol_steps):
            self.out_filenames.append(self.result_dir + "/" + "out_{:.2f}.ply".format(fraction[z_index]))

        for z_index in np.arange(interpol_steps):
            start_time = time.time()
            model_z = interpolated_z[z_index:z_index + 1].astype(np.float64)
            # print('current latent vector:')
            # print(model_z.shape)

            model_z = torch.from_numpy(model_z).float()
            model_z = model_z.to(self.device)
            model_float = self.z2voxel(model_z)
            # img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
            # img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
            # img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
            # cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
            # cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
            # cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # vertices = self.optimize_mesh(vertices,model_z)
            write_ply_triangle(self.result_dir + "/" + "out_{:.2f}.ply".format(fraction[z_index]), vertices, triangles)

            end_time = time.time() - start_time
            print("computed interpolation {} in {} seconds".format(z_index, end_time))

    def latent_gradient(self, z_base, z_target, step, config):
        '''
        Computes the average derivative evaluated over the sample field

        :param self:
        :param z1:
        :param z2:
        :param step_size:
        :return:
        '''

        # make sure gradients are set to zero to begin with on each pass.
        self.im_network.zero_grad()

        # print(z_base.grad)
        # create a numpy array to store accumulated derivatives
        # accumulated_grad = np.zeros(self.z_dim, dtype=np.float64)

        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size  # coarse model evaluation

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []

        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values: this gets frame voxels that contain above threshold values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)

            # TODO: remove dummy data
            _, model_out_ = self.im_network(None, z_base, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            # model_out = np.random.random(size=[1, 4096, 1])

            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

        # get queue and fill up ones
        # This puts together a que of frame points to compute values.
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1] = 1.0

        print("running queue:", len(queue))
        que_len = len(queue)
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0
        # run queue
        iter_num = 0
        total_iter = que_len // cell_batch_num
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)

            # Call the model on the target to get target encoder layer
            _, model_out_batch_ = self.im_network(None, z_target, cell_coords, is_training=False)
            style_activation = self.target_activation[0].clone().detach()
            # style_points_mask = model_out_batch_.detach().expand(-1, -1,
            #                                                     style_activation.size()[2]) > self.sampling_threshold

            style_points_mask = model_out_batch_.detach().expand(-1, -1, style_activation.size()[2])
            style_points_mask_above = style_points_mask > (1 - self.boundary) * self.sampling_threshold
            style_points_mask_below = style_points_mask > (1 + self.boundary) * self.sampling_threshold
            style_points_mask = torch.logical_and(style_points_mask_above, style_points_mask_below)

            # Call the model on the base to get base encoder layer, first set z_base to is_training = true
            _, model_out_batch_ = self.im_network(None, z_base, cell_coords, is_training=False)
            base_activation = self.target_activation[0]
            # base_points_mask = model_out_batch_.detach().expand(-1, -1,
            #                                                    base_activation.size()[2]) > self.sampling_threshold

            base_points_mask = model_out_batch_.detach().expand(-1, -1, base_activation.size()[2])
            base_points_mask_above = base_points_mask > (1 - self.boundary) * self.sampling_threshold
            base_points_mask_below = base_points_mask > (1 + self.boundary) * self.sampling_threshold
            base_points_mask = torch.logical_and(base_points_mask_above, base_points_mask_below)

            # Now compute the gradient
            # gradient_ = np.tanh(np.abs((style_activation + base_activation) / (style_activation - base_activation)))
            # gradient = torch.from_numpy(gradient_).to(self.device)

            # no need to track gradient for these operations
            '''
            with torch.no_grad():
                diff = torch.abs(style_activation - base_activation)
                mean = diff.mean()
                print(mean)
                shift_diff = diff - mean
                sigma2 = torch.pow(shift_diff, 2).mean()
                print(sigma2)
                mantissa = torch.pow(shift_diff / sigma2, 2) # divide by total number of points
                # print(mantissa)
                loss = torch.exp(-mantissa)
            '''

            # loss = base_activation / torch.abs((style_activation - base_activation) + .001)
            # loss = style_activation - base_activation

            difference = style_activation - base_activation
            # loss = torch.tanh(torch.sign(difference) / (torch.abs(difference) + .001))
            # loss = torch.sign(difference) / (torch.abs(difference) + .001)
            loss = difference.detach()

            # Union
            loss[torch.logical_not(torch.logical_or(style_points_mask, base_points_mask))] = 0

            # Intersection
            # loss[torch.logical_or(torch.logical_not(style_points_mask), torch.logical_not(base_points_mask))] = 0

            # loss[torch.logical_not(base_points_mask)] = 0
            # print(loss)

            base_activation.backward(loss)

            # print(z_base.grad)

            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, 0]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf: continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf: continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf: continue
                                if (frame_flag[pi, pj, pk] == 0):
                                    frame_flag[pi, pj, pk] = 1
                                    queue.append((pi, pj, pk))

            print('iteration {} of {} completed'.format(iter_num, total_iter))
            iter_num += 1

        # This transform nescessary to accomodate coordinate transform induced in marching cubes
        model_float = np.flip(np.transpose(model_float, (2, 1, 0)), 0)

        # TODO: uncomment file writing
        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
        # vertices = self.optimize_mesh(vertices, model_z)
        print('writing file: ' + self.result_dir + "/" + str(step) + "_vox.ply")
        write_ply_triangle(self.result_dir + "/" + str(step) + "_vox.ply", vertices, triangles)

        # update base sample points for next iteration.
        # base_mesh = Meshes(verts=[vertices])
        # self.base_sample_points = sample_points_from_meshes(meshes=base_mesh, num_samples=int(1e5))

        return z_base.grad

    def build_gram_matrix(self,
                          z_style: torch.Tensor,
                          z_target: torch.Tensor,
                          config):
        '''
        Computes the gram correlation matrix of z_style and z_transfer
        or
        accumulates the style loss of the loss layer and the gram matrix.

        :param z_base: base model latent vector
        :param z_style: style model latent vector
        :param z_transfer: transfer style model
        :param step: iteration step
        :param plot: whether to plot
        :param flag: whether to compute the content (true) or style (false) error
        :param config: configuration parameters
        :return:
        '''

        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size  # coarse model evaluation

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []

        style_gram_record = []
        target_gram_record = []

        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        volume_z = z_target

        # get frame grid values: this gets frame voxels that contain above threshold values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)

            _, model_out_ = self.im_network(None, volume_z, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            # TODO: remove dummy data
            # model_out = np.random.random(size=[1, 4096, 1])

            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

        # get queue and fill up ones
        # This puts together a que of frame points to compute values.
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1] = 1.0

        print("running queue:", len(queue))
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0

        # run queue
        iter_num = 0
        points_num = 0
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            num_points = cell_coords.shape[0]  # record the number of points
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)

            # Call the model on the target to get target encoder layer
            _, model_out_batch_ = self.im_network(None, z_style, cell_coords, is_training=False)
            style_gram_activation = self.gram_activation[0].clone().detach().squeeze()

            style_gram_record.append(style_gram_activation)
            # print(style_gram_record.shape)

            # Call the model on the target to get target encoder layer
            _, model_out_batch_ = self.im_network(None, z_target, cell_coords, is_training=False)
            target_gram_activation = self.gram_activation[0].clone().detach().squeeze()

            target_gram_record.append(target_gram_activation)

            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, 0]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf: continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf: continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf: continue
                                if (frame_flag[pi, pj, pk] == 0):
                                    frame_flag[pi, pj, pk] = 1
                                    queue.append((pi, pj, pk))

            # print('iteration {} of {} completed'.format(iter_num, total_iter))
            points_num += num_points
            # print(points_num)
            iter_num += 1

        # Now compute the global gram correlation matrix
        style_gram_record = torch.cat(style_gram_record, dim=0)
        #del style_gram_record
        #style_gram_record = style_gram_record_
        print(style_gram_record.shape)
        target_gram_record = torch.cat(target_gram_record, dim=0)
        # del target_gram_record
        # target_gram_record = target_gram_record_
        target_gram_record.requires_grad_()
        target_gram_matrix = torch.matmul(torch.transpose(target_gram_record, 0, 1), target_gram_record)
        style_gram_matrix = torch.matmul(torch.transpose(style_gram_record, 0, 1), style_gram_record)
        # print(style_gram_matrix.shape)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(target_gram_matrix, style_gram_matrix)
        loss.backward(retain_graph=False)

        # catch memory leak
        # if isinstance(self.target_gram_loss, torch.Tensor):
        #     del self.target_gram_loss

        # self.target_gram_loss.data = target_gram_record.grad.detach().clone()

        # clean up memory explicitely
        # del loss, target_gram_matrix, style_gram_matrix, target_gram_record, style_gram_record
        print('gram_record grad:')
        print(target_gram_record.grad.norm())
        gram_out = target_gram_record.grad.detach().clone()
        # gram_out = gram_out / gram_out.norm()
        # print(gram_out)
        return gram_out

    def latent_style_transfer(self,
                              z_base: torch.Tensor,
                              z_style: torch.Tensor,
                              z_target: torch.Tensor,
                              gram_loss: torch.Tensor,
                              step: int,
                              plot: bool,
                              flag: bool,
                              config):
        '''
        Computes the gram correlation matrix of z_style and z_transfer
        or
        accumulates the style loss of the loss layer and the gram matrix.

        :param z_base: base model latent vector
        :param z_style: style model latent vector
        :param z_transfer: transfer style model
        :param step: iteration step
        :param plot: whether to plot
        :param flag: whether to compute the content (true) or style (false) error
        :param config: configuration parameters
        :return:
        '''

        # scaling variables for style (alpha) and activation (beta) loss.
        alpha = 1.0
        beta = config.beta

        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size  # coarse model evaluation

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []

        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values: this gets frame voxels that contain above threshold values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)

            _, model_out_ = self.im_network(None, z_target, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            # TODO: remove dummy data
            # model_out = np.random.random(size=[1, 4096, 1])

            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

        # get queue and fill up ones
        # This puts together a que of frame points to compute values.
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1] = 1.0

        print("running queue:", len(queue))
        que_len = len(queue)
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0

        # run queue
        iter_num = 0
        points_num = 0
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            num_points = cell_coords.shape[0]  # record the number of points
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)

            # Call the model on the base to get base encoder layer
            _, model_out_batch_ = self.im_network(None, z_base, cell_coords, is_training=False)
            base_activation = self.target_activation[0].clone().detach()

            # Call the model on the base to get base encoder layer
            _, model_out_batch_ = self.im_network(None, z_target, cell_coords, is_training=False)
            target_activation = self.target_activation[0]

            # Now compute the gradient
            mse_loss = torch.nn.MSELoss()
            L2_base = alpha * mse_loss(target_activation, base_activation)
            L2_base.backward(retain_graph=False)

            # Call the model on the target to get base target layer
            _, model_out_batch_ = self.im_network(None, z_target, cell_coords, is_training=False)
            target_gram_activation = self.gram_activation[0]


            L2_style = target_gram_activation
            L2_style.backward(beta * gram_loss[points_num:(points_num + num_points), :].unsqueeze(0),
                              retain_graph=False)

            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, 0]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf: continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf: continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf: continue
                                if (frame_flag[pi, pj, pk] == 0):
                                    frame_flag[pi, pj, pk] = 1
                                    queue.append((pi, pj, pk))

            # print('iteration {} of {} completed'.format(iter_num, total_iter))
            points_num += num_points
            iter_num += 1

        if plot:
            # This transform nescessary to accomodate coordinate transform induced in marching cubes
            model_float = np.flip(np.transpose(model_float, (2, 1, 0)), 0)

            # TODO: uncomment file writing
            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # vertices = self.optimize_mesh(vertices, model_z)
            print('writing file: ' + self.result_dir + "/" + str(step) + "_vox.ply")
            write_ply_triangle(self.result_dir + "/" + str(step) + "_vox.ply", vertices, triangles)

        # return accumulated gradient of z_transfer
        return z_target.grad.clone().detach()

    def deep_dream(self, config):
        '''
        This function applies a gradient step to the latent vector z1 based on the difference of the activations
        at layer x between z1 and z2, computed at surface points on z1

        :param sefl:
        :param config:
        :return:
        '''

        # TODO: uncomment checkpoint load
        # load previous checkpoint
        self.load_checkpoint()

        # set the dreaming rate and boundary size
        self.dream_rate = config.dream_rate
        self.boundary = .1

        # setup style scaling
        self.base_scale = 1
        self.style_scale = 1

        # Set up forward hook to pull values
        self.layer_num = config.layer_num
        # list index includes as zero entry the generator module itself. (layer indices start from 1)
        num_model_layers = len(list(self.im_network.generator.named_modules())) - 1
        if self.layer_num >= num_model_layers:
            print('Layer number is too large: select layer numbers from 1 to {}'.format(num_model_layers))
            exit(0)

        # this is the way to get the actual model variable
        self.target_layer = list(self.im_network.generator.named_modules())[self.layer_num][1]
        self.target_activation = [None]
        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))
        self.target_gram_loss = None

        # choose the first layer as the style layer to produce the gram matrices
        self.gram_layer = list(self.im_network.generator.named_modules())[self.layer_num][1]
        self.gram_activation = [None]
        self.gram_layer.register_forward_hook(self.get_activation(self.gram_activation))
        target_gram_grad = None

        # get config values
        z1 = int(config.interpol_z1)
        z2 = int(config.interpol_z2)

        interpol_steps = int(config.interpol_steps)
        result_base_directory = config.interpol_directory
        result_dir_name = 'DeepDream_' + str(z1) + '_' + str(z2) + '_layer_' + str(self.layer_num)
        self.result_dir = result_base_directory + '/' + result_dir_name

        # Create output directory
        # TODO: re-create directory
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
            print('creating directory ' + self.result_dir)

        # get z vectors from forward pass of encoder

        # TODO: comment out dummy data
        z1_vec_ = torch.from_numpy(self.get_zvec(z1)).float().to(self.device)
        # z1_vec_copy = z1_vec_.clone()
        # z1_vec_ = torch.from_numpy(np.random.random(size=[256])).type(torch.FloatTensor).to(self.device)
        z1_base = z1_vec_.detach().clone()
        # z1_vec = torch.rand(256, requires_grad=True, device=self.device)
        z1_vec = torch.autograd.Variable(z1_vec_, requires_grad=True)

        self.create_model_mesh(z1_base, 'base_model')

        # TODO: comment out dummy data
        z2_vec = torch.from_numpy(self.get_zvec(z2)).to(self.device)
        # z2_vec = torch.from_numpy(np.random.random(size=[256])).type(torch.FloatTensor).to(self.device)

        self.create_model_mesh(z2_vec, 'style_model')


        for step in range(interpol_steps):
            start_time = time.perf_counter()

            if step < 2:
                # keep the style scaling at 1 for the first few iterations.
                self.base_scale = 1
                self.style_scale = 1

            # accumulate the gradient below

            # zero out the gradient on each step
            self.im_network.zero_grad()
            if isinstance(z1_vec.grad, torch.Tensor):
                print('reset leaf gradient to zero.')
                z1_vec.grad.zero_()

            # compute non-local style loss

            # must delete large style vector between runs, otherwise memory leak
            if target_gram_grad is not None:
                del target_gram_grad
                torch.cuda.empty_cache() # must completely free memory from torch to avoid segmentation problems

            target_gram_grad = self.build_gram_matrix(z_style=z2_vec,
                                                      z_target=z1_vec,
                                                      config=config)

            # zero out the gradient on each step
            self.im_network.zero_grad()

            # accumulate style and content loss via backprop
            style_loss = self.latent_style_transfer(z_base=z1_base,
                                                    z_style=z2_vec,
                                                    z_target=z1_vec,
                                                    gram_loss=target_gram_grad,
                                                    step=step,
                                                    plot=False,
                                                    flag=False,
                                                    config=config)



            # to ensure the graph does not keep a gradient record, use no_grad
            with torch.no_grad():
                grad_step = style_loss / style_loss.norm() * self.dream_rate

                print(grad_step)
                z1_vec.data -= grad_step
                z1_vec.data = torch.clamp(z1_vec.data, 0, 1)

            self.create_model_mesh(z1_vec, str(step))

            end_time = time.perf_counter()
            print('Completed dream {} in {} seconds'.format(step, end_time - start_time))

        print('Done Dreaming..')
