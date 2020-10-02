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

from .utils import *
from .modelAE import IM_AE


class IM_AE_DD(IM_AE):
    def __init__(self, config):
        super().__init__(config)

    def get_activation(self, output_list):
        '''
        A wrapper function to establish teh forward hook

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

            return (z_num)

        else:
            print("z_num not a valid number")

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
        result_dir_name = 'interpol_' + str(z1) + '_' + str(z2)
        result_dir = config.interpol_directory + '/' + result_dir_name

        # Create output directory
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
            print('creating directory ' + result_dir)

        # get latent vectors from hdf5
        # hdf5_path = self.checkpoint_dir + '/' + self.model_dir + '/' + self.dataset_name + '_train_z.hdf5'
        # hdf5_file = h5py.File(hdf5_path, mode='r')
        # num_z = hdf5_file["zs"].shape[0]

        # get the z vectors via forward pass through encoder
        z1_vec = self.get_zvec(z1)
        z2_vec = self.get_zvec(z1)

        # compute linear interpolation between vectors
        fraction = np.linspace(0, 1, interpol_steps)
        interpolated_z = np.multiply.outer(np.ones_like(fraction), z1_vec) + np.multiply.outer(fraction,
                                                                                               z2_vec - z1_vec)
        interpolated_z = interpolated_z.astype(np.float64)

        for z_index in np.arange(interpol_steps):
            start_time = time.time()
            model_z = interpolated_z[z_index:z_index + 1].astype(np.float64)
            # print('current latent vector:')
            # print(model_z)

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
            write_ply_triangle(result_dir + "/" + "out_" + str(z_index) + ".ply", vertices, triangles)

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

        # make sure z_base will accumulate gradients
        z_base.requires_grad = True
        # make sure gradients are set to zero to begin with.
        self.im_network.zero_grad()

        # create a numpy array to store accumulated derivatives
        accumulated_grad = np.zeros(self.z_dim, dtype=np.float64)

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

            _, model_out_ = self.im_network(None, z_base, point_coord, is_training=False)
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
        total_iter = len(queue) // cell_batch_num
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
            style_activation = self.target_activation[0]

            # Call the model on the base to get base encoder layer, first set z_base to is_training = true
            _, model_out_batch_ = self.im_network(None, z_base, cell_coords, is_training=False)
            base_activation = self.target_activation[0]

            # Now compute the gradient
            # gradient_ = np.tanh(np.abs((style_activation + base_activation) / (style_activation - base_activation)))
            # gradient = torch.from_numpy(gradient_).to(self.device)
            loss = 1 - torch.exp(-torch.pow(style_activation - base_activation, 2))
            base_activation.backward(loss)

            # Store gradient
            # batch_grad = z_base.grad
            # accumulated_grad += batch_grad.detach().cpu().numpy() / que_len

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

        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
        # vertices = self.optimize_mesh(vertices,model_z)
        write_ply_triangle(config.sample_dir + "/" + str(step) + "_vox.ply", vertices, triangles)

        return z_base.grad

    def deep_dream(self, config):
        '''
        This function applies a gradient step to the latent vector z1 based on the difference of the activations
        at layer x between z1 and z2, computed at surface points on z1

        :param sefl:
        :param config:
        :return:
        '''

        # load previous checkpoint
        self.load_checkpoint()

        #
        self.dream_rate = config.dream_rate

        # Set up forward hook to pull values
        self.layer_num = config.layer_num
        # this is the way to get the actual model variable
        self.target_layer = list(self.im_network.generator.named_modules())[self.layer_num][1]
        self.target_activation = [None]

        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))

        # get config values
        z1 = int(config.interpol_z1)
        z2 = int(config.interpol_z2)
        interpol_steps = int(config.interpol_steps)
        result_base_directory = config.interpol_directory
        result_dir_name = 'DeepDream_' + str(z1) + '_' + str(z2)
        result_dir = result_base_directory + '/' + result_dir_name

        # Create output directory
        # TODO: re-create directory
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
            print('creating directory ' + result_dir)

        # get z vectors from forward pass of encoder

        # z1_vec = self.get_zvec(z1)
        z1_vec = torch.from_numpy(np.random.random(size=[256])).type(torch.FloatTensor).to(self.device)
        # z2_vec = self.get_zvec(z1)
        z2_vec = torch.from_numpy(np.random.random(size=[256])).type(torch.FloatTensor).to(self.device)

        for step in range(config.interpol_steps):
            start_time = time.perf_counter()
            # accumulate the gradient over the whole volume
            grad = self.latent_gradient(z1_vec, z2_vec, step, config)

            z1_vec += grad * self.dream_rate

            end_time = time.perf_counter()
            print('Completed dream {} in {} seconds'.format(step, end_time - start_time))

        print('Done Dreaming..')
