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

import DeepDream3D.ModelDefinition.base_model as base_model

# pytorch 1.2.0 implementation

class im_network(nn.Module):
    def __init__(self, img_ef_dim, gf_dim, z_dim, point_dim):
        super(im_network, self).__init__()
        self.img_ef_dim = img_ef_dim
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.img_encoder = base_model.img_encoder(self.img_ef_dim, self.z_dim)
        self.generator = base_model.generator(self.z_dim, self.point_dim, self.gf_dim)

    def forward(self, inputs, z_vector, point_coord, is_training=False):
        if is_training:
            z_vector = self.img_encoder(inputs, is_training=is_training)
            net_out = None
        else:
            if inputs is not None:
                z_vector = self.img_encoder(inputs, is_training=is_training)
            if z_vector is not None and point_coord is not None:
                net_out = self.generator(point_coord, z_vector, is_training=is_training)
            else:
                net_out = None

        return z_vector, net_out


class IM_SVR(base_model.BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.input_size = 64  # input voxel grid size

        self.img_ef_dim = 64

        # actual batch size
        self.shape_batch_size = 64

        self.view_size = 137
        self.crop_size = 128
        self.view_num = 24
        self.test_idx = 23

        # TODO: comment in data loading
        # load the data
        # self.load_data(config)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


        # build model
        self.im_network = im_network(self.img_ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.im_network.to(self.device)
        # print params
        # for param_tensor in self.im_network.state_dict():
        #	print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.im_network.img_encoder.parameters(), lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))


        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 10
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'IM_SVR.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        self.checkpoint_AE_path = os.path.join(self.checkpoint_dir, self.modelAE_dir)
        self.checkpoint_AE_name = 'IM_AE.model'

        # loss
        def network_loss(pred_z, gt_z):
            return torch.mean((pred_z - gt_z) ** 2)

        self.loss = network_loss

    @property
    def model_dir(self):
        return "{}_svr_{}".format(
            self.dataset_name, self.crop_size)

    @property
    def modelAE_dir(self):
        return "{}_ae_{}".format(
            self.dataset_name, self.input_size)

    def load_data(self, config):
        self.crop_edge = self.view_size - self.crop_size
        data_hdf5_name = self.data_dir + '/' + self.dataset_load + '.hdf5'
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            offset_x = int(self.crop_edge / 2)
            offset_y = int(self.crop_edge / 2)
            # reshape to NCHW
            self.data_pixels = np.reshape(
                data_dict['pixels'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
                [-1, self.view_num, 1, self.crop_size, self.crop_size])
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)
        if config.train:
            dataz_hdf5_name = self.checkpoint_dir + '/' + self.modelAE_dir + '/' + self.dataset_name + '_train_z.hdf5'
            if os.path.exists(dataz_hdf5_name):
                dataz_dict = h5py.File(dataz_hdf5_name, 'r')
                self.data_zs = dataz_dict['zs'][:]
            else:
                print("error: cannot load " + dataz_hdf5_name)
                exit(0)
            if len(self.data_zs) != len(self.data_pixels):
                print("error: len(self.data_zs) != len(self.data_pixels)")
                print(len(self.data_zs), len(self.data_pixels))
                exit(0)

    def load_checkpoint(self):
        # load previous checkpoint
        if not self.checkpoint_loaded:
            if os.path.exists(self.checkpoint_dir):
                self.im_network.load_state_dict(torch.load(self.checkpoint_dir))
                print(" [*] Load SUCCESS")
                self.checkpoint_loaded = True
                return
            else:
                print(" [!] Load failed...")
                exit(0)

    def train(self, config):

        # load full data
        self.load_data(config)

        # load AE weights
        checkpoint_txt = os.path.join(self.checkpoint_AE_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir), strict=False)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(-1)

        shape_num = len(self.data_pixels)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0
        training_epoch = config.epoch + int(config.iteration / shape_num)
        batch_num = int(shape_num / self.shape_batch_size)

        self.im_network.train()
        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)
            avg_loss = 0
            avg_num = 0
            for idx in range(batch_num):
                dxb = batch_index_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size]

                which_view = np.random.randint(self.view_num)
                batch_view = self.data_pixels[dxb, which_view].astype(np.float32) / 255.0
                batch_zs = self.data_zs[dxb]

                batch_view = torch.from_numpy(batch_view)
                batch_zs = torch.from_numpy(batch_zs)

                batch_view = batch_view.to(self.device)
                batch_zs = batch_zs.to(self.device)

                self.im_network.zero_grad()
                z_vector, _ = self.im_network(batch_view, None, None, is_training=True)
                err = self.loss(z_vector, batch_zs)

                err.backward()
                self.optimizer.step()

                avg_loss += err
                avg_num += 1
            print("Epoch: [%2d/%2d] time: %4.4f, loss: %.8f" % (
            epoch, training_epoch, time.time() - start_time, avg_loss / avg_num))
            if epoch % 10 == 9:
                self.test_1(config, "train_" + str(epoch))
            if epoch % 100 == 99:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path, self.checkpoint_name + "-" + str(epoch) + ".pth")
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
                # delete checkpoint
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                # save checkpoint
                torch.save(self.im_network.state_dict(), save_dir)
                # update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                # write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                fout = open(checkpoint_txt, 'w')
                for i in range(self.max_to_keep):
                    pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
                    if self.checkpoint_manager_list[pointer] is not None:
                        fout.write(self.checkpoint_manager_list[pointer] + "\n")
                fout.close()

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path, self.checkpoint_name + "-" + str(training_epoch) + ".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save(self.im_network.state_dict(), save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()

    def test_1(self, config, name):
        multiplier = int(self.frame_grid_size / self.test_size)
        multiplier2 = multiplier * multiplier
        self.im_network.eval()
        t = np.random.randint(len(self.data_pixels))
        model_float = np.zeros([self.frame_grid_size + 2, self.frame_grid_size + 2, self.frame_grid_size + 2],
                               np.float32)
        batch_view = self.data_pixels[t:t + 1, self.test_idx].astype(np.float32) / 255.0
        batch_view = torch.from_numpy(batch_view)
        batch_view = batch_view.to(self.device)
        z_vector, _ = self.im_network(batch_view, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i * multiplier2 + j * multiplier + k
                    point_coord = self.coords[minib:minib + 1]
                    _, net_out = self.im_network(None, z_vector, point_coord, is_training=False)
                    # net_out = torch.clamp(net_out, min=0, max=1)
                    model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(
                        net_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])

        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.frame_grid_size - 0.5
        # output ply sum
        write_ply_triangle(config.sample_dir + "/" + name + ".ply", vertices, triangles)
        print("[sample]")

    def z2voxel(self, z):
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []

        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)
            _, model_out_ = self.im_network(None, z, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

        # get queue and fill up ones
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


            _, model_out_batch_ = self.im_network(None, z, cell_coords, is_training=False)
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
        return model_float

    # may introduce foldovers
    def optimize_mesh(self, vertices, z, iteration=3):
        new_vertices = np.copy(vertices)

        new_vertices_ = np.expand_dims(new_vertices, axis=0)
        new_vertices_ = torch.from_numpy(new_vertices_)
        new_vertices_ = new_vertices_.to(self.device)
        _, new_v_out_ = self.im_network(None, z, new_vertices_, is_training=False)
        new_v_out = new_v_out_.detach().cpu().numpy()[0]

        for iter in range(iteration):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0: continue
                        offset = np.array([[i, j, k]], np.float32) / (self.real_size * 6 * 2 ** iter)
                        current_vertices = vertices + offset
                        current_vertices_ = np.expand_dims(current_vertices, axis=0)
                        current_vertices_ = torch.from_numpy(current_vertices_)
                        current_vertices_ = current_vertices_.to(self.device)
                        _, current_v_out_ = self.im_network(None, z, current_vertices_, is_training=False)
                        current_v_out = current_v_out_.detach().cpu().numpy()[0]
                        keep_flag = abs(current_v_out - self.sampling_threshold) < abs(
                            new_v_out - self.sampling_threshold)
                        keep_flag = keep_flag.astype(np.float32)
                        new_vertices = current_vertices * keep_flag + new_vertices * (1 - keep_flag)
                        new_v_out = current_v_out * keep_flag + new_v_out * (1 - keep_flag)
            vertices = new_vertices

        return vertices

    # output shape as ply
    def test_mesh(self, config):

        # load full data
        self.load_data(config)

        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        self.im_network.eval()
        for t in range(config.start, min(len(self.data_pixels), config.end)):
            batch_view_ = self.data_pixels[t:t + 1, self.test_idx].astype(np.float32) / 255.0
            batch_view = torch.from_numpy(batch_view_)
            batch_view = batch_view.to(self.device)
            model_z, _ = self.im_network(batch_view, None, None, is_training=False)
            model_float = self.z2voxel(model_z)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # vertices = self.optimize_mesh(vertices,model_z)
            write_ply_triangle(config.sample_dir + "/" + str(t) + "_vox.ply", vertices, triangles)

            print("[sample]")

    # output shape as ply and point cloud as ply
    def test_mesh_point(self, config):

        # load full data
        self.load_data(config)

        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        self.im_network.eval()
        for t in range(config.start, min(len(self.data_pixels), config.end)):
            batch_view_ = self.data_pixels[t:t + 1, self.test_idx].astype(np.float32) / 255.0
            batch_view = torch.from_numpy(batch_view_)
            batch_view = batch_view.to(self.device)
            model_z, _ = self.im_network(batch_view, None, None, is_training=False)
            model_float = self.z2voxel(model_z)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # vertices = self.optimize_mesh(vertices,model_z)
            write_ply_triangle(config.sample_dir + "/" + str(t) + "_vox.ply", vertices, triangles)

            print("[sample]")

            # sample surface points
            sampled_points_normals = sample_points_triangle(vertices, triangles, 4096)
            np.random.shuffle(sampled_points_normals)
            write_ply_point_normal(config.sample_dir + "/" + str(t) + "_pc.ply", sampled_points_normals)

            print("[sample]")
