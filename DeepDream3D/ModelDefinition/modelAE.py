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


class im_network(nn.Module):
    def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
        super(im_network, self).__init__()
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.encoder = base_model.encoder(self.ef_dim, self.z_dim)
        self.generator = base_model.generator(self.z_dim, self.point_dim, self.gf_dim)

    def forward(self, inputs, z_vector, point_coord, is_training=False):
        if is_training:
            z_vector = self.encoder(inputs, is_training=is_training)
            net_out = self.generator(point_coord, z_vector, is_training=is_training)
        else:
            if inputs is not None:
                z_vector = self.encoder(inputs, is_training=is_training)
            if z_vector is not None and point_coord is not None:
                net_out = self.generator(point_coord, z_vector, is_training=is_training)
            else:
                net_out = None

        return z_vector, net_out


# pytorch 1.2.0 implementation

class IM_AE(base_model.BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.ef_dim = 32

        # load data
        #TODO: uncomment this
        self.load_data()

        # build model
        self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.im_network.to(self.device)
        # print params
        # for param_tensor in self.im_network.state_dict():
        #	print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())

        self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate,
                                           betas=(config.beta1, 0.999))

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        # loss
        def network_loss(G, point_value):
            return torch.mean((G - point_value) ** 2)

        self.loss = network_loss

    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

    def load_data(self):
        data_hdf5_name = self.data_dir + '/' + self.dataset_load + '.hdf5'
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            self.data_points = (data_dict['points_' + str(self.sample_vox_size)][:].astype(
                np.float32) + 0.5) / 256 - 0.5
            self.data_values = data_dict['values_' + str(self.sample_vox_size)][:].astype(np.float32)
            self.data_voxels = data_dict['voxels'][:]
            # reshape to NCHW
            self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.input_size, self.input_size, self.input_size])
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)

    def load_checkpoint(self):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
            return
        else:
            print(" [!] Load failed...")
            exit(0)


    def train(self, config):
        # load previous checkpoint
        self.load_checkpoint()

        shape_num = len(self.data_voxels)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0
        training_epoch = config.epoch + int(config.iteration / shape_num)
        batch_num = int(shape_num / self.shape_batch_size)
        point_batch_num = int(self.load_point_batch_size / self.point_batch_size)

        for epoch in range(0, training_epoch):
            self.im_network.train()
            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_num = 0
            for idx in range(batch_num):
                dxb = batch_index_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size]
                batch_voxels = self.data_voxels[dxb].astype(np.float32)
                if point_batch_num == 1:
                    point_coord = self.data_points[dxb]
                    point_value = self.data_values[dxb]
                else:
                    which_batch = np.random.randint(point_batch_num)
                    point_coord = self.data_points[dxb,
                                  which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]
                    point_value = self.data_values[dxb,
                                  which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]

                batch_voxels = torch.from_numpy(batch_voxels)
                point_coord = torch.from_numpy(point_coord)
                point_value = torch.from_numpy(point_value)

                batch_voxels = batch_voxels.to(self.device)
                point_coord = point_coord.to(self.device)
                point_value = point_value.to(self.device)

                self.im_network.zero_grad()
                _, net_out = self.im_network(batch_voxels, None, point_coord, is_training=True)
                errSP = self.loss(net_out, point_value)

                errSP.backward()
                self.optimizer.step()

                avg_loss_sp += errSP.item()
                avg_num += 1
            print(str(self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (
                epoch, training_epoch, time.time() - start_time, avg_loss_sp / avg_num))
            if epoch % 10 == 9:
                self.test_1(config, "train_" + str(self.sample_vox_size) + "_" + str(epoch))
            if epoch % 20 == 19:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path,
                                        self.checkpoint_name + str(self.sample_vox_size) + "-" + str(epoch) + ".pth")
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
        save_dir = os.path.join(self.checkpoint_path,
                                self.checkpoint_name + str(self.sample_vox_size) + "-" + str(epoch) + ".pth")
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
        t = np.random.randint(len(self.data_voxels))
        model_float = np.zeros([self.frame_grid_size + 2, self.frame_grid_size + 2, self.frame_grid_size + 2],
                               np.float32)
        batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
        batch_voxels = torch.from_numpy(batch_voxels)
        batch_voxels = batch_voxels.to(self.device)
        z_vector, _ = self.im_network(batch_voxels, None, None, is_training=False)
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
        # load previous checkpoint
        self.load_checkpoint()

        self.im_network.eval()
        for t in range(config.start, min(len(self.data_voxels), config.end)):
            batch_voxels_ = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels_)
            batch_voxels = batch_voxels.to(self.device)
            model_z, _ = self.im_network(batch_voxels, None, None, is_training=False)
            model_float = self.z2voxel(model_z)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # vertices = self.optimize_mesh(vertices,model_z)
            write_ply_triangle(config.sample_dir + "/" + str(t) + "_vox.ply", vertices, triangles)

            print("[sample]")

    # output shape as ply and point cloud as ply
    def test_mesh_point(self, config):
        # load previous checkpoint
        # This checkpoint file records the most recent checkpoint.. otherwise there is no record.
        self.load_checkpoint()

        self.im_network.eval()
        for t in range(config.start, min(len(self.data_voxels), config.end)):
            batch_voxels_ = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels_)
            batch_voxels = batch_voxels.to(self.device)
            model_z, _ = self.im_network(batch_voxels, None, None, is_training=False)
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

    def get_z(self, config):
        # load previous checkpoint
        self.load_checkpoint()

        hdf5_path = self.checkpoint_dir + '/' + self.model_dir + '/' + self.dataset_name + '_train_z.hdf5'
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num, self.z_dim], np.float32)

        self.im_network.eval()
        print(shape_num)
        for t in range(shape_num):
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            out_z, _ = self.im_network(batch_voxels, None, None, is_training=False)
            hdf5_file["zs"][t:t + 1, :] = out_z.detach().cpu().numpy()

        hdf5_file.close()
        print("[z]")

    def test_z(self, config, batch_z, dim):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        for t in range(batch_z.shape[0]):
            model_z = batch_z[t:t + 1]
            model_z = torch.from_numpy(model_z)
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
            write_ply_triangle(config.sample_dir + "/" + "out" + str(t) + ".ply", vertices, triangles)

            print("[sample Z]")
