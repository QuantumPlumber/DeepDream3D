import os
import time
import math
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from pytorch3d.io import save_ply, save_obj, load_objs_as_meshes, load_obj, load_ply

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    Textures,
    TexturesVertex
)

import cv2

import mcubes
from typing import List

from ..preprocessing.utils import shapenet_cam_params

from .ShapeNetRendering import ShapeNetRendering

from .utils import *
from .modelSVR import IM_SVR


class IM_SVR_DD(IM_SVR):
    def __init__(self, config):
        super().__init__(config)

        self.shapenet_cam_params = shapenet_cam_params

    def load_data(self, config):
        '''
        Overrides base class method in order to only load data required for deep dreaming.
        :param config:
        :return:
        '''

        # get config values
        z_base = int(config.interpol_z1)
        z_target = int(config.interpol_z2)

        self.crop_edge = self.view_size - self.crop_size
        data_hdf5_name = self.data_dir + '/' + self.dataset_load + '.hdf5'
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            offset_x = int(self.crop_edge / 2)
            offset_y = int(self.crop_edge / 2)
            # reshape to NCHW

            # get the shape of the first two cropped pictures
            cropped_shape = np.reshape(
                data_dict['pixels'][0:2, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
                [-1, self.view_num, 1, self.crop_size, self.crop_size]).shape

            self.data_pixels = np.empty(shape=cropped_shape)

            # now grab only the data that is needed. This must be done iteratively or hdf5 can throw and error
            # (selection indices must be of increasing order only)
            for ind, z in enumerate([z_base, z_target]):
                self.data_pixels[ind, ...] = np.reshape(
                    data_dict['pixels'][z, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
                    [self.view_num, 1, self.crop_size, self.crop_size])
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)

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
        if z_num < len(self.data_pixels):
            batch_view = self.data_pixels[z_num:z_num + 1, self.test_idx].astype(np.float32) / 255.0
            batch_view = torch.from_numpy(batch_view)
            batch_view = batch_view.to(self.device)
            z_vec_, _ = self.im_network(batch_view, None, None, is_training=False)
            z_vec = z_vec_.detach().cpu().numpy()

            return (z_vec)

        else:
            print("z_num not a valid number")

    def create_saved_images(self, images, name):
        num_images = int(images.shape[0])
        cols = 3
        rows = -int(-num_images // cols)

        # convert back to grayscale
        rescale_images = images

        print(images.max())
        print(images.min())

        fig, axs = plt.subplots(nrows=rows,
                                ncols=cols,
                                sharex='all',
                                sharey='all',
                                figsize=(cols * 2, rows * 2),
                                gridspec_kw={'wspace': 0, 'hspace': 0}
                                )
        for ax, im in zip(axs.flatten(), range(num_images)):
            ax.imshow(rescale_images[im, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        plt.savefig(self.result_dir + '/' + name)

    # output shape as ply
    def create_model_mesh(self, batch_view, num, config):
        # TODO: uncomment load checkpoint
        # load previous checkpoint
        self.load_checkpoint()

        self.im_network.eval()
        model_z, _ = self.im_network(batch_view, None, None, is_training=False)
        model_float = self.z2voxel(model_z)

        print('model_float shape')
        print(model_float.shape)

        model_float = np.flip(np.transpose(model_float, (2, 1, 0)), 0)

        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
        # vertices = self.optimize_mesh(vertices,model_z)
        full_path = self.result_dir + "/" + str(num) + "_vox.ply"
        write_ply_triangle(full_path, vertices, triangles)

        print("created .ply for image {}".format(num))

        return full_path

    def cv2_image_transform(self, img):
        '''
        Basic image transform used as input to IM_SVR

        :param img:
        :return:
        '''

        '''
        imgo = img[:, :, :3] * 255
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        imga = (img[:, :, 3])
        img_out = imgo * imga + 255.0 * (1 - imga)
        img_out = np.round(img_out).astype(np.uint8)
        '''
        img[:, :, :3] = img[:, :, :3] * 255
        img_out = cv2.cvtColor(img[:, :, :], cv2.COLOR_BGRA2GRAY) / 255
        # img_out = np.round(img_out).astype(np.uint8)
        # print(img_out.shape)

        img_out = cv2.resize(img_out, dsize=(128, 128))

        img_out = img_out[np.newaxis, :, :].astype(np.float32)

        return img_out

    def annealing_view(self, ply_path):
        # param_num = self.test_idx
        param_num = 7

        # get image transform
        R, T = look_at_view_transform(
            dist=shapenet_cam_params["distance"][param_num] * 3,
            elev=shapenet_cam_params["elevation"][param_num],
            azim=shapenet_cam_params["azimuth"][param_num])

        cameras = FoVPerspectiveCameras(device=self.device,
                                        R=R,
                                        T=T,
                                        fov=shapenet_cam_params["field_of_view"][param_num]
                                        )

        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )

        verts = []
        faces = []
        verts_rgb = []
        titles = []

        vert, face = load_ply(ply_path)
        verts.append(vert.to(self.device))
        faces.append(face.to(self.device))
        verts_rgb.append(torch.ones_like(vert).to(self.device))

        textures = Textures(verts_rgb=verts_rgb)
        interpol_mesh = Meshes(verts, faces, textures)

        image = renderer(interpol_mesh).cpu().numpy()
        print(image.shape)

        reformatted_image = self.cv2_image_transform(image[0])
        print(reformatted_image.min())

        out = torch.from_numpy(reformatted_image).unsqueeze(0).type(torch.float32).to(self.device)

        # print(out)
        return out

    def annealing_view_pytorch3d(self, ply_paths: List[str]):

        verts = []
        faces = []
        verts_rgb = []
        for ply_path in ply_paths:
            vert, face = load_ply(ply_path)
            verts.append(vert.to(self.device))
            faces.append(face.to(self.device))
            verts_rgb.append(torch.ones_like(vert).to(self.device))
            #verts_rgb.append(torch.rand(size=vert.size()).to(self.device))

        textures = Textures(verts_rgb=verts_rgb)
        interpol_mesh = Meshes(verts, faces, textures)

        # print(interpol_mesh.isempty())
        # print(interpol_mesh.num_verts_per_mesh())

        image = self.shapenet_render.render(model_ids=[0],
                                            meshes=interpol_mesh,
                                            device=self.device
                                            ).cpu().numpy()
        # print(image.shape)

        reformatted_image = self.cv2_image_transform(image[0])

        out = torch.from_numpy(reformatted_image).unsqueeze(0).type(torch.float32).to(self.device)

        return out

    def latent_gradient(self, base_batch_view, target_batch_view, step, config):
        # zero gradients
        self.im_network.zero_grad()

        # re-register forward hook on each forward pass.
        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))

        z_vec_, _ = self.im_network(target_batch_view, None, None, is_training=False)
        style_activation = self.target_activation[0].detach().clone().squeeze()

        # zero gradients
        self.im_network.zero_grad()

        # re-register forward hook on each forward pass.
        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))

        z_vec_, _ = self.im_network(base_batch_view, None, None, is_training=False)
        base_activation = self.target_activation[0]

        # compute best feature maps
        features, width, height = style_activation.shape
        style_activation = style_activation.view(features, -1)

        comp_base_activation = base_activation.squeeze().view(features, -1)

        # Matrix of best matching feature maps.
        A = torch.matmul(torch.transpose(comp_base_activation, 0, 1), style_activation)
        # A = comp_base_activation.T.dot(style_activation)

        loss = comp_base_activation[:, torch.argmax(A, 1)].view(features, width, height).detach()

        # run the graph in reverse
        base_activation.backward(loss.unsqueeze(0))

        return base_batch_view.grad

    def image_deepdream(self, config):

        # TODO: uncomment load data
        self.load_data(config)

        # TODO: uncomment checkpoint load
        # load previous checkpoint
        self.load_checkpoint()

        # get config values
        z_base = int(config.interpol_z1)
        base_im_num = int(config.z1_im_view)
        z_target = int(config.interpol_z2)
        target_im_num = int(config.z1_im_view)

        # instantiate camera rendering class
        self.shapenet_render = ShapeNetRendering([z_base, z_target],
                                                 config.R2N2_dir,
                                                 model_views=[[base_im_num], [target_im_num]],
                                                 )

        # set the dreaming rate and boundary size
        self.dream_rate = config.dream_rate
        annealing_step = config.annealing_rate

        # Set up forward hook to pull values
        self.layer_num = config.layer_num

        # list index includes as zero entry the generator module itself.
        # 2 layers up front should not be used
        num_model_layers = len(list(self.im_network.img_encoder.named_children())) - 2
        if self.layer_num < 2 or self.layer_num >= num_model_layers:
            print('Layer number is too large: select layer numbers from 2 to {}'.format(num_model_layers))
            exit(0)

        # self.target_layer = list(list(self.im_network.img_encoder.children())[self.layer_num].children())[-1]
        self.target_layer = list(self.im_network.img_encoder.children())[self.layer_num]
        self.target_activation = [None]

        interpol_steps = int(config.interpol_steps)
        result_base_directory = config.interpol_directory
        result_dir_name = 'DeepDream_SVR' + str(z_base) + '_' + str(z_target) + '_layer_' + str(self.layer_num)
        self.result_dir = result_base_directory + '/' + result_dir_name

        # Create output directory
        # TODO: re-create directory
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
            print('creating directory ' + self.result_dir)

        # store images
        num_images = interpol_steps // annealing_step
        annealing_images = np.empty(shape=(num_images + 2, 1, 128, 128))
        deepdream_images = np.empty(shape=(num_images + 2, 1, 128, 128))

        # TODO: remove dummy data
        # batch_view = np.random.random(size=(1, 1, 128, 128))
        batch_view = self.data_pixels[0:1, base_im_num, ...].astype(np.float32) / 255.0
        base_batch_view_ = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        base_batch_view = torch.autograd.Variable(base_batch_view_, requires_grad=True)
        deepdream_images[0, ...] = batch_view[0, ...]

        # TODO: uncomment mesh save
        self.create_model_mesh(base_batch_view, 'base', config)

        # TODO: remove dummy data
        # batch_view = np.random.random(size=(1, 1, 128, 128))
        batch_view = self.data_pixels[1:2, target_im_num, ...].astype(np.float32) / 255.0
        target_batch_view = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        deepdream_images[1, ...] = batch_view[0, ...]

        # TODO: uncomment mesh save
        self.create_model_mesh(target_batch_view, 'target', config)

        # get target activation
        # z_vec_, _ = self.im_network(target_batch_view, None, None, is_training=False)
        # self.style_activation = self.target_activation[0].data.clone().detach().squeeze()

        for step in range(interpol_steps):
            start_time = time.perf_counter()

            # mask zero valued areas
            mask = base_batch_view < 1.99e5

            grad = self.latent_gradient(base_batch_view, target_batch_view, step, config)
            grad = grad[mask]

            #print(grad.shape)

            # mask low value fluctuations, one standard deviation below mean
            grad_mean = grad.mean()
            #print(grad_mean)
            grad_var = torch.pow(torch.mean(torch.pow(grad-grad_mean, 2)), .5)
            #print(grad_var)
            #grad[grad < grad_mean - grad_var] = 0

            grad_step = grad * self.dream_rate / torch.abs(grad_mean)

            #grad_step = self.dream_rate * (grad - grad_mean) / grad_var
            #print(grad_step.shape)

            # print(torch.max(grad_step))

            # clamp output to min,max input values.
            # base_batch_view.data = torch.clamp(base_batch_view.data - grad_step, min=0., max=1.)
            with torch.no_grad():
                base_batch_view.data[mask] += grad_step
                base_batch_view.clamp_(min=0, max=1)

                print(base_batch_view.shape)

                # apply a mask to remove border artifacts

                border = 8
                # right border
                base_batch_view.data[..., :, 0:border] = 1
                # left border
                base_batch_view[..., :, -border:] = 1
                # top border
                base_batch_view[..., 0:border, :] = 1
                # bottom border
                base_batch_view[..., -border:, :] = 1

                # print(torch.max(grad))

            # Make sure gradients flow on the update
            # base_batch_view.requires_grad = True

            # create ply models
            if (step) % annealing_step == 0:
                if step != 0:
                    # TODO: uncomment mesh save
                    # save model
                    ply_path = self.create_model_mesh(base_batch_view, step, config)

                    # save image
                    deepdream_images[step // annealing_step + 1, ...] = base_batch_view.clone().detach().cpu().numpy()[
                        0, ...]

                    # get a new annealing model image
                    with torch.no_grad():
                        # base_batch_view.data = self.annealing_view(ply_path=ply_path)
                        base_batch_view.data = self.annealing_view_pytorch3d(ply_paths=[ply_path])

                    # save image
                    annealing_images[step // annealing_step + 1, ...] = base_batch_view.clone().detach().cpu().numpy()[
                        0, ...]

            end_time = time.perf_counter()
            print('Completed dream {} in {} seconds'.format(step, end_time - start_time))

        self.create_model_mesh(base_batch_view, step, config)
        self.create_saved_images(deepdream_images, 'deepdream_images')
        self.create_saved_images(annealing_images, 'annealing_images')

        print('Done Dreaming..')
