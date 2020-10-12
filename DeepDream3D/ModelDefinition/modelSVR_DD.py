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

    def create_saved_images(self, images):
        num_images = int(images.shape[0])
        cols = 3
        rows = -int(-num_images // cols)

        # convert back to grayscale
        rescale_images = images * 255

        fig, axs = plt.subplots(nrows=rows,
                                ncols=cols,
                                sharex='all',
                                sharey='all',
                                figsize=(cols * 2, rows * 2),
                                gridspec_kw={'wspace': 0, 'hspace': 0}
                                )
        for ax, im in zip(axs.flatten(), range(num_images)):
            ax.imshow(rescale_images[im, 0, :, :], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

        plt.savefig(self.result_dir + '/' + 'image_progression.png')

    # output shape as ply
    def create_model_mesh(self, batch_view, num, config):
        # TODO: uncomment load checkpoint
        # load previous checkpoint
        self.load_checkpoint()

        self.im_network.eval()
        model_z, _ = self.im_network(batch_view, None, None, is_training=False)
        model_float = self.z2voxel(model_z)

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
        imgo = img[:, :, :3]
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY) * 255.
        # imga = (img[:, :, 3]) / 255.0
        # img_out = imgo * imga + 255.0 * (1 - imga)
        # img_out = np.round(img_out).astype(np.uint8)

        img_out = imgo[np.newaxis, :, :].astype(np.float32) / 255.

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

        textures = TexturesVertex(verts_features=verts_rgb)
        interpol_mesh = Meshes(verts, faces, textures)

        image = self.shapenet_render.render(model_ids=[0]).cpu().numpy()
        print(image.shape)

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

        # TODO: uncomment checkpoint load
        # load previous checkpoint
        self.load_checkpoint()

        # get config values
        z_base = int(config.interpol_z1)
        z_target = int(config.interpol_z2)

        # instantiate camera rendering class
        self.shapenet_render = ShapeNetRendering([z_base, z_target],
                                            config.R2N2_dir,
                                            model_views=[[self.test_idx], [self.test_idx]])

        # set the dreaming rate and boundary size
        self.dream_rate = config.dream_rate
        annealing_step = 100

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
        saved_images = np.empty(shape=(num_images + 2, 1, 128, 128))

        # TODO: remove dummy data
        # batch_view = np.random.random(size=(1, 1, 128, 128))
        batch_view = self.data_pixels[z_base:z_base + 1, self.test_idx, ...].astype(np.float32) / 255.0
        base_batch_view_ = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        base_batch_view = torch.autograd.Variable(base_batch_view_, requires_grad=True)
        saved_images[0, ...] = batch_view[0, ...]

        # TODO: uncomment mesh save
        self.create_model_mesh(base_batch_view, 'base', config)

        # TODO: remove dummy data
        # batch_view = np.random.random(size=(1, 1, 128, 128))
        batch_view = self.data_pixels[z_target:z_target + 1, self.test_idx, ...].astype(np.float32) / 255.0
        target_batch_view = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        saved_images[1, ...] = batch_view[0, ...]

        # TODO: uncomment mesh save
        self.create_model_mesh(target_batch_view, 'target', config)

        # get target activation
        # z_vec_, _ = self.im_network(target_batch_view, None, None, is_training=False)
        # self.style_activation = self.target_activation[0].data.clone().detach().squeeze()

        for step in range(interpol_steps):
            start_time = time.perf_counter()

            # mask zero valued areas
            mask = base_batch_view < .99

            grad = self.latent_gradient(base_batch_view, target_batch_view, step, config)
            grad = grad[mask]

            grad_step = grad * self.dream_rate / torch.abs(grad.mean())

            # print(torch.max(grad_step))

            # clamp output to min,max input values.
            # base_batch_view.data = torch.clamp(base_batch_view.data - grad_step, min=0., max=1.)
            with torch.no_grad():
                base_batch_view.data[mask] += grad_step
                base_batch_view.clamp_(min=0, max=1)
                # print(torch.max(grad))

            # Make sure gradients flow on the update
            # base_batch_view.requires_grad = True

            # create ply models
            if (step + 1) % annealing_step == 0:
                if step != 0:
                    # save image
                    saved_images[step // annealing_step + 2, ...] = base_batch_view.clone().detach().cpu().numpy()[
                        0, ...]

                    # TODO: uncomment mesh save
                    # save model
                    ply_path = self.create_model_mesh(base_batch_view, step, config)

                    # get a new annealing model image
                    with torch.no_grad():
                        #base_batch_view.data = self.annealing_view(ply_path=ply_path)
                        base_batch_view.data = self.annealing_view_pytorch3d(ply_paths=[ply_path])

            end_time = time.perf_counter()
            print('Completed dream {} in {} seconds'.format(step, end_time - start_time))

        self.create_model_mesh(base_batch_view, step, config)
        self.create_saved_images(saved_images)

        print('Done Dreaming..')
