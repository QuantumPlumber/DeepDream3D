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

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

import mcubes

from .utils import *
from .modelSVR import IM_SVR


class IM_SVR_DD(IM_SVR):
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
        if z_num < len(self.data_pixels):
            batch_view = self.data_pixels[z_num:z_num + 1, self.test_idx].astype(np.float32) / 255.0
            batch_view = torch.from_numpy(batch_view)
            batch_view = batch_view.to(self.device)
            z_vec_, _ = self.im_network(batch_view, None, None, is_training=False)
            z_vec = z_vec_.detach().cpu().numpy()

            return (z_vec)

        else:
            print("z_num not a valid number")

    def latent_gradient(self, base_batch_view, step, config):

        self.im_network.zero_grad()

        z_vec_, _ = self.im_network(base_batch_view, None, None, is_training=False)
        base_activation = self.target_activation[0]

        # compute best feature maps
        feature, width, height = self.style_activation.shape
        style_activation = self.style_activation.view(feature, -1)

        comp_base_activation = base_activation.view(feature, -1)

        # Matrix of best matching feature maps.
        A = torch.matmul(torch.transpose(comp_base_activation, 0, 1), style_activation)
        # A = comp_base_activation.T.dot(style_activation)

        loss = comp_base_activation[:, torch.argmax(A, 1)].view(feature, width, height)

        # run the graph in reverse
        base_activation.backward(loss.unsqueeze(0))

        return base_batch_view.grad

    def create_saved_images(self, images):
        num_images = int(images.shape[0])
        num_rows = 3
        cols = int(num_images) // num_rows
        rows = -int(-num_images // cols)

        # convert back to grayscale
        rescale_images = images * 255

        fig, axs = plt.subplots(nrows=rows,
                                ncols=cols,
                                sharex='all',
                                sharey='all',
                                figsize=(rows * 2, cols * 2),
                                gridspec_kw={'wspace': 0, 'hspace': 0}
                                )
        for ax, im in zip(axs.flatten('F'), range(num_images)):
            ax.imshow(rescale_images[im, 0, :, :], cmap='gray')
            ax.axis('off')

        plt.savefig(self.result_dir + '/' + 'image_progression.png')

    def image_deepdream(self, config):

        # TODO: uncomment checkpoint load
        # load previous checkpoint
        # self.load_checkpoint()

        # set the dreaming rate and boundary size
        self.dream_rate = config.dream_rate

        # Set up forward hook to pull values
        self.layer_num = config.layer_num
        # list index includes as zero entry the generator module itself.
        # 2 layers up front should not be used
        num_model_layers = len(list(self.im_network.img_encoder.named_children())) - 2
        if self.layer_num < 2 or self.layer_num >= num_model_layers:
            print('Layer number is too large: select layer numbers from 2 to {}'.format(num_model_layers))
            exit(0)

        # this is the way to get the model variable of interest
        # take the last batch normalization layer in each res-net block before running through the relu
        # self.target_layer = list(list(self.im_network.img_encoder.children())[self.layer_num].children())[-1]
        self.target_layer = list(self.im_network.img_encoder.children())[self.layer_num]
        self.target_activation = [None]

        # register the forward hook
        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))

        # get config values
        z_base = int(config.interpol_z1)
        z_target = int(config.interpol_z2)

        interpol_steps = int(config.interpol_steps)
        result_base_directory = config.interpol_directory
        result_dir_name = 'DeepDream_SVR' + str(z_base) + '_' + str(z_target) + '_layer_' + str(self.layer_num)
        self.result_dir = result_base_directory + '/' + result_dir_name

        # Create output directory
        # TODO: re-create directory
        # if not os.path.isdir(self.result_dir):
        #    os.mkdir(self.result_dir)
        #    print('creating directory ' + self.result_dir)

        # store images
        saved_images = np.empty(shape=(interpol_steps + 2, 1, 128, 128))

        # TODO: remove dummy data
        batch_view = np.random.random(size=(1, 1, 128, 128))
        # batch_view = self.data_pixels[z_base:z_base + 1, self.test_idx].astype(np.float32) / 255.0
        base_batch_view_ = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        base_batch_view = torch.autograd.Variable(base_batch_view_, requires_grad=True)
        saved_images[0, ...] = batch_view[0, ...]

        # TODO: remove dummy data
        batch_view = np.random.random(size=(1, 1, 128, 128))
        # batch_view = self.data_pixels[z_target:z_target + 1, self.test_idx].astype(np.float32) / 255.0
        target_batch_view = torch.from_numpy(batch_view).type(torch.float32).to(self.device)
        saved_images[0, ...] = batch_view[0, ...]

        # get target activation
        z_vec_, _ = self.im_network(target_batch_view, None, None, is_training=False)
        self.style_activation = self.target_activation[0].clone().detach().squeeze()



        for step in range(interpol_steps):
            start_time = time.perf_counter()

            grad = self.latent_gradient(base_batch_view, step, config)

            grad_step = grad * self.dream_rate / torch.abs(grad.mean())

            base_batch_view.data += grad_step

            # save image
            saved_images[step, ...] = base_batch_view.clone().detach().cpu().numpy()[0, ...]


            end_time = time.perf_counter()
            print('Completed dream {} in {} seconds'.format(step, end_time - start_time))

        self.create_saved_images(saved_images)

        print('Done Dreaming..')
