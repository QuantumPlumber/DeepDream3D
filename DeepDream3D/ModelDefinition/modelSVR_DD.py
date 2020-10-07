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
from .modelSVR import IM_SVR


class IM_AE_DD(IM_SVR):
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

    def latent_gradient(self, z_base_num, z_target_num, step, config):

        self.im_network.zero_grad()

        if not self.style_activation_recorded:
            batch_view = self.data_pixels[z_target_num:z_target_num + 1, self.test_idx].astype(np.float32) / 255.0
            batch_view = torch.from_numpy(batch_view)
            batch_view = batch_view.to(self.device)
            z_vec_, _ = self.im_network(batch_view, None, None, is_training=False)
            self.style_activation = self.target_activation[0].cpu().numpy().copy()
            self.style_activation_recorded = True

        batch_view = self.data_pixels[z_base_num:z_base_num + 1, self.test_idx].astype(np.float32) / 255.0
        batch_view = torch.from_numpy(batch_view)
        batch_view = batch_view.to(self.device)
        z_vec_, _ = self.im_network(batch_view, None, None, is_training=False)
        base_activation = self.target_activation[0]

        # compute best feature maps
        feature, width, height = self.style_activation.shape
        style_activation = self.style_activation.reshape(feature, -1)

        comp_base_activation = base_activation.cpu().numpy().copy()
        comp_base_activation = comp_base_activation.reshape(feature, -1)

        # Matrix of best matching feature maps.
        A = comp_base_activation.T.dot(style_activation)

        loss = torch.Tensor(comp_base_activation[:, A.Argmax(1)].reshape(feature, width, height),
                            dtype=torch.float32).to(self.device)

        # run the graph in reverse
        base_activation.backward(loss)

        return batch_view.grad

    def image_deepdream(self, config):

        # TODO: uncomment checkpoint load
        # load previous checkpoint
        self.load_checkpoint()

        # set the dreaming rate and boundary size
        self.dream_rate = config.dream_rate

        # record if style vector has been processed
        self.style_activation_recorded = False

        # get config values
        z1 = int(config.interpol_z1)
        z2 = int(config.interpol_z2)

        # this is the way to get the actual model variable
        self.target_layer = list(self.im_network.generator.named_modules())[self.layer_num][1]
        self.target_activation = [None]

        self.target_layer.register_forward_hook(self.get_activation(self.target_activation))
