import os
import time
import math
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch

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
    HardPhongShader,
    TexturesUV,
    Textures,
    TexturesVertex
)

from pytorch3d.datasets.utils import collate_batched_meshes

from pytorch3d.datasets.r2n2.utils import (
    BlenderCamera,
    align_bbox,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)

import cv2

import mcubes
from typing import List

from ..preprocessing.utils import shapenet_cam_params

import path
from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parents[2]
MAX_CAMERA_DISTANCE = 1.75  # Constant from R2N2.
# Intrinsic matrix extracted from Blender. Taken from meshrcnn codebase:
# https://github.com/facebookresearch/meshrcnn/blob/master/shapenet/utils/coords.py
BLENDER_INTRINSIC = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
)


class ShapeNetRendering:
    '''
    This is a class finds the rendering parameters for the shapenet model in question and reconstructs the rendering
    pipeline as done in PyTorch3D from facebook research.

    Most of this code has been adapted from PyTorch3D R2N2 dataset architecture:
    https://pytorch3d.readthedocs.io/en/latest/modules/datasets.html

    Subclassing the architecture was deemed more difficult than re-factoring only what functions were necessary.

    '''

    def __init__(self,
                 model_nums: List[int],
                 R2N2_dir: str,
                 splitfile: str = 'data/metadata/all_vox256_img_test.txt',
                 model_views: List[list] = None,
                 views_rel_path: str = "ShapeNetRendering"
                 ):
        '''
        :param model_num: model number to find.
        :param splitfile: File containing the sysnet ids for each entry in the hdf5 datafile
        :param R2N2_dir: directory where shapenet v2 renderings are stored
        :param model_view: view number to render
        :param views_rel_path: the relative path from the R2N2 directory for views
        '''

        # self.model_num = model_nums
        self.r2n2_dir = R2N2_dir
        self.splitfile = splitfile
        # self.model_views = model_views
        self.views_rel_path = views_rel_path

        self.models = []
        for model_num, model_view in zip(model_nums, model_views):
            self.models.append(self.get_model(model_num=model_num, model_views=model_view))

    def get_model(self, model_num, model_views):
        model = {}

        # get model based on id number in splitfile
        with open(path.join(METADATA_DIR, self.splitfile), "r") as f:
            synset_lines = f.readlines()
            synset_id, model_id = synset_lines[model_num].split('/')
            model["synset_id"] = synset_id
            model["model_id"] = model_id

        model["images"] = None
        images, Rs, Ts, voxel_RTs = [], [], [], []
        # Retrieve R2N2's renderings if required.

        rendering_path = path.join(
            self.r2n2_dir,
            self.views_rel_path,
            model["synset_id"],
            model["model_id"],
            "rendering",
        )

        # Read metadata file to obtain params for calibration matrices.
        with open(path.join(rendering_path, "rendering_metadata.txt"), "r") as f:
            metadata_lines = f.readlines()
        for i in model_views:
            # Get camera calibration.
            azim, elev, yaw, dist_ratio, fov = [
                float(v) for v in metadata_lines[i].strip().split(" ")
            ]
            dist = dist_ratio * MAX_CAMERA_DISTANCE
            # Extrinsic matrix before transformation to PyTorch3D world space.
            RT = compute_extrinsic_matrix(azim, elev, dist)
            R, T = self._compute_camera_calibration(RT)
            Rs.append(R)
            Ts.append(T)
            voxel_RTs.append(RT)

        # Intrinsic matrix extracted from the Blender with slight modification to work with
        # PyTorch3D world space. Taken from meshrcnn codebase:
        # https://github.com/facebookresearch/meshrcnn/blob/master/shapenet/utils/coords.py
        K = torch.tensor(
            [
                [2.1875, 0.0, 0.0, 0.0],
                [0.0, 2.1875, 0.0, 0.0],
                [0.0, 0.0, -1.002002, -0.2002002],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        model["images"] = torch.stack(images)
        model["R"] = torch.stack(Rs)
        model["T"] = torch.stack(Ts)
        model["K"] = K.expand(len(model_views), 4, 4)

        return model

    def _compute_camera_calibration(self, RT):
        """
        Helper function for calculating rotation and translation matrices from ShapeNet
        to camera transformation and ShapeNet to PyTorch3D transformation.

        Args:
            RT: Extrinsic matrix that performs ShapeNet world view to camera view
                transformation.

        Returns:
            R: Rotation matrix of shape (3, 3).
            T: Translation matrix of shape (3).
        """
        # Transform the mesh vertices from shapenet world to pytorch3d world.
        shapenet_to_pytorch3d = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
        # Extract rotation and translation matrices from RT.
        R = RT[:3, :3]
        T = RT[3, :3]
        return R, T

    def render(
            self,
            model_ids: List[int] = None,
            meshes: Meshes = None,
            shader_type=HardPhongShader,
            device="cpu"
    ) -> torch.Tensor:
        """
        Render models with BlenderCamera by default to achieve the same orientations as the
        R2N2 renderings. Also accepts other types of cameras and any of the args that the
        render function in the ShapeNetBase class accepts.

        Args:
            model_ids: List[str] of model_ids of models intended to be rendered.
            meshes: List[Meshes] mesh with textures corresponding to model ids
            shader_type: Shader to use for rendering. Examples include HardPhongShader
            (default), SoftPhongShader etc or any other type of valid Shader class.
            device: torch.device on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports and any of the
                args that BlenderCamera supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """

        # unpack values for models
        r_, t_, k_, mesh_ = [], [], [], []
        for id in model_ids:
            r_.append(self.models[id]["R"])
            t_.append(self.models[id]["T"])
            k_.append(self.models[id]["K"])

        r = torch.cat(r_)
        t = torch.cat(t_)
        k = torch.cat(k_)
        # Initialize default camera using R, T, K from kwargs or R, T, K of the specified views.
        blend_cameras = BlenderCamera(
            R=r,
            T=t,
            K=k,
            device=device,
        )
        cameras = blend_cameras

        if len(cameras) != 1 and len(cameras) % len(meshes) != 0:
            raise ValueError("Mismatch between batch dims of cameras and meshes.")
        if len(cameras) > 1:
            # When rendering R2N2 models, if more than one views are provided, broadcast
            # the meshes so that each mesh can be rendered for each of the views.
            meshes = meshes.extend(len(cameras) // len(meshes))

        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=shader_type(
                device=device,
                cameras=cameras,
                lights=PointLights().to(device),
            ),
        )
        return renderer(meshes)
