import streamlit as st
import os
import sys
import argparse
import yaml
import json
import re
import h5py as h5
import numpy as np
import copy
import matplotlib.pyplot as plt

from skimage.io import imread

from pytorch3d.ops import cubify

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
    Textures
)

import torch

# Setup Torch Device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# insert project directory, useful if DeepDream3D module is not installed.
sys.path.insert(0, os.path.dirname(os.getcwd()))

from DeepDream3D.ModelDefinition.modelAE import IM_AE
from DeepDream3D.ModelDefinition.modelSVR import IM_SVR
from DeepDream3D.ModelDefinition.modelAE_DD import IM_AE_DD

# ---------------------------------------------------------------------------------------------------------------------#
parser_actions = []

parser = argparse.ArgumentParser(conflict_handler='resolve')

# --------------- specify yml config -------------------------------

# parser.add_argument("--yaml_config", action="store", dest="yaml_config", default=None, type=str,
#                    help="Optionally specify parameters with a yaml file. YAML file overrides command line args")

# --------------- specify model architecture -------------------------------

parser_actions.append(
    parser.add_argument("--ae", action='store_true', dest="ae", default=False, help="Check for 3D auto-encoder"))

parser_actions.append(parser.add_argument("--svr", action='store_true', dest="svr", default=False,
                                          help="Check for single view reconstruction"))

parser_actions.append(parser.add_argument("--deepdream", action='store_true', dest="deepdream", default=False,
                                          help="Check for DeepDream "))

# --------------- specify model mode -------------------------------

parser_actions.append(parser.add_argument("--train", action='store_true', dest="train", default=False,
                                          help="Check for training data, otherwise use testing data"))

parser_actions.append(parser.add_argument("--getz", action='store_true', dest="getz", default=False,
                                          help="True for getting latent codes "))

parser_actions.append(parser.add_argument("--interpol", action='store_true', dest="interpol", default=False,
                                          help="Check to linearly interpolate."))

# --------------- training -------------------------------

parser_actions.append(
    parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
                        help="Voxel resolution for coarse-to-fine training [64]"))

parser_actions.append(
    parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]"))

parser_actions.append(parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int,
                                          help="Iteration to train. Either epoch or iteration need to be zero [0]"))

parser_actions.append(
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float,
                        help="Learning rate for adam [0.00005]"))

parser_actions.append(parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float,
                                          help="Momentum term of adam [0.5]"))

# --------------- testing -------------------------------

parser_actions.append(parser.add_argument("--start", action="store", dest="start", default=0, type=int,
                                          help="In testing, output shapes [start:end]"))

parser_actions.append(parser.add_argument("--end", action="store", dest="end", default=16, type=int,
                                          help="In testing, output shapes [start:end]"))

# --------------- Data and Directories -------------------------------

parser_actions.append(parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img",
                                          help="The name of dataset"))

parser_actions.append(
    parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint",
                        help="Directory name to save the checkpoints"))

parser_actions.append(
    parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/",
                        help="Root directory of dataset [data]"))

parser_actions.append(parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/",
                                          help="Directory name to save the image samples"))

parser_actions.append(
    parser.add_argument("--interpol_directory", action="store", dest="interpol_directory", default=None,
                        help="First Interpolation latent vector"))

# --------------- Interpolation -------------------------------

parser_actions.append(parser.add_argument("--interpol_z1", action="store", dest="interpol_z1", type=int, default=0,
                                          help="First Interpolation latent vector"))

parser_actions.append(parser.add_argument("--interpol_z2", action="store", dest="interpol_z2", type=int, default=1,
                                          help="Second Interpolation latent vector"))

parser_actions.append(
    parser.add_argument("--interpol_steps", action="store", dest="interpol_steps", type=int, default=5,
                        help="number of steps to take between values"))

# --------------- deepdream -------------------------------

# dreaming uses the interpolation targets from interpolation as well as the number of steps.

parser_actions.append(parser.add_argument("--layer_num", action="store", dest="layer_num", default=3, type=int,
                                          help="activation layer to maximize"))

parser_actions.append(parser.add_argument("--dream_rate", action="store", dest="dream_rate", default=.01, type=float,
                                          help="dream update rate"))


# ---------------------------------------------------------------------------------------------------------------------#

# parameter reading function for parsing YAML file.
def read_config(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            yaml_config = yaml.safe_load(stream)
            print(yaml_config)
        except yaml.YAMLError as exc:
            print(exc)

    # overwrite command line FLAGS
    command_string = []
    for key, value in yaml_config.items():
        command_string.append('--' + str(key))
        if value is not None:
            command_string.append(str(value))
    FLAGS = parser.parse_args(args=command_string)

    return FLAGS


# ---------------------------------------------------------------------------------------------------------------------#
@st.cache(allow_output_mutation=True)
def create_model_instance(FLAGS):
    im_ae_dd = IM_AE_DD(FLAGS)
    # im_svr = IM_SVR(FLAGS)

    return None, im_ae_dd


# ---------------------------------------------------------------------------------------------------------------------#
def data_image(images):
    fig, axs = plt.subplots(nrows=6,
                            ncols=4,
                            sharex='all',
                            sharey='all',
                            figsize=(10, 10),
                            gridspec_kw={'wspace': 0, 'hspace': 0}
                            )

    for ax, im in zip(axs.flatten(), range(24)):
        ax.imshow(images[im, :, :])
        ax.axis('off')

    return fig


# ---------------------------------------------------------------------------------------------------------------------#
@st.cache
def define_render(num):
    shapenet_cam_params_file = '../data/metadata/rendering_metadata.json'
    with open(shapenet_cam_params_file) as f:
        shapenet_cam_params = json.load(f)

    param_num = num
    R, T = look_at_view_transform(
        dist=shapenet_cam_params["distance"][param_num] * 5,
        elev=shapenet_cam_params["elevation"][param_num],
        azim=shapenet_cam_params["azimuth"][param_num])
    cameras = FoVPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    fov=shapenet_cam_params["field_of_view"][param_num]
                                    )

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


# ---------------------------------------------------------------------------------------------------------------------#
def interpolate(FLAGS):
    '''
    General interpolating wrapped in a cache to prevent recomputation

    :param FLAGS:
    :return:
    '''

    global im_ae_dd, renderer_instance

    im_ae_dd.interpolate_z(FLAGS)

    interpolation_dir = im_ae_dd.result_dir

    files = os.listdir(interpolation_dir)
    verts = []
    faces = []
    verts_rgb = []
    for file in files:
        vert, face = load_ply(interpolation_dir + '/' + file)
        verts.append(vert.to(device))
        faces.append(face.to(device))
        verts_rgb.append(torch.ones_like(vert).to(device))

    textures = Textures(verts_rgb=verts_rgb)
    interpol_mesh = Meshes(verts, faces, textures)

    print('rendering images')
    images = renderer_instance(interpol_mesh).cpu().numpy()

    print('processing images')
    num_images = int(images.shape[0])
    rows = int(num_images ** .5)
    print(rows)
    cols = num_images // rows
    print(cols)

    fig, axs = plt.subplots(nrows=rows,
                            ncols=cols,
                            sharex='all',
                            sharey='all',
                            figsize=(20, 20),
                            gridspec_kw={'wspace': 0, 'hspace': 0}
                            )

    for ax, im in zip(axs.flatten(), range(num_images)):
        ax.imshow(images[im, :, :, :3])
        ax.axis('off')

    return fig


# ---------------------------------------------------------------------------------------------------------------------#
def deepdream(FLAGS):
    '''
    General deepdreaming wrapped in a cache to prevent recomputation

    :param FLAGS:
    :return:
    '''

    global im_ae_dd, renderer_instance

    im_ae_dd.deep_dream(FLAGS)

    deep_dream_dir = im_ae_dd.result_dir

    files = os.listdir(deep_dream_dir)
    verts = []
    faces = []
    verts_rgb = []
    for file in files:
        vert, face = load_ply(deep_dream_dir + '/' + file)
        verts.append(vert.to(device))
        faces.append(face.to(device))
        verts_rgb.append(torch.ones_like(vert).to(device))

    textures = Textures(verts_rgb=verts_rgb)
    interpol_mesh = Meshes(verts, faces, textures)

    print('rendering images')
    images = renderer_instance(interpol_mesh).cpu().numpy()

    print('processing images')
    num_images = int(images.shape[0])
    cols = 2
    rows = -int(-num_images // cols)

    fig, axs = plt.subplots(nrows=rows,
                            ncols=cols,
                            sharex='all',
                            sharey='all',
                            figsize=(20, 20),
                            gridspec_kw={'wspace': 0, 'hspace': 0}
                            )

    for ax, im in zip(axs.flatten(), range(num_images)):
        ax.imshow(images[im, :, :, :3])
        ax.axis('off')

    return fig


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    process_flag = False

    st.title('Deep Dreamin\' in 3D')

    # ----------------------- read in yaml config file ----------------------------------

    nominal_YAML = '../configs/default_config.yml'
    nominal_FLAGS = read_config(nominal_YAML)

    user_FLAGS = copy.deepcopy(nominal_FLAGS)
    user_flags_dict = vars(user_FLAGS)

    st.sidebar.text('Click the below button to process \n nominal settings.')
    process_button_flag = st.sidebar.button(
        label='Process',
        key='reset'
    )
    if process_button_flag:
        process_flag = True

    st.sidebar.header('Configuration Options')

    st.sidebar.text('Click the below button to revert to \n nominal settings.')
    reset_model_flag = st.sidebar.button(
        label='Reset',
        key='reset'
    )
    if reset_model_flag:
        nominal_YAML = '../configs/default_config.yml'
        nominal_FLAGS = read_config(nominal_YAML)

    st.sidebar.text(
        'Changing model type requires a model \n reload after selection. Click here \n to reload the model.')
    reload_model_flag = st.sidebar.button(
        label='Change model type',
        key='model re-load'
    )

    if reload_model_flag:
        nominal_FLAGS = user_FLAGS.deepcopy()

    # ----------------------- set up buttons for config file ----------------------------------

    toggle_models = ['ae', 'svr']
    toggle_modes = ['deepdream', 'interpol']
    param_to_expose = ['ae', 'svr',
                       'deepdream', 'interpol',
                       'train',
                       'interpol_z1', 'interpol_z2', 'interpol_steps',
                       'layer_num', 'dream_rate']

    st.sidebar.text('Model Type:')
    for action, [flag, param] in enumerate(user_flags_dict.items()):
        if flag in param_to_expose:
            if type(param) is bool:
                user_flags_dict[flag] = st.sidebar.checkbox(
                    label=parser._actions[action + 1].help,
                    value=param,
                    key=flag
                )

            if type(param) is int:

                if flag is 'layer_num':
                    user_flags_dict[flag] = st.sidebar.number_input(
                        label=parser._actions[action + 1].help,
                        value=param,
                        max_value=6,
                        min_value=1,
                        key=flag
                    )
                else:
                    user_flags_dict[flag] = st.sidebar.number_input(
                        label=parser._actions[action + 1].help,
                        value=param,
                        key=flag
                    )
            if type(param) is float:
                user_flags_dict[flag] = st.sidebar.number_input(
                    label=parser._actions[action + 1].help,
                    value=param,
                    step=.001,
                    key=flag
                )

            if flag == 'svr':
                st.sidebar.text('Model operation:')

    camera_num = st.sidebar.number_input(
        label='ShapeNet rendering camera view number',
        value=8,
        key='camera_view_num'
    )

    # ----------------------- Check parameters for consistency ----------------------------------

    data_path = user_FLAGS.data_dir
    if not os.path.isdir(data_path):
        st.error('Data directory does not exist.')
        process_flag = False

    if not os.path.isdir(user_FLAGS.checkpoint_dir):
        st.error('Checkpoint directory does not exist.')
        process_flag = False

    if not os.path.isdir(user_FLAGS.sample_dir):
        st.error('Sample destination directory does not exist.')
        process_flag = False

    if not os.path.isdir(user_FLAGS.sample_dir):
        st.error('Interpol destination directory does not exist.')
        process_flag = False

    if user_flags_dict['ae'] and user_flags_dict['svr']:
        st.error('Select only one model architecture: Auto-Encoder (AE) or Single View Reconstruction (SVR)')
        process_flag = False

    if user_flags_dict['deepdream'] and user_flags_dict['interpol']:
        st.error('Select only one model mode: interpolation or deep dreaming')
        process_flag = False

    # st.subheader('Nominal Configuration:')
    # st.write(user_flags_dict)

    # ----------------------- begin heavy lifting  ----------------------------------

    # only continue if the settings are correct
    if process_flag == True:

        # instantiate models
        with st.spinner('Loading models, this may take a few moments..'):
            m_svr, im_ae_dd = create_model_instance(nominal_FLAGS)

        # load data
        dataset_name = user_FLAGS.dataset
        if user_FLAGS.train:
            dataset_load = dataset_name + '_train.hdf5'
        else:
            dataset_load = dataset_name + '_test.hdf5'

        with st.spinner('Loading datafile, this may take a few moments..'):
            data_file = h5.File(data_path + '/' + dataset_load, 'r')

        z1 = user_FLAGS.interpol_z1
        z2 = user_FLAGS.interpol_z2

        # display images for each object:
        st.text('The 24 rendered images of shapenet training object {} are displayed below'.format(z1))
        st.pyplot(data_image(data_file['pixels'][z1][...]))

        st.text('The 24 rendered images of shapenet training object {} are displayed below'.format(z2))
        st.pyplot(data_image(data_file['pixels'][z2][...]))

        renderer_instance = define_render(camera_num)

        # Create first and last starting shapes
        st.text("Below are renderings of the shapes after being run through the encoder and IMNET decoder..")
        diagnostic_flags = copy.deepcopy(user_FLAGS)
        diagnostic_flags.interpol_steps = 2
        st.pyplot(interpolate(FLAGS=diagnostic_flags))

        if user_FLAGS.interpol:
            st.text('Interpolation results:')
            with st.spinner('Interpolating: expected wait time is {} seconds..'.format(user_FLAGS.interpol_steps * 6)):
                st.pyplot(interpolate(FLAGS=user_FLAGS))
        if user_FLAGS.deepdream:
            st.text('Deepdream results:')
            with st.spinner('Interpolating: expected wait time is {} seconds..'.format(user_FLAGS.interpol_steps * 10)):
                st.pyplot(deepdream(user_FLAGS))
