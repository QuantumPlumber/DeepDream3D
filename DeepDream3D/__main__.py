import os
import sys
import argparse
import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from DeepDream3D.ModelDefinition.modelAE import IM_AE
from DeepDream3D.ModelDefinition.modelSVR import IM_SVR
from DeepDream3D.ModelDefinition.modelAE_DD import IM_AE_DD

parser = argparse.ArgumentParser(conflict_handler='resolve')

# --------------- specify yml config -------------------------------

# parser.add_argument("--yaml_config", action="store", dest="yaml_config", default=None, type=str,
#                    help="Optionally specify parameters with a yaml file. YAML file overrides command line args")

# --------------- specify model architecture -------------------------------

parser.add_argument("--ae", action='store_true', dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--svr", action='store_true', dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--deepdream", action='store_true', dest="deepdream", default=False,
                    help="True for deepdream [False]")

# --------------- specify model mode -------------------------------

parser.add_argument("--train", action='store_true', dest="train", default=False,
                    help="True for training, False for testing [False]")
parser.add_argument("--getz", action='store_true', dest="getz", default=False,
                    help="True for getting latent codes [False]")
parser.add_argument("--interpol", action='store_true', dest="interpol", default=False,
                    help="True for getting latent codes [False]")

# --------------- training -------------------------------

parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
                    help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int,
                    help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float,
                    help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float,
                    help="Momentum term of adam [0.5]")

# --------------- testing -------------------------------

parser.add_argument("--start", action="store", dest="start", default=0, type=int,
                    help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int,
                    help="In testing, output shapes [start:end]")

# --------------- Data and Directories -------------------------------

parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img",
                    help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint",
                    help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/",
                    help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/",
                    help="Directory name to save the image samples [samples]")
parser.add_argument("--interpol_directory", action="store", dest="interpol_directory", default=None,
                    help="First Interpolation latent vector")

# --------------- Interpolation -------------------------------

parser.add_argument("--interpol_z1", action="store", dest="interpol_z1", type=int, default=0,
                    help="First Interpolation latent vector")
parser.add_argument("--interpol_z2", action="store", dest="interpol_z2", type=int, default=1,
                    help="Second Interpolation latent vector")
parser.add_argument("--interpol_steps", action="store", dest="interpol_steps", type=int, default=5,
                    help="number of steps to take between values")

# --------------- deepdream -------------------------------

# dreaming uses the interpolation targets from interpolation as well as the number of steps.

parser.add_argument("--layer_num", action="store", dest="layer_num", default=3, type=int,
                    help="activation layer to maximize")

parser.add_argument("--dream_rate", action="store", dest="dream_rate", default=.01, type=float,
                    help="dream update rate")

if sys.argv[1] is not None:
    with open(sys.argv[1], 'r') as stream:
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

# TODO: uncomment directory creation
# if not os.path.exists(FLAGS.sample_dir):
#    os.makedirs(FLAGS.sample_dir)

if FLAGS.ae:
    im_ae = IM_AE(FLAGS)

    if FLAGS.train:
        im_ae.train(FLAGS)
    elif FLAGS.getz:
        im_ae.get_z(FLAGS)
    else:
        # im_ae.test_mesh(FLAGS)
        im_ae.test_mesh_point(FLAGS)

elif FLAGS.svr:
    im_svr = IM_SVR(FLAGS)

    if FLAGS.train:
        im_svr.train(FLAGS)
    else:
        # im_svr.test_mesh(FLAGS)
        im_svr.test_mesh_point(FLAGS)

elif FLAGS.deepdream:
    im_ae_dd = IM_AE_DD(FLAGS)

    if FLAGS.interpol:
        im_ae_dd.interpolate_z(FLAGS)
    else:
        im_ae_dd.deep_dream(FLAGS)

else:
    print("Please specify a model type?")
