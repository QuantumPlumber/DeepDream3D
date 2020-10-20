import os
import sys
import argparse
import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from DeepDream3D.ModelDefinition.modelAE import IM_AE
# from DeepDream3D.ModelDefinition.modelSVR import IM_SVR
from DeepDream3D.ModelDefinition.modelAE_DD import IM_AE_DD
from DeepDream3D.ModelDefinition.modelSVR_DD import IM_SVR_DD
from DeepDream3D.ModelDefinition.utils import get_parser

parser = get_parser()

# --------------- process -------------------------------

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
    im_ae = IM_AE_DD(FLAGS)

    if FLAGS.train:
        im_ae.train(FLAGS)
    elif FLAGS.getz:
        im_ae.get_z(FLAGS)
    elif FLAGS.interpol:
        im_ae.interpolate_z(FLAGS)
    elif FLAGS.deepdream:
        im_ae.deep_dream(FLAGS)
    else:
        # im_ae.test_mesh(FLAGS)
        im_ae.test_mesh_point(FLAGS)

elif FLAGS.svr:
    im_svr = IM_SVR_DD(FLAGS)

    if FLAGS.train:
        im_svr.train(FLAGS)
    elif FLAGS.interpol:
        print('use ae to interpolate')
        exit()
    elif FLAGS.deepdream:
        im_svr.deep_dream(FLAGS)
    else:
        # im_svr.test_mesh(FLAGS)
        im_svr.test_mesh_point(FLAGS)

else:
    print("Please specify a model type?")
