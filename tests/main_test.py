import argparse
import yaml
import h5py
import copy

default_yml = "../configs/default_config.yml"
args = ['--yaml_config', default_yml]


def test_main_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", action="store", dest="yaml_config", default=None, type=str,
                        help="Optionally specify parameters with a yaml file. YAML file overrides command line args")
    parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
    parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int,
                        help="Iteration to train. Either epoch or iteration need to be zero [0]")
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float,
                        help="Learning rate for adam [0.00005]")
    parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float,
                        help="Momentum term of adam [0.5]")
    parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img",
                        help="The name of dataset")
    parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint",
                        help="Directory name to save the checkpoints [checkpoint]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/",
                        help="Root directory of dataset [data]")
    parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/",
                        help="Directory name to save the image samples [samples]")
    parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
                        help="Voxel resolution for coarse-to-fine training [64]")
    parser.add_argument("--train", action="store", dest="train", default=False, type=bool,
                        help="True for training, False for testing [False]")
    parser.add_argument("--start", action="store", dest="start", default=0, type=int,
                        help="In testing, output shapes [start:end]")
    parser.add_argument("--end", action="store", dest="end", default=16, type=int,
                        help="In testing, output shapes [start:end]")
    parser.add_argument("--ae", action="store", dest="ae", default=False, type=bool, help="True for ae [False]")
    parser.add_argument("--svr", action="store", dest="svr", default=False, type=bool, help="True for svr [False]")
    parser.add_argument("--getz", action="store", dest="getz", default=False, type=bool,
                        help="True for getting latent codes [False]")
    parser.add_argument("--interpol", action="store", dest="interpol", default=False, type=bool,
                        help="True for getting latent codes [False]")
    parser.add_argument("--interpol_directory", action="store", dest="interpol_directory", default=None,
                        help="First Interpolation latent vector")
    parser.add_argument("--interpol_z1", action="store", dest="interpol_z1", default=0,
                        help="First Interpolation latent vector")
    parser.add_argument("--interpol_z2", action="store", dest="interpol_z2", default=1,
                        help="Second Interpolation latent vector")
    parser.add_argument("--interpol_steps", action="store", dest="interpol_steps", default=5,
                        help="number of steps to take between values")

    FLAGS = copy.deepcopy(parser.parse_args(args=args))

    # process yaml_file
    if FLAGS.yaml_config is not None:
        with open(FLAGS.yaml_config, 'r') as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # overwrite command line FLAGS
        command_string = args
        for key, value in yaml_config.items():
            command_string.append('--' + str(key))
            command_string.append(str(value))

        print(command_string)

        FLAGS_test = parser.parse_args(args=command_string)

    assert FLAGS != FLAGS_test

    for key in vars(FLAGS):
        print(key)
        print(type(vars(FLAGS)[key]))
        print(type(vars(FLAGS_test)[key]))
        print(vars(FLAGS)[key])
        print(vars(FLAGS_test)[key])
        assert vars(FLAGS)[key] == vars(FLAGS_test)[key]


test_main_parser(args=args)
