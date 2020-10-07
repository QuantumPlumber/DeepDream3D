import argparse
import yaml
import sys

default_yml = '../configs/default_config.yml'
args = [None, default_yml]


def test_main_parser(args):
    parser = argparse.ArgumentParser( conflict_handler='resolve')
    # --------------- specify yml config -------------------------------

    #parser.add_argument("--yaml_config", action="store", dest="yaml_config", default=None, type=str,
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

    parser.add_argument("--interpol_z1", action="store", dest="interpol_z1", default=0,
                        help="First Interpolation latent vector")
    parser.add_argument("--interpol_z2", action="store", dest="interpol_z2", default=1,
                        help="Second Interpolation latent vector")
    parser.add_argument("--interpol_steps", action="store", dest="interpol_steps", default=5,
                        help="number of steps to take between values")

    if args[1] is not None:
        print(args[1])
        with open(args[1], 'r') as stream:
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

        print(command_string)

        FLAGS = parser.parse_args(args=command_string)

    for key in vars(FLAGS):
        print([key, vars(FLAGS)[key]])
        #assert yaml_config[key] == vars(FLAGS)[key]


test_main_parser(args=args)
