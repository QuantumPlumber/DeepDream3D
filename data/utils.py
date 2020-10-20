import os
import h5py as h5
import re
from pathlib import Path
import shutil

METADATA_DIR = Path().resolve().parents[0]


def regex_directory_parser(directory_path, regex):
    files = os.listdir(directory_path)
    files_list = []
    for filename in files:
        if regex.match(filename) is not None:
            files_list.append(filename)

    return files_list


def recursive_hdf5(entity, numtab=0):
    if isinstance(entity, h5.Group):
        for key in entity.keys():
            print('\t' * numtab + key)
            new_tab_level = numtab + 1
            recursive_hdf5(entity[key], new_tab_level)
    elif isinstance(entity, h5.Dataset):
        print('\t shape: ' + str(entity.shape))

    return


def copy_hdf5(src_file: h5.File, dest_file: h5.File, indices: list):
    first_dim = len(indices)
    for key in src_file.keys():
        src_data = src_file[key]
        shape = list(src_data.shape)
        shape[0] = first_dim
        dest_data = dest_file.create_dataset(name=key,
                                             shape=shape,
                                             dtype=src_data.dtype)

        for dest_i, src_i in enumerate(indices):
            dest_data[dest_i] = src_data[src_i]
            print('copied {}'.format(dest_i))
    return


def copy_shapenet_model(model_nums=[300],
                        dest_folder='data/raw',
                        R2N2_dir='/data/shapenet',
                        splitfile='data/metadata/all_vox256_img_test.txt',
                        views_rel_path="ShapeNetRendering"):
    print(model_nums)
    for model_num in model_nums:
        print('copying model number {}'.format(model_num))

        model = {}

        # get model based on id number in splitfile
        with open(os.path.join(METADATA_DIR, splitfile), "r") as f:
            synset_lines = f.readlines()
            synset_id, model_id = synset_lines[model_num].split('/')
            model["synset_id"] = synset_id
            model["model_id"] = model_id.rstrip()

        with open(os.path.join(METADATA_DIR, 'data/metadata/test_all_vox256_img_test.txt'), "a") as local:
            local.write(model["synset_id"] + '/' + model["model_id"] + '\n')

        '''
        source_dir = os.path.join(
            R2N2_dir,
            views_rel_path,
            model["synset_id"],
            model["model_id"]
        )

        destination_dir = os.path.join(METADATA_DIR, dest_folder)

        # Copy ShapeNetData

        test_path = os.path.join(destination_dir, views_rel_path)
        if not os.path.isdir(test_path):
            os.mkdir(test_path)
            print('created directory {}'.format(test_path))

        test_path = os.path.join(test_path, model["synset_id"])
        if not os.path.isdir(test_path):
            os.mkdir(test_path)
            print('created directory {}'.format(test_path))

        test_path = os.path.join(test_path, model["model_id"])
        if not os.path.isdir(test_path):
            #os.mkdir(test_path)
            print('created directory {}'.format(test_path))

        dest = shutil.copytree(source_dir, test_path)
        print('copied')
        print('copied {} to {}'.format(source_dir, dest))
        '''

if __name__ == "__main__":

    data_path = '/data/IM-NET-pytorch/data/all_vox256_img/'

    reg_data = re.compile('\w*.hdf5')
    datafiles_list = regex_directory_parser(data_path, reg_data)
    print(datafiles_list)

    src_file = h5.File(data_path + datafiles_list[0], 'r')
    recursive_hdf5(src_file, 0)
    for key in src_file.keys():
        num_models = src_file[key].shape[0]
    print(num_models)

    dest_folder = 'data/processed'
    hdf5_dest_path = os.path.join(METADATA_DIR, dest_folder, datafiles_list[0])
    dest_file = h5.File(name=hdf5_dest_path, mode='w')

    indices = list(range(0, num_models, 100))
    print(indices)
    copy_hdf5(src_file=src_file, dest_file=dest_file, indices=indices)

    indices = list(range(0, num_models, 100))
    print(indices)
    print('calling copy shapenet')

    '''
    copy_shapenet_model(model_nums=indices,
                        dest_folder='data/raw',
                        R2N2_dir='/data/shapenet',
                        splitfile='data/metadata/all_vox256_img_test.txt',
                        views_rel_path="ShapeNetRendering")
    '''