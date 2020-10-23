# DeepDream3D
This is a project exploring the concept of [DeepDream](https://github.com/google/deepdream.git) applied to 
3D deep learning generative modeling. Specifically, the [IM-NET](https://github.com/czq142857/implicit-decoder.git) model.
The DeepDream3D module is an encapsulation of the code provided 
by the IM-NET team, with extended functionality to enable deep dreaming. 

## Project Format:
The code from IM-NET has been re-factored into a base model class which is subclassed into the two archetypes,
the auto encoder (AE) and single view reconstruction (SVR). These models are further subclassed into deep dreaming 
classes AE_DD and SVR_DD. The DeepDream3D routine requires rendering the models in the identical way
as the original ShapeNetV2 renderings used for training. The Facebook Research team has put together a project for
working in the 3D modeling space: PyTorch3D. Their implementation of the standard rendering scheme is deployed in the
dreaming loop. 

## Setup
Clone repository to your local computer. The base docker builds on top of the Amazon deep learning ECR
repository:

```
# for us-west-1 deep learning repos login after configuring AWS CLI
sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 763104351884.dkr.ecr.us-west-1.amazonaws.com

# pull pytorch p36: 8GB
sudo docker pull 763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

```

First build the Dockerfile_base and then the Dockerfile. Afterwards, assuming your hardware is compatible with 
the AWS Linux ECR, you can run the streamlit app on port 8501 of localhost with:

```
nvidia-docker run -it --rm -v /data:/data -p 8501:8501 deep-dream-3d:1.2
```

** The Dockerfile assumes that the IM-NET pre-processed dataset has been donwloaded to /data **

#### Dependencies for Non-Docker install

The following notable packages are required to run successfully:

- [Streamlit](https://www.streamlit.io/)
- [pymcubes](https://pypi.org/project/PyMCubes/)
- [openCV](https://pypi.org/project/opencv-python/)

To install these package and others, please run:

```shell
pip install -r requiremnts
```

**Additionally, you must install:** 

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

Follow the directions according to the link as it pertains to your environment.

ShapeNet v2 is also needed to re-render training data and to extract camera parameters for each
model.

```
wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip -O ~/data/ShapeNetCore.v1.zip
```


## Pre-traiend Model & Data

- Please refer to [IM-NET](https://github.com/czq142857/implicit-decoder.git) to download the pre-trained model, 
training, and testing data. 


## Running the code

### Tests
- You can test the deep dream models with included test datasets by running the following commands:
```
# Test IM_AE: Navigate to the bash run script directory

./test_ae_deepdream.sh
```
```
# Test IM_AE: Navigate to the bash run script directory

./test_svr_deepdream.sh
```
Original run scripts for training and testing can be found in the 
original_run_scripts folder. Additionally, the code for processing raw data has been included
in the preprocessing folder.


### Analysis
Jupyter notebooks for interrogating the original datafiles can be found in the Jupyter_Notebooks folder.
