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
You can install the module using pip:
```
pip install DeepDream3D
```

Or clone repository and update python path
```
repo_name=Insight_Project_Framework # URL of your new repository
username=mrubash1 # Username for your personal github account
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```

#### Dependencies

The following notable packages are required to run successfully:

- [Streamlit](https://www.streamlit.io/)
- [pymcubes](https://pypi.org/project/PyMCubes/)
- [openCV](https://pypi.org/project/opencv-python/)

To install these package and others, pleae run:
```shell
pip install -r requiremnts
```

**Additionally, you must install:** 

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

Follow the directions according to the link as it pertains to your environment.

ShapeNet v2 is also needed to re-render training data and to extract camera parameters for each
model.

```
wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip -O [your/path/to/destination/dir]
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Running the code

### Configs


### Test
- You can test the models by running the following commands:
```
# Test IM_AE: Navigate to the bash run script directory

./test_ae.sh
```
```
# Test IM_AE: Navigate to the bash run script directory

./test_svr.sh
```

```
# Test IM_AE: Navigate to the bash run script directory

./deepdream_test.sh
```

```
# Test IM_AE: Navigate to the bash run script directory

./deepdream_svr_test.sh
```


### Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

### Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

### Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

### Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
