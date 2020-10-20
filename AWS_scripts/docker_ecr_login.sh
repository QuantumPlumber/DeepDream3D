#!/bin/bash

# login to ecr to pull base images for deep learning
# for personal repository:
#sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 850965295882.dkr.ecr.us-west-1.amazonaws.com

# for us-west-1 deep learning repos
sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 763104351884.dkr.ecr.us-west-1.amazonaws.com

# pull pytorch p36

sudo docker pull 763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04