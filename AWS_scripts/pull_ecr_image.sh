#!/bin/bash
# first login
sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 850965295882.dkr.ecr.us-west-1.amazonaws.com

# pull image
sudo docker pull 850965295882.dkr.ecr.us-west-1.amazonaws.com/deep-dream-3d:1.0