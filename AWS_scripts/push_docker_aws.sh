#!/bin/bash

sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 850965295882.dkr.ecr.us-west-1.amazonaws.com

sudo docker tag deepdream3d:2.0 850965295882.dkr.ecr.us-west-1.amazonaws.com/deep-dream-3d:2.0

sudo docker push 850965295882.dkr.ecr.us-west-1.amazonaws.com/deep-dream-3d:2.0

# on AWS

# sudo docker login --username AWS -p $(aws ecr get-login-password --region us-west-1) 850965295882.dkr.ecr.us-west-1.amazonaws.com
# sudo docker pull 850965295882.dkr.ecr.us-west-1.amazonaws.com/deep-dream-3d:1.2