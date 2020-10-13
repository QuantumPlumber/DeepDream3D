#!/bin/bash

# sync both ways,
# u option respects more recently updated files, this should make both directories consistent with
# most up to date files
rsync -avu -e "ssh -i ~/.ssh/Averell-Gatton-IAM-keypair.pem" /home/ag/Insight/DeepDream3D/ ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com:~/Insight/DeepDream3D
rsync -avu -e "ssh -i ~/.ssh/Averell-Gatton-IAM-keypair.pem" ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com:~/Insight/DeepDream3D/ /home/ag/Insight/DeepDream3D