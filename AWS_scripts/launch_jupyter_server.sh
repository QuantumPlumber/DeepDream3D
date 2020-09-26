#!/bin/bash

########################################################################################################################
# After setting up ssh keys on the AWS server this script will
# start the jupyter server and open a web browser to begin development

########################################################################################################################

# run on remote computer
ssh -i ~/.ssh/Averell-Gatton-IAM-keypair.pem ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com \
"cd ~/Insight/PoorMansDeepSDF/;jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key"

# ensure redirection at local computer
ssh -i ~/.ssh/Averell-Gatton-IAM-keypair.pem -N -f -L 8888:localhost:8888 ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com

# launch window
/usr/bin/firefox --new-window https://localhost:8888

# enter password at prompt