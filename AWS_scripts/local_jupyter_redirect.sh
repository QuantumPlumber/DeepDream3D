#!/bin/bash

ssh -i ~/.ssh/Averell-Gatton-IAM-keypair.pem -N -f -L 8888:localhost:8888 ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com
