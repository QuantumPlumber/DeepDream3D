#!/bin/bash

# run with AWS CLI configured
aws configure

# create volume
aws ec2 create-volume \
    --volume-type gp2 \
    --size 150 \
    --availability-zone us-west-1c

# attach volume to insance
aws ec2 attach-volume --volume-id vol-1234567890abcdef0 --instance-id i-01474ef662b89480 --device /dev/sdf

# interrogate filesystem
lsblk
df -H
sudo file -s /dev/xvdf

# create filesystem
sudo mkfs -t xfs /dev/xvdf

# make a mount point
sudo mkdir /data

# attach drive
sudo mount /dev/xvdf /data

# change ownership of device drive to be able to write to it
sudo chown ubuntu /data
