#!/bin/bash

# sync both ways,
# u option respects more recently updated files, this should make both directories consistent with
# most up to date files
rsync -avu /home/ag/Insight/PoorMansDeepSDF/ ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com:~/Insight/PoorMansDeepSDF
rsync -avu ubuntu@ec2-54-215-102-50.us-west-1.compute.amazonaws.com:~/Insight/PoorMansDeepSDF/ /home/ag/Insight/PoorMansDeepSDF