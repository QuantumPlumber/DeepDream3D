#!/bin/bash

nvidia-docker run -it --rm -v /data:/data -p 80:8501 850965295882.dkr.ecr.us-west-1.amazonaws.com/deep-dream-3d:1.2