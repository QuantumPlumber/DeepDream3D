#!/bin/bash

cd ../
python -m DeepDream3D configs/deepdream_exp_a.yml
python -m DeepDream3D configs/deepdream_exp_b.yml
python -m DeepDream3D configs/deepdream_exp_c.yml
python -m DeepDream3D configs/deepdream_exp_d.yml
python -m DeepDream3D configs/deepdream_exp_e.yml
python -m DeepDream3D configs/deepdream_exp_f.yml
# python -m DeepDream3D configs/deepdream_exp_g.yml