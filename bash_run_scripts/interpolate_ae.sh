python ../Insight_Project_Framework/main.py \
                --ae \
                --interpol \
                --data_dir /data/IM-NET-pytorch/data/all_vox256_img \
                --checkpoint_dir /data/IM-NET-pytorch/checkpoint \
                --sample_dir /data/IM-NET-pytorch/data/all_vox256_img \
                --interpol_directory /data/IM-NET-pytorch/data/all_vox256_img \
                --interpol_z1 100 \
                --interpol_z2 1000 \
                --interpol_steps 16
