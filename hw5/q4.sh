#!/bin/bash

#python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
#--exp_name q4_awac_medium_lam0.1 --use_rnd \
#--offline_exploitation --awac_lambda=0.1 --num_exploration_steps=20000 -gpu_id 1
#python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
#--exp_name q4_awac_medium_lam1 --use_rnd \
#--offline_exploitation --awac_lambda=1 --num_exploration_steps=20000 -gpu_id 0


nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam0.1 --use_rnd \
--offline_exploitation --awac_lambda=0.1 --num_exploration_steps=20000 -gpu_id 0 &

CUDA_LAUNCH_BLOCKING=1 nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam1 --use_rnd \
--offline_exploitation --awac_lambda=1 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam2 --use_rnd \
--offline_exploitation --awac_lambda=2 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam10 --use_rnd \
--offline_exploitation --awac_lambda=10 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam20 --use_rnd \
--offline_exploitation --awac_lambda=20 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_lam50 --use_rnd \
--offline_exploitation --awac_lambda=50 --num_exploration_steps=20000 -gpu_id 0 &





nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam0.1 --use_rnd \
--offline_exploitation --awac_lambda=0.1 --num_exploration_steps=20000 -gpu_id 0 &


nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam1 --use_rnd \
--offline_exploitation --awac_lambda=1 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam2 --use_rnd \
--offline_exploitation --awac_lambda=2 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam10 --use_rnd \
--offline_exploitation --awac_lambda=10 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam20 --use_rnd \
--offline_exploitation --awac_lambda=20 --num_exploration_steps=20000 -gpu_id 0 &

nohup python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_lam50 --use_rnd \
--offline_exploitation --awac_lambda=50 --num_exploration_steps=20000 -gpu_id 0 &
