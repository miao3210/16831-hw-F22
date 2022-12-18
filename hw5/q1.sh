#!/bin/bash

python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_medium_cql -gpu_id 1 

python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_hard_cql -gpu_id 1 

exit 0

python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_hard_dqn_transform -gpu_id 1 --exploit_rew_shift 1 --exploit_rew_scale 100 

python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_hard_cql_transform -gpu_id 1 --exploit_rew_shift 1 --exploit_rew_scale 100


python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_medium_cql_transform -gpu_id 1 --exploit_rew_shift 1 --exploit_rew_scale 100 


python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_medium_dqn_transform -gpu_id 1 --exploit_rew_shift 1 --exploit_rew_scale 100 


exit 0

nohup python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_medium_dqn -gpu_id 1 &


nohup python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_medium_cql -gpu_id 1 &


nohup python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_hard_dqn -gpu_id 1 &


nohup python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_hard_cql -gpu_id 1 &