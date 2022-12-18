#!/bin/bash

#python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam0_target_update_freq_3000_original --seed 1

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam0_target_update_freq_3000_original --seed 2 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam0_target_update_freq_3000_original --seed 3 & 

exit 0

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3 &
