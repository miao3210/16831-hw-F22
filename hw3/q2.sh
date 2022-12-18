#!/bin/bash

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3 &

# critic
#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3 &

# agent
#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_4 --double_q --seed 1 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_5 --double_q --seed 2 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_6 --double_q --seed 3 &


# unsqueeze -1 target split
#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_7 --double_q --seed 1 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_8 --double_q --seed 2 &

#nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_9 --double_q --seed 3 &


# target split
nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_10 --double_q --seed 1 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_11 --double_q --seed 2 &

nohup python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_12 --double_q --seed 3 &
