#!/bin/bash

echo 3

python rob831/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-rob831-v0 \
--add_sl_noise --num_agent_train_steps_per_iter 20 \
--batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 \
--video_log_freq -1 --mpc_action_sampling_strategy 'random' --no_gpu -ngpu
# &> q3_1.out &
exit 0

python rob831/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-rob831-v0 \
--add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 \
--batch_size_initial 5000 --batch_size 5000 --n_iter 15 \
--video_log_freq -1 --mpc_action_sampling_strategy 'random' --no_gpu -ngpu
# &> q3_2.out &

#exit 0
python rob831/scripts/run_hw4_mb.py --exp_name q3_cheetah --env_name cheetah-rob831-v0 \
--mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 \
--batch_size_initial 5000 --batch_size 5000 --n_iter 20 \
--video_log_freq -1 --mpc_action_sampling_strategy 'random' --no_gpu -ngpu &> q3_3.out &