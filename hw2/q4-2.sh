#CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
#--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 \
#--exp_name q4_b_50000_r_0.02
#CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
#--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg \
#--exp_name q4_b_50000_r_0.02_rtg
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline \
--exp_name q4_b_50000_r_0.02_nnbaseline --no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_b_50000_r_0.02_rtg_nnbaseline --no_gpu -ngpu