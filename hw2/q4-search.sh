CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 5e-3 -rtg --nn_baseline \
--exp_name q4_search_b_10000_lr_5e-3_rtg_nnbaseline # --no_gpu -ngpu 

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 5e-3 -rtg --nn_baseline \
--exp_name q4_search_b_30000_lr_5e-3_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 5e-3 -rtg --nn_baseline \
--exp_name q4_search_b_50000_lr_5e-3_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 1e-2 -rtg --nn_baseline \
--exp_name q4_search_b_10000_lr_1e-2_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 1e-2 -rtg --nn_baseline \
--exp_name q4_search_b_30000_lr_1e-2_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 1e-2 -rtg --nn_baseline \
--exp_name q4_search_b_50000_lr_1e-2_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 2e-2 -rtg --nn_baseline \
--exp_name q4_search_b_10000_lr_2e-2_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 2e-2 -rtg --nn_baseline \
--exp_name q4_search_b_30000_lr_2e-2_rtg_nnbaseline # --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 2e-2 -rtg --nn_baseline \
--exp_name q4_search_b_50000_lr_2e-2_rtg_nnbaseline # --no_gpu -ngpu
