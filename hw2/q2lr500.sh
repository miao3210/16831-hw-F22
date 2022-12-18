CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 1e-3 -rtg \
--exp_name q2_b_500_r_1e-3 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 2e-3 -rtg \
--exp_name q2_b_500_r_2e-3 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 5e-3 -rtg \
--exp_name q2_b_500_r_5e-3 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 1e-2 -rtg \
--exp_name q2_b_500_r_1e-2 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 2e-2 -rtg \
--exp_name q2_b_500_r_2e-2 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 5e-2 -rtg \
--exp_name q2_b_500_r_5e-2 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 1e-1 -rtg \
--exp_name q2_b_500_r_1e-1 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 2e-1 -rtg \
--exp_name q2_b_500_r_2e-1 --no_gpu -ngpu

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 5e-1 -rtg \
--exp_name q2_b_500_r_5e-1 --no_gpu -ngpu
