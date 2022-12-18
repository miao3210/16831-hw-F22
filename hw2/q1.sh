CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-dsa --exp_name q1_sb_no_rtg_dsa --no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -dsa --exp_name q1_sb_rtg_dsa --no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name q1_sb_rtg_na --no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-dsa --exp_name q1_lb_no_rtg_dsa #--no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg -dsa --exp_name q1_lb_rtg_dsa #--no_gpu -ngpu
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg --exp_name q1_lb_rtg_na #--no_gpu -ngpu
