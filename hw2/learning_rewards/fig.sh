### q1
#python figure.py --figname q1_sb.png --title CartPole-v0+sb --cond q1_sb+eval --direc ./1st/ --loc lower+right

#python figure.py --figname q1_lb.png --title CartPole-v0+lb --cond q1_lb+eval --direc ./1st/ --loc lower+right

### q2
#python figure.py --figname q2_batchsize.png --title InvertedPendulum-v4+Search+for+the+Smallest+b --cond q2+r_5e-3+eval --excep seed

#python figure.py --figname q2_lr_b500.png --title InvertedPendulum-v4+Search+for+the+Largest+lr+with+b=500 --cond q2_b_500_+eval

#python figure.py --figname q2_lr_b5000.png --title InvertedPendulum-v4+Search+for+the+Largest+lr+with+b=5000 --cond q2_b_5000_+eval --excep e-4+e-5+e-6

### q3 
#python figure.py --figname q3.png --title LunarLanderContinuous-v2+Learning_Curve --cond q3+eval 

### q4
#python figure.py --figname q4.png --title HalfCheetah-v4+Search+for+Best+Batchsize+and+Learning+Rate --cond q4_search+eval --clip _rtg_nnbaseline

python figure.py --figname q4-2.png --title HalfCheetah-v4+Reward_to_go+and+NN-Baseline --cond q4_b+eval


### q5
python figure.py --figname q5_lambda_3.png --title Hopper-v4+GAE --cond q5+eval --excep 0_+vanilla

python figure.py --figname q5_lambda_4.png --title Hopper-v4+GAE --cond q5+lambda+eval


