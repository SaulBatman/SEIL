# # explicit aug64, TS0, save every 5000 steps, max20000steps. no fix tanh
# # D4 planner3
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=3\
#                 --log_pre=./outputs --log_sub=Trash5_explicit_D4_planner3_aug64_TS0_0821

# # D4 planner2
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=2\
#                 --log_pre=./outputs --log_sub=Trash5_explicit_D4_planner2_aug64_TS0_0821

# # D4 planner1
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=Trash5_explicit_D4_planner1_aug64_TS0_0821

# implicit aug64, TS0, save every 5000 steps, max20000steps. fix tanh
# D4 planner3
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --ibc_ts=4096 --ibc_is=16384\
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=3\
                --log_pre=./outputs --log_sub=Trash5_implicit_D4_planner3_aug64_TS0_0821

# D4 planner2
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --ibc_ts=4096 --ibc_is=16384\
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=2\
                --log_pre=./outputs --log_sub=Trash5_implicit_D4_planner2_aug64_TS0_0821

# D4 planner1
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --ibc_ts=4096 --ibc_is=16384\
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --save_multi_freq=5000 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy --load_n=1\
                --log_pre=./outputs --log_sub=Trash5_implicit_D4_planner1_aug64_TS0_0821





