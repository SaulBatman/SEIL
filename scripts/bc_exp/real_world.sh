# # equi_d bc planner20
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1  \
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy\

# equi_d bc planner10, real data
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=10\
                --log_pre=./outputs --log_sub=real_equi_d_planner10

# equi_d bc planner5, real data
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=5\
                --log_pre=./outputs --log_sub=real_equi_d_planner5

# cnn bc planner20, real data
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy\
                --log_pre=./outputs --log_sub=real_cnn_planner20

# cnn bc planner10, real data
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=5\
                --log_pre=./outputs --log_sub=real_cnn_planner10

# cnn bc planner20, simulation data
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
                --log_pre=./outputs --log_sub=simulation_cnn_planner10