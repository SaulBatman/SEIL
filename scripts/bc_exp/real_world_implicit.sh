# # explicit equi_d bc planner20(load_n), real data
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
#                 --log_pre=./outputs --log_sub=real_equi_d_explicit_planner20_new

# # implicit equi_d bc planner10(load_n), real data
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=real_equi_d_explicit_planner10_new

# # implicit equi_d bc planner20(load_n), real data
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=real_equi_d_implicit_planner10_new



# # implicit equi_d bc planner10(load_n), real data
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
#                 --log_pre=./outputs --log_sub=real_equi_d_implicit_planner20_new

#-----------0730--------------

# explicit equi_d bc planner5(load_n), real data: explicit equiD planner20/10 work, so try planner5
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=5\
                --log_pre=./outputs --log_sub=real_equi_d_explicit_planner5_0730


# explicit cnn bc planner20(load_n), real data: implicit cnn works, so try explicit 
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
                --log_pre=./outputs --log_sub=real_cnn_explicit_planner20_0730

# explicit cnn bc planner20(load_n), real data: implicit cnn works, so try explicit 
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=10\
                --log_pre=./outputs --log_sub=real_cnn_explicit_planner10_0730

# implicit equi_d ssm bc planner20(load_n), real data: im_equiD_planner20 does not work ,so try equi_d ssm because cnn_ssm planner20 works
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
                --log_pre=./outputs --log_sub=real_equi_d_ssm_implicit_planner20_0730

# implicit cnn ssm bc planner10(load_n), real data: im cnn ssm planner20 works, so try planner10
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=10\
                --log_pre=./outputs --log_sub=real_cnn_implicit_planner10_0730


# implicit equi_d ssm bc planner20(load_n), real data: im_equiD_planner20 does not work ,so try equi_d ssm because cnn_ssm planner20 works
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d_ssm_1\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
                --log_pre=./outputs --log_sub=real_equi_d_ssm1_implicit_planner20_0730

# implicit equi_d ssm bc planner20(load_n), real data: im_equiD_planner20 does not work ,so try equi_d ssm because cnn_ssm planner20 works
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d_ssm_2\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=panda --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/bowl_20.npy --load_n=20\
                --log_pre=./outputs --log_sub=real_equi_d_ssm2_implicit_planner20_0730


#------------------------------------------------