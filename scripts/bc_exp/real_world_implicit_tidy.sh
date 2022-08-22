
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_equi_d_explicit_planner5_aug16_0802


python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_equi_d_explicit_planner5_aug64_0802

python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_cnn_ssm_explicit_planner5_aug64_0802

python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_equi_d_implicit_planner5_aug16_0802

python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=equi_d_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=16 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_equi_d_ssm_implicit_planner5_aug16_0802


python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                 --max_train_step=10000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=10 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/obj2_in_box_planner5.npy --load_n=5\
                --log_pre=./outputs --log_sub=TIDY_real_cnn_ssm_implicit_planner5_aug64_0802