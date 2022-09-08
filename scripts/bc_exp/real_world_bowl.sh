# # explicit aug64, TS0, max20000steps. SEED1.
# # D4 planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner10_aug64_TS4_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner5_aug64_TS4_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner1_aug64_TS4_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # explicit aug64, TS4, max20000steps. SEED2.
# # D4 planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner10_aug64_TS4_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner5_aug64_TS4_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner1_aug64_TS4_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # explicit aug64, TS4, max20000steps. SEED3.
# # D4 planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner10_aug64_TS4_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner5_aug64_TS4_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # D4 planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=4 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_D4_planner1_aug64_TS4_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5


# #-----------------------CNN------------------------#
# # explicit aug64, TS0, max20000steps. SEED1.
# # CNN planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner10_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner5_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner1_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # explicit aug64, TS4, max20000steps. SEED2.
# # CNN planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner10_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner5_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner1_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # explicit aug64, TS0, max20000steps. SEED3.
# # CNN planner10
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner10_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner5
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner5_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# # CNN planner1
# python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
#                 --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_con --model=cnn\
#                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
#                 --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
#                 --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1\
#                 --log_pre=./outputs --log_sub=bowl_explicit_CNN_planner1_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5






#-----------------ibc---------------#
# explicit aug64, TS0, max20000steps. SEED1.
# CNN planner10
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner10_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner5
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner5_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner1
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=1 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner1_aug64_TS0_SEED1_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# explicit aug64, TS4, max20000steps. SEED2.
# CNN planner10
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner10_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner5
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner5_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner1
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=2 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_explicit_CNNSSM_planner1_aug64_TS0_SEED2_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# explicit aug64, TS0, max20000steps. SEED3.
# CNN planner10
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=10 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNN_planner10_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner5
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=5 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner5_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5

# CNN planner1
python main.py --env=real --num_processes=5 --num_eval_process=1 --render=f\
                --max_episode_steps=50 --planner_episode=0 --dpos=0.02 --drot_n=8 --alg=bc_implicit --model=cnn_ssm\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --buffer_aug_n=64 --lr=1e-3 --gamma=0.99 --seed=3 --transparent_bin=t \
                --max_train_step=20000 --simulate_n=0 --eval_freq=-1 --num_eval_episodes=1 --robot=kuka --sigma=0.4 --view_type=render_center\
                --load_buffer=/home/xxslab/Documents/BC/equi_close_loop/scripts/buffer/bowl_new10.npy --load_n=1 --ibc_ts=4096 --ibc_is=16384\
                --log_pre=./outputs --log_sub=bowl_implicit_CNNSSM_planner1_aug64_TS0_SEED3_0907 --ts_from_cloud=t --workspace_size=0.4 --view_scale=1.5


