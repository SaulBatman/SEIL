# # planner=2, no TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner2_no_TS
# # planner=2, TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner2_TS
# # batch size is 128 
# # planner=2, no TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=128 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner2_no_TS_batch128
# # planner=2, TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=128 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner2_TS_batch128
# # batch size is 128 
# # planner=3, no TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=3 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=128 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner3_no_TS_batch128
# # planner=3, TS
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=3 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=128 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center --data_balancing\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs --log_sub=bowl_planner3_TS_batch128

# # planner=2, TS, no data balancing, batch_size=64
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner3_TS

# # planner=2, no TS, aggressive aug(16), batch_size=64
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=2 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner2_no_TS_aug16 --buffer_aug_n=16

# # planner=1, TS, batch_size=64, equi_d
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=1 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner1_TS_aug16 --buffer_aug_n=4 --num_eval_episodes=50

# # planner=1, no TS, aggressive aug(16), batch_size=64
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=1 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner1_no_TS_aug16 --buffer_aug_n=16 --num_eval_episodes=50

# # planner=1, no TS, batch_size=64
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=1 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner1_no_TS --buffer_aug_n=4 --num_eval_episodes=50

# # planner=1, no TS, aggressive aug(16), batch_size=64, equi_d
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=1 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=0 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner1_no_TS_aug16_equi_d --buffer_aug_n=16 --num_eval_episodes=50

# # planner=1, TS, batch_size=64, equi_d
# python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
#                 --max_episode_steps=50 --planner_episode=1 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi_d\
#                  --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
#                  --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
#                  --log_sub=bowl_planner1_balanced_TS_aug4 --buffer_aug_n=4 --num_eval_episodes=50 --data_balancing

# planner=3, TS, batch_size=64, equi_d
python main.py --env=close_loop_block_in_bowl --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f\
                --max_episode_steps=50 --planner_episode=3 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi_d\
                 --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
                 --max_train_step=10000 --simulate_n=4 --eval_freq=1000 --robot=panda --sigma=0.4 --log_pre=./outputs\
                 --log_sub=bowl_planner3_balanced_TS_aug4 --buffer_aug_n=4 --num_eval_episodes=50 --data_balancing