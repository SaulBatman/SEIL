# #default workspace is 0.3

# #Test different choices of dpos and drot without sim

# # dpos=0.05, drot_n=8, simulate_n=4, no sim
# python mytest.py --log_sub="dpos0.05_drot_n8_simulate_n4_no_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=panda

# # dpos=0.02, drot_n=4, simulate_n=4, no sim
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_no_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=panda


# #Test different choices of dpos with 4sim
# # dpos=0.05, drot_n=4, simulate_n=4, flower sim
# python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# # dpos=0.02, drot_n=4, simulate_n=4, flower sim
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


# #Test different choices of simulate_n with sim
# # dpos=0.02, drot_n=4, simulate_n=8, flower sim
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n8_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=8 --eval_freq=500 --robot=panda

# dpos=0.02, drot_n=4, simulate_n=16, flower sim
python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n16_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=16 --eval_freq=500 --robot=panda


# Test different choices of sigma with sim
# dpos=0.02, drot_n=4, simulate_n=4, flower sim, sigma=0.1
python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.1" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.1 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# dpos=0.02, drot_n=4, simulate_n=4, flower sim, sigma=0.1
python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.4" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


# Test different choices of workspace_size with sim
# dpos=0.02, drot_n=4, simulate_n=8, flower sim, sigma=0.1
python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.1_ws0.5" --workspace_size=0.5 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# dpos=0.02, drot_n=4, simulate_n=8, flower sim, sigma=0.1
python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.1_ws0.7" --workspace_size=0.7 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


