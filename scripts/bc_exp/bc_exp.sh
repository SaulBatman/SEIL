# # #default workspace is 0.3

# # #Test different choices of dpos and drot without sim

# # #EXP1 dpos=0.05, drot_n=8, simulate_n=4, no sim
# # python mytest.py --log_sub="dpos0.05_drot_n8_simulate_n4_no_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=panda

# # #EXP2 dpos=0.02, drot_n=4, simulate_n=4, no sim
# # python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_no_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=panda


# # #Test different choices of dpos with 4sim
# # #EXP3 dpos=0.05, drot_n=4, simulate_n=4, flower sim
# # python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# # #EXP4 dpos=0.02, drot_n=4, simulate_n=4, flower sim
# # python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


# # #Test different choices of simulate_n with sim
# # #EXP5 dpos=0.02, drot_n=4, simulate_n=8, flower sim
# # python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n8_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=8 --eval_freq=500 --robot=panda

# #EXP6 dpos=0.02, drot_n=4, simulate_n=16, flower sim
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n16_with_sim" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=16 --eval_freq=500 --robot=panda


# # Test different choices of sigma with sim
# #EXP7 dpos=0.02, drot_n=4, simulate_n=4, flower sim, sigma=0.1
# python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.1" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.1 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# #EXP8 dpos=0.02, drot_n=4, simulate_n=4, flower sim, sigma=0.1
# python mytest.py --log_sub="dpos0.05_drot_n4_simulate_n4_with_sim_sig0.4" --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


# # Test different choices of workspace_size with sim
# #EXP9 dpos=0.02, drot_n=4, simulate_n=8, flower sim, sigma=0.1
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_with_sim_sig0.1_ws0.5" --workspace_size=0.5 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# #EXP10 dpos=0.02, drot_n=4, simulate_n=8, flower sim, sigma=0.1
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_with_sim_sig0.1_ws0.7" --workspace_size=0.7 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda


# # 0524
# # EXP11 dpos=0.02, drot_n=4, simulate_n=0, no flower sim, ws0.7 | for EXP10 comparison
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n0_no_sim_ws0.7" --workspace_size=0.7 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=panda

# # EXP12 dpos=0.02, drot_n=4, simulate_n=16, flower sim, sigma0.4 | for EXP10 comparison
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n16_sim_sigma0.4_ws0.7" --workspace_size=0.7 --env=close_loop_block_in_bowl --num_obj=2 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=16 --eval_freq=500 --robot=panda

# # EXP13 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.2, flower sim | GRASPING EXP
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_sim_sigma0.2_ws0.3" --workspace_size=0.3 --env=close_loop_clutter_picking --num_obj=5 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# # EXP13 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.4, flower sim | GRASPING EXP
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n4_sim_sigma0.4_ws0.3" --workspace_size=0.3 --env=close_loop_clutter_picking --num_obj=5 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# # EXP13 dpos=0.02, drot_n=4, simulate_n=8, sigma=0.4, flower sim | GRASPING EXP
# python mytest.py --log_sub="dpos0.02_drot_n4_simulate_n8_sim_sigma0.4_ws0.3" --workspace_size=0.3 --env=close_loop_clutter_picking --num_obj=5 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=8 --eval_freq=500 --robot=panda

# 0525
# EXP14 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.2, flower sim | PUSHING EXP
python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n4_sim_sigma0.2_ws0.3_seed1" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.2 --max_train_step=10000 --simulate_n=4 --eval_freq=500 --robot=panda
# EXP14 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.2, flower sim | PUSHING EXP
python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n4_sim_sigma0.2_ws0.3_seed2" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=2 --sigma=0.2 --max_train_step=10000 --simulate_n=4 --eval_freq=500 --robot=panda
# EXP14 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.2, flower sim | PUSHING EXP
python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n4_sim_sigma0.2_ws0.3_seed3" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=3 --sigma=0.2 --max_train_step=10000 --simulate_n=4 --eval_freq=500 --robot=panda
# EXP14 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.2, flower sim | PUSHING EXP
python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n4_sim_sigma0.2_ws0.3_seed4" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=4 --sigma=0.2 --max_train_step=10000 --simulate_n=4 --eval_freq=500 --robot=panda



# # EXP15 dpos=0.02, drot_n=4, simulate_n=4, sigma=0.4, flower sim | PUSHING EXP
# python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n4_sim_sigma0.4_ws0.3" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=4 --eval_freq=500 --robot=panda

# # EXP16 dpos=0.02, drot_n=4, simulate_n=8, sigma=0.4, flower sim | PUSHING EXP
# python mytest.py --log_sub="push_dpos0.02_drot_n4_simulate_n8_sim_sigma0.4_ws0.3" --workspace_size=0.3 --env=close_loop_block_pushing --num_obj=1 --num_processes=1 --num_eval_process=5 --render=f --max_episode_steps=50 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con --model=equi --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --sigma=0.4 --max_train_step=2000 --simulate_n=8 --eval_freq=500 --robot=panda
