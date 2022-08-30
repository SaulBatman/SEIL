
python main.py --env=close_loop_shoe_packing --num_obj=1 --num_processes=5 --num_eval_process=5 --render=f\
                --max_episode_steps=100 --planner_episode=100 --dpos=0.02 --drot_n=8 --alg=bc_con --model=equi_d\
                --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1 --view_type=render_center\
                --max_train_step=10000 --simulate_n=0 --eval_freq=2000 --robot=kuka --sigma=0.4\
                --buffer_aug_n=64 --num_eval_episodes=50 --workspace_size=0.4 --view_scale=1.5 --log_pre=./outputs



python test.py --workspace_size=0.4 --env=close_loop_shoe_packing --num_obj=2 --num_processes=1 --num_eval_process=5\
                --render=t --max_episode_steps=100 --planner_episode=20 --dpos=0.02 --drot_n=4 --alg=bc_con\
                --model=equi_d --equi_n=4 --n_hidden=64 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --seed=1\
                --sigma=0.4 --max_train_step=2000 --simulate_n=0 --eval_freq=500 --robot=kuka --view_type=render_center\
                --load_model_pre='/home/xxslab/Documents/equi_close_loop/scripts/outputs/bc_con_equi_d/train_close_loop_shoe_packing_2022-08-28.15:36:30/models/snapshot_close_loop_shoe_packing'