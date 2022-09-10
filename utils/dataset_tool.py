import numpy as np
import os

def combineTwoDataset(data_path1, data_path2, saving_name="new_data.npy", saving_path="./"):
    data_path = [data_path1, data_path2]
    data_all = []
    for path in data_path:
        print(f"loading data: {path}")
        myload = np.load(path, allow_pickle=True)
        print(f"-------------This dataset has {len(myload)} episodes in it.--------------")
        for traj in myload:
            print(f"trajectory has a length of {len(traj)}")
            data_all.append(traj)
    print(f"saving whole dataset to {saving_path}")
    np.save(os.path.join(saving_path, saving_name), data_all, allow_pickle=True)


if __name__ == "__main__":
    combineTwoDataset(data_path1="/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_3epi_obs0.5.npy",
                      data_path2="/home/mingxi/ws/equi_close_loop_new/BC/equi_close_loop/scripts/outputs/buffer/trash5_7epi_obs0.6.npy",
                      saving_name="trash_10epi.npy",
                      saving_path="./")