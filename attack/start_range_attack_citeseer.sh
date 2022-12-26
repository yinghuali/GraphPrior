#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00
#SBATCH -C skylake
#SBATCH --mem 100G

python get_range_attack.py --path_x_np '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --save_edge_index '/home/users/yili/pycharm/GraphPrior/data/ratio_attack/citeseer/citeseer'
