#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00
#SBATCH -C skylake
#SBATCH --mem 700G

python get_attack.py --path_x_np '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --save_edge_index '/home/users/yili/pycharm/GNNEST/data/attack_data/pubmed/pubmed'