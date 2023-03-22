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


python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np '../mutation/data/cora/x_np.pkl' --path_edge_index '../mutation/data/cora/edge_index_np.pkl' --path_y '../mutation/data/cora/y_np.pkl' --save_name './res/cora_gcn.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/cora_gat' --model_name 'gat' --target_model_path './target_models/cora_gat.pt' --path_x_np '../mutation/data/cora/x_np.pkl' --path_edge_index '../mutation/data/cora/edge_index_np.pkl' --path_y '../mutation/data/cora/y_np.pkl' --save_name './res/cora_gat.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/cora_graphsage' --model_name 'graphsage' --target_model_path './target_models/cora_graphsage.pt' --path_x_np '../mutation/data/cora/x_np.pkl' --path_edge_index '../mutation/data/cora/edge_index_np.pkl' --path_y '../mutation/data/cora/y_np.pkl' --save_name './res/cora_graphsage.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/cora_tagcn' --model_name 'tagcn' --target_model_path './target_models/cora_tagcn.pt' --path_x_np '../mutation/data/cora/x_np.pkl' --path_edge_index '../mutation/data/cora/edge_index_np.pkl' --path_y '../mutation/data/cora/y_np.pkl' --save_name './res/cora_tagcn.pkl'

python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np '../mutation/data/citeseer/x_np.pkl' --path_edge_index '../mutation/data/citeseer/edge_index_np.pkl' --path_y '../mutation/data/citeseer/y_np.pkl' --save_name './res/citeseer_gcn.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np '../mutation/data/citeseer/x_np.pkl' --path_edge_index '../mutation/data/citeseer/edge_index_np.pkl' --path_y '../mutation/data/citeseer/y_np.pkl' --save_name './res/citeseer_gat.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np '../mutation/data/citeseer/x_np.pkl' --path_edge_index '../mutation/data/citeseer/edge_index_np.pkl' --path_y '../mutation/data/citeseer/y_np.pkl' --save_name './res/citeseer_graphsage.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np '../mutation/data/citeseer/x_np.pkl' --path_edge_index '../mutation/data/citeseer/edge_index_np.pkl' --path_y '../mutation/data/citeseer/y_np.pkl' --save_name './res/citeseer_tagcn.pkl'


python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np '../mutation/data/lastfm/x_np.pkl' --path_edge_index '../mutation/data/lastfm/edge_index_np.pkl' --path_y '../mutation/data/lastfm/y_np.pkl' --save_name './res/lastfm_gcn.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np '../mutation/data/lastfm/x_np.pkl' --path_edge_index '../mutation/data/lastfm/edge_index_np.pkl' --path_y '../mutation/data/lastfm/y_np.pkl' --save_name './res/lastfm_gat.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np '../mutation/data/lastfm/x_np.pkl' --path_edge_index '../mutation/data/lastfm/edge_index_np.pkl' --path_y '../mutation/data/lastfm/y_np.pkl' --save_name './res/lastfm_graphsage.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np '../mutation/data/lastfm/x_np.pkl' --path_edge_index '../mutation/data/lastfm/edge_index_np.pkl' --path_y '../mutation/data/lastfm/y_np.pkl' --save_name './res/lastfm_tagcn.pkl'


python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np '../mutation/data/pubmed/x_np.pkl' --path_edge_index '../mutation/data/pubmed/edge_index_np.pkl' --path_y '../mutation/data/pubmed/y_np.pkl' --save_name './res/pubmed_gcn.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np '../mutation/data/pubmed/x_np.pkl' --path_edge_index '../mutation/data/pubmed/edge_index_np.pkl' --path_y '../mutation/data/pubmed/y_np.pkl' --save_name './res/pubmed_gat.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np '../mutation/data/pubmed/x_np.pkl' --path_edge_index '../mutation/data/pubmed/edge_index_np.pkl' --path_y '../mutation/data/pubmed/y_np.pkl' --save_name './res/pubmed_graphsage.pkl'
python retraining_new.py --path_model_file '../mutation/all_mutation_models/repeat_1/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np '../mutation/data/pubmed/x_np.pkl' --path_edge_index '../mutation/data/pubmed/edge_index_np.pkl' --path_y '../mutation/data/pubmed/y_np.pkl' --save_name './res/pubmed_tagcn.pkl'







