#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch

python mutation_rule_analysis.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --data_name 'cora'
python mutation_rule_analysis.py --path_model_file './mutation_models/cora_gat' --model_name 'gat' --target_model_path './target_models/cora_gat.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --data_name 'cora'
python mutation_rule_analysis.py --path_model_file './mutation_models/cora_graphsage' --model_name 'graphsage' --target_model_path './target_models/cora_graphsage.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --data_name 'cora'
python mutation_rule_analysis.py --path_model_file './mutation_models/cora_tagcn' --model_name 'tagcn' --target_model_path './target_models/cora_tagcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y './data/cora/y_np.pkl' --data_name 'cora'

python mutation_rule_analysis.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y './data/citeseer/y_np.pkl' --data_name 'citeseer'
python mutation_rule_analysis.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y './data/citeseer/y_np.pkl' --data_name 'citeseer'
python mutation_rule_analysis.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y './data/citeseer/y_np.pkl' --data_name 'citeseer'
python mutation_rule_analysis.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y './data/citeseer/y_np.pkl' --data_name 'citeseer'

python mutation_rule_analysis.py --path_model_file './mutation_models/lastfm_gcn' --model_name 'gcn' --target_model_path './target_models/lastfm_gcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --data_name 'lastfm'
python mutation_rule_analysis.py --path_model_file './mutation_models/lastfm_gat' --model_name 'gat' --target_model_path './target_models/lastfm_gat.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --data_name 'lastfm'
python mutation_rule_analysis.py --path_model_file './mutation_models/lastfm_graphsage' --model_name 'graphsage' --target_model_path './target_models/lastfm_graphsage.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --data_name 'lastfm'
python mutation_rule_analysis.py --path_model_file './mutation_models/lastfm_tagcn' --model_name 'tagcn' --target_model_path './target_models/lastfm_tagcn.pt' --path_x_np './data/lastfm/x_np.pkl' --path_edge_index '../data/lastfm/edge_index_np.pkl' --path_y './data/lastfm/y_np.pkl' --data_name 'lastfm'

python mutation_rule_analysis.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y './data/pubmed/y_np.pkl' --data_name 'pubmed'
python mutation_rule_analysis.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y './data/pubmed/y_np.pkl' --data_name 'pubmed'
python mutation_rule_analysis.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y './data/pubmed/y_np.pkl' --data_name 'pubmed'
python mutation_rule_analysis.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y './data/pubmed/y_np.pkl' --data_name 'pubmed'
