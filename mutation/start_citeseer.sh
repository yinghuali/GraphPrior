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

python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_gcn_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_dice.pkl' --subject_name 'citeseer_gcn_dice' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_minmax.pkl' --subject_name 'citeseer_gcn_minmax' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_pgdattack.pkl' --subject_name 'citeseer_gcn_pgdattack' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_add.pkl' --subject_name 'citeseer_gcn_randomattack_add' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_flip.pkl' --subject_name 'citeseer_gcn_randomattack_flip' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gcn' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_remove.pkl' --subject_name 'citeseer_gcn_randomattack_remove' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_gat_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_dice.pkl' --subject_name 'citeseer_gat_dice' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_minmax.pkl' --subject_name 'citeseer_gat_minmax' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_pgdattack.pkl' --subject_name 'citeseer_gat_pgdattack' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_add.pkl' --subject_name 'citeseer_gat_randomattack_add' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_flip.pkl' --subject_name 'citeseer_gat_randomattack_flip' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_gat' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_remove.pkl' --subject_name 'citeseer_gat_randomattack_remove' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_tagcn_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_dice.pkl' --subject_name 'citeseer_tagcn_dice' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_minmax.pkl' --subject_name 'citeseer_tagcn_minmax' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_pgdattack.pkl' --subject_name 'citeseer_tagcn_pgdattack' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_add.pkl' --subject_name 'citeseer_tagcn_randomattack_add' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_flip.pkl' --subject_name 'citeseer_tagcn_randomattack_flip' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_tagcn' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_remove.pkl' --subject_name 'citeseer_tagcn_randomattack_remove' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_graphsage_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_dice.pkl' --subject_name 'citeseer_graphsage_dice' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_minmax.pkl' --subject_name 'citeseer_graphsage_minmax' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_pgdattack.pkl' --subject_name 'citeseer_graphsage_pgdattack' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_add.pkl' --subject_name 'citeseer_graphsage_randomattack_add' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_flip.pkl' --subject_name 'citeseer_graphsage_randomattack_flip' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/citeseer_graphsage' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/attack_data/citeseer/citeseer_randomattack_remove.pkl' --subject_name 'citeseer_graphsage_randomattack_remove' --path_y './data/citeseer/y_np.pkl' --path_result 'res/res_misclassification_models_citeseer.csv' --path_pre_result 'res/pre_res_misclassification_models_citeseer.csv'



