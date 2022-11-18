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

python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --subject_name 'pubmed_gcn_clean' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_dice.pkl' --subject_name 'pubmed_gcn_dice' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_add.pkl' --subject_name 'pubmed_gcn_nodeembeddingattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_remove.pkl' --subject_name 'pubmed_gcn_nodeembeddingattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_add.pkl' --subject_name 'pubmed_gcn_randomattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_flip.pkl' --subject_name 'pubmed_gcn_randomattack_flip' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gcn' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_remove.pkl' --subject_name 'pubmed_gcn_randomattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --subject_name 'pubmed_gat_clean' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_dice.pkl' --subject_name 'pubmed_gat_dice' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_add.pkl' --subject_name 'pubmed_gat_nodeembeddingattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_remove.pkl' --subject_name 'pubmed_gat_nodeembeddingattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_add.pkl' --subject_name 'pubmed_gat_randomattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_flip.pkl' --subject_name 'pubmed_gat_randomattack_flip' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_gat' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_remove.pkl' --subject_name 'pubmed_gat_randomattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --subject_name 'pubmed_tagcn_clean' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_dice.pkl' --subject_name 'pubmed_tagcn_dice' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_add.pkl' --subject_name 'pubmed_tagcn_nodeembeddingattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_remove.pkl' --subject_name 'pubmed_tagcn_nodeembeddingattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_add.pkl' --subject_name 'pubmed_tagcn_randomattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_flip.pkl' --subject_name 'pubmed_tagcn_randomattack_flip' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_tagcn' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_remove.pkl' --subject_name 'pubmed_tagcn_randomattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'

python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --subject_name 'pubmed_graphsage_clean' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_dice.pkl' --subject_name 'pubmed_graphsage_dice' --path_y './data/pubmed/y_np.pkl'  --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_add.pkl' --subject_name 'pubmed_graphsage_nodeembeddingattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_nodeembeddingattack_remove.pkl' --subject_name 'pubmed_graphsage_nodeembeddingattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_add.pkl' --subject_name 'pubmed_graphsage_randomattack_add' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_flip.pkl' --subject_name 'pubmed_graphsage_randomattack_flip' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'
python mutation_feature_model_train.py --path_model_file './mutation_models/pubmed_graphsage' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x_np './data/pubmed/x_np.pkl' --path_edge_index '../data/attack_data/pubmed/pubmed_randomattack_remove.pkl' --subject_name 'pubmed_graphsage_randomattack_remove' --path_y './data/pubmed/y_np.pkl' --path_result 'res/res_misclassification_models_pubmed.csv' --path_pre_result 'res/pre_res_misclassification_models_pubmed.csv'




