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

repeat_array=(1 2 3 4 5 6 7 8 9 10)
name="repeat_"
cmd="python mutation_feature_model_train.py --path_model_file './all_mutation_models/repeat/citeseer_gcn' --model_name 'gcn' --target_model_path './all_target_models/repeat/citeseer_gcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_gcn_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/all_res/repeat/res_misclassification_models_citeseer.csv' --path_pre_result 'res/all_res/repeat/pre_res_misclassification_models_citeseer.csv'"
for(( i=0;i<${#repeat_array[@]};i++)) do
var=${repeat_array[i]}
new_repeat=$name"$var"
new_cmd=${cmd//repeat/$new_repeat}
eval $new_cmd;
done;

repeat_array=(1 2 3 4 5 6 7 8 9 10)
name="repeat_"
cmd="python mutation_feature_model_train.py --path_model_file './all_mutation_models/repeat/citeseer_gat' --model_name 'gat' --target_model_path './all_target_models/repeat/citeseer_gat.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_gat_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/all_res/repeat/res_misclassification_models_citeseer.csv' --path_pre_result 'res/all_res/repeat/pre_res_misclassification_models_citeseer.csv'"
for(( i=0;i<${#repeat_array[@]};i++)) do
var=${repeat_array[i]}
new_repeat=$name"$var"
new_cmd=${cmd//repeat/$new_repeat}
eval $new_cmd;
done;


repeat_array=(1 2 3 4 5 6 7 8 9 10)
name="repeat_"
cmd="python mutation_feature_model_train.py --path_model_file './all_mutation_models/repeat/citeseer_tagcn' --model_name 'tagcn' --target_model_path './all_target_models/repeat/citeseer_tagcn.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_tagcn_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/all_res/repeat/res_misclassification_models_citeseer.csv' --path_pre_result 'res/all_res/repeat/pre_res_misclassification_models_citeseer.csv'"
for(( i=0;i<${#repeat_array[@]};i++)) do
var=${repeat_array[i]}
new_repeat=$name"$var"
new_cmd=${cmd//repeat/$new_repeat}
eval $new_cmd;
done;

repeat_array=(1 2 3 4 5 6 7 8 9 10)
name="repeat_"
cmd="python mutation_feature_model_train.py --path_model_file './all_mutation_models/repeat/citeseer_graphsage' --model_name 'graphsage' --target_model_path './all_target_models/repeat/citeseer_graphsage.pt' --path_x_np './data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --subject_name 'citeseer_graphsage_clean' --path_y './data/citeseer/y_np.pkl'  --path_result 'res/all_res/repeat/res_misclassification_models_citeseer.csv' --path_pre_result 'res/all_res/repeat/pre_res_misclassification_models_citeseer.csv'"
for(( i=0;i<${#repeat_array[@]};i++)) do
var=${repeat_array[i]}
new_repeat=$name"$var"
new_cmd=${cmd//repeat/$new_repeat}
eval $new_cmd;
done;
