ls -lh

# ag
ag /home/yinghua/pycharm/MobileModelIdentif/sha256_play_google_com_list_0_50.pkl -i -l --silent -m2 /Users/yinghua.li/Documents/Server/MobileModelIdentif
ag hed_lite_model_quantize.tflite -i -l --silent -m2 /Users/yinghua.li/Documents/Server/MobileModelIdentif/ai_apps_analysis/3F71EFAF7D2A4B0645EBBF819F4AC4BD80C8B76ADCD03E57321CDE40AAA5B885_compile
ag tflite -i -l --silent -m2 /Users/yinghua.li/Documents/Server/MobileModelIdentif/ai_apps_analysis/D3BB1EE87D425BD5B9C318A7900DF79EB95D384749E5380843D495B5E240B5E3_compile

# 远程文件拉取
scp -r yinghua@serval09.uni.lu:/home/yinghua/pycharm/MobileModelIdentif/ai_apps_analysis/tfmodel_updates_analysis /Users/yinghua.li/Documents/Server/Models/TFmodel_updates
scp yinghua@serval09.uni.lu:/home/yinghua/pycharm/MobileModelIdentif/result_sha256_list_0_4000.txt /Users/yinghua.li/Documents/Server/MobileModelIdentif
scp yinghua@serval09.uni.lu:/home/yinghua/pycharm/MobileModelIdentif/result/google_sha256_list_25.txt /Users/yinghua.li/Documents/Server/MobileModelIdentif/result/
scp -r yinghua@trux01.uni.lu:/home/yinghua/pycharm/AIApps/ai_apps_analysis/df_latest_new.csv /Users/yinghua.li/Documents/Pycharm/AIApps/data/analysis_result

scp yinghua@trux01.uni.lu:0A0819A816111FE5F2F4AB19E9396C3D6AC158CAEA5A3AEA0ED3F54A2429C66E_compile /Users/yinghua.li/Documents/Server/MobileModelIdentif/result

scp yinghua@trux01.uni.lu:/home/yinghua/pycharm/MobileModelIdentif/ai_apps_analysis/unable_determine_framework.txt /Users/yinghua.li/Documents/Server/MobileModelIdentif/result
scp -r iris-cluster:/home/users/yili/pycharm/MobileModelIdentif/000001A94F46A0C3DDA514E1F24E675648835BBA5EF3C3AA72D9C378534FCAD6_test /Users/yinghua.li/Desktop

scp -r iris-cluster:/mnt/irisgpfs/users/yili/pycharm/data/cifar10_pkl /Users/yinghua.li/Documents/imagenet/tmp

scp iris-cluster:/mnt/irisgpfs/users/yili/pycharm/robustness_mobile_model/data/mnist/bim_x_train_adv.pkl /Users/yinghua.li/Documents/Pycharm/robustness_mobile_model/RQ2/mnist

# 本地上传
scp -r /Users/yinghua.li/Documents/imagenet/mobile_model_data/mnist_pkl/contrast_test.pkl iris-cluster:/mnt/irisgpfs/users/yili/pycharm/data/mnist_pkl
scp -r /Users/yinghua.li/Documents/imagenet/insect_150_150 iris-cluster:/work/projects/acc-estima-aiapp

scp -r /Users/yinghua.li/Documents/Pycharm/GNNEST yinghua@trux01.uni.lu:/home/yinghua/pycharm


# 后台运行shen
nohup /home/yinghua/anaconda3/bin/python -u analysis_lightweight_models_multiple.py > 2.log 2>&1 &
nohup /home/yinghua/anaconda3/bin/python -u down_decompilation_apk.py > /dev/null 2>&1 &
nohup /home/yinghua/anaconda3/bin/python -u test_unable_determine_framework.py > unable.log 2>&1 &
2211198

nohup /home/yinghua/anaconda3/bin/python main_multiprocessing.py > /dev/null 2>&1 &
nohup /home/yinghua/anaconda3/bin/python dogpedia_dogpedia.py > dogpedia_dogpedia.log 2>&1 &
18:24

/home/users/yili/anaconda3/bin/python -u main_multiprocessing.py


nohup python get_embedding_feature.py -p ../data/cora/edge_index.txt -s cora > 2.log 2>&1 &

nohup python get_pubmet_label_deepgini.py > get_pubmet_label_deepgini.log 2>&1 &
nohup python get_pubmet_label_margin.py > get_pubmet_label_margin.log 2>&1 &
nohup python get_pubmet_label_least.py > get_pubmet_label_least.log 2>&1 &
nohup python get_pubmet_label_varance.py > get_pubmet_label_varance.log 2>&1 &


nohup python get_est_pubmed_gat.py > get_est_pubmed_gat.log 2>&1 &

