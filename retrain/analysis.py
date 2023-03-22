import pandas as pd
import os
import pickle
import numpy as np


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pkl'):
                    path_list.append(file_absolute_path)
    return path_list


def main():
    path_list = get_path('./res')
    print(len(path_list))
    data_list = [pickle.load(open(i, 'rb')) for i in path_list]
    approach_list = ['KMGP', 'DNGP', 'LGGP', 'LRGP', 'XGGP', 'RFGP', 'DeepGini', 'Entropy', 'LeastConfidence', 'Margin', 'VanillaSM', 'PCS', 'Random']
    cols = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'Average']
    res = []
    for approach in approach_list:
        tmp = []
        for dic in data_list:
            tmp.append(dic[approach])
        tmp = np.array(tmp)
        tmp = np.mean(tmp, axis=0)
        average = str(sum(tmp) / len(tmp))[:5]
        tmp = [str(i)[:5] for i in tmp]
        tmp.append(average)
        res.append(tmp)
    df = pd.DataFrame(res, columns=cols)
    df['Approach'] = approach_list
    df.to_excel('res/res.xlsx', index=False)
    print(df)


if __name__ == '__main__':
    main()



