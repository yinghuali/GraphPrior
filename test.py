import pickle
import pandas as pd

edge_index_np = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl', 'rb'))

print(edge_index_np.shape)
print(edge_index_np)


re = open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index.txt', 'a')
for i in range(len(edge_index_np[0])):
    re.write(str(edge_index_np[0][i]) + ' ' + str(edge_index_np[1][i]) + '\n')
re.close()