import networkx as nx
import pickle
import argparse
from GraphEmbedding.ge.models.node2vec import Node2Vec
from GraphEmbedding.ge.models.deepwalk import DeepWalk
from GraphEmbedding.ge.models.line import LINE
from GraphEmbedding.ge.models.sdne import SDNE
from GraphEmbedding.ge.models.struc2vec import Struc2Vec

# nohup python get_embedding_feature.py -p ../data/cora/edge_index.txt -s cora > 2.log 2>&1 &
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_edge", type=str, default='')
ap.add_argument("-s", "--save_name", type=str, default='')
args = vars(ap.parse_args())
path_edge = args['path_edge']
save_name = args['save_name']


def run_node2vec():
    save_embedding = save_name + '_' + 'node2vec.pkl'
    G = nx.read_edgelist(path_edge, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=16)
    model.train(window_size=5, iter=300, embed_size=128)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_line():
    save_embedding = save_name + '_' + 'line.pkl'
    G = nx.read_edgelist(path_edge, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=300, verbose=2)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_deepwalk():
    save_embedding = save_name + '_' + 'deepwalk.pkl'
    G = nx.read_edgelist(path_edge, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=16)
    model.train(window_size=5, iter=300, embed_size=128,)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_struc2vec():
    save_embedding = save_name + '_' + 'struc2vec.pkl'
    G = nx.read_edgelist(path_edge, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Struc2Vec(G, 10, 80, workers=16, verbose=40, )
    model.train(window_size=5, iter=300, embed_size=128)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


if __name__ == '__main__':
    run_node2vec()
    run_line()
    run_deepwalk()
    run_struc2vec()



