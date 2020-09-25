import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from gensim.models import Word2Vec
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tensorflow import keras
# import regress
# from matplotlib import pyplot as plt
NODE2VEC_DIMENSION = 64
EPOCHS = 100
BATCH_SIZE = 2 ** 10
SEED = random.randint(0, 100)
# from node2vec.edges import HadamardEmbedder
with open("nodes.json", "r") as anet_json, open("train.txt", "r") as relation_map:
    relation_mapping = dict()
    for relation in relation_map:
        aid, *nodes = map(int, relation.split())
        relation_mapping.update({aid: nodes})
    edge_list = [[], []]
    for n1, ln2 in relation_mapping.items():
        for p_ln2 in ln2:
            edge_list[0].append(n1)
            edge_list[1].append(p_ln2)
    edge_list = pd.DataFrame({"node_1": edge_list[0], "node_2": edge_list[1]})
    edge_list["node_1"] = pd.to_numeric(edge_list["node_1"])
    edge_list["node_2"] = pd.to_numeric(edge_list["node_2"])
    author_graph = nx.from_pandas_edgelist(edge_list, "node_1", "node_2", create_using=nx.Graph())


def get_random_walk(node, path_length):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(author_graph.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk


# upper_triangle_mat = np.triu(adjacency + 1, k=1)
# all_unconnected_pairs = np.argwhere(upper_triangle_mat == 1)
if not os.path.exists(os.path.join(os.curdir, "unconnected.npy")):
    node_list = [i for i in author_graph.nodes.keys()]
    adjacency = nx.to_numpy_matrix(author_graph, nodelist=node_list)
    all_unconnected_pairs = []
    offset = 0
    for i in tqdm(range(adjacency.shape[0])):
        for j in range(offset, adjacency.shape[1]):
            try:
                if i != j:
                    if adjacency[i, j] == 0:
                        if nx.shortest_path_length(author_graph, i, j) > 3:
                            all_unconnected_pairs.append([node_list[i], node_list[j]])
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                all_unconnected_pairs.append([node_list[i], node_list[j]])
        offset = offset + 1
    all_unconnected_pairs = np.array(all_unconnected_pairs)
    np.save("unconnected", all_unconnected_pairs)
else:
    all_unconnected_pairs = np.load("unconnected.npy")
if not os.path.exists(os.path.join(os.curdir, "omissible.csv")):
    initial_node_count = len(author_graph.nodes)
    edge_list_temp = edge_list.copy()
    omissible_links_index = []
    initial_connected_components = nx.number_connected_components(author_graph)
    for i in tqdm(edge_list.index.values):
        G_temp = nx.from_pandas_edgelist(edge_list_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())
        if (nx.number_connected_components(G_temp) == initial_connected_components) and (
                len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            edge_list_temp = edge_list_temp.drop(index=i)
    edge_list_ghost = edge_list.loc[omissible_links_index]
    edge_list_ghost['link'] = 1
else:
    edge_list_ghost = pd.read_csv("omissible.csv", index_col=0)

data = pd.DataFrame({'node_1': all_unconnected_pairs[:, 0],
                     'node_2': all_unconnected_pairs[:, 1]})
data['link'] = 0
# fb_df_partial = edge_list.drop(index=edge_list_ghost.index.values)
data = data.sample(frac=2 * (len(edge_list) / len(data)), random_state=SEED)
fb_df_partial = edge_list.copy()
G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())
fb_df_partial["link"] = 1
# data = data.append(edge_list_ghost[['node_1', 'node_2', 'link']], ignore_index=True)
data = data.append(fb_df_partial, ignore_index=True)
if not os.path.exists(os.path.join(os.curdir, "author.embedding")):
    node2vec = Node2Vec(G_data, dimensions=NODE2VEC_DIMENSION)
    n2w_model = node2vec.fit(window=7, min_count=1, workers=12)
    n2w_model.save("author.embedding")
else:
    n2w_model = Word2Vec.load("author.embedding")

edge_embeddings = HadamardEmbedder(keyed_vectors=n2w_model.wv)
if not os.path.exists(os.path.join(os.curdir, "mapping.npy")):
    x = list()
    for i, j in zip(data['node_1'], data['node_2']):
        try:
            x.append(edge_embeddings[(str(i), str(j))])
        except KeyError:
            x.append(np.zeros(NODE2VEC_DIMENSION))
    x = np.array(x)
    np.save("mapping", x)
else:
    x = np.load("mapping.npy")

# y = list()
# for i, j in zip(data['node_1'], data['node_2']):
#     y.append(list(set(nx.neighbors(author_graph, i)).intersection(set(nx.neighbors(author_graph, j)))))

# largest_node = max(map(int, author_graph.nodes))+2
# weights = np.zeros((largest_node, NODE2VEC_DIMENSION))

# for node in author_graph.nodes:
#     weights[node] = n2w_model.wv.get_vector(str(node))
#
# y = keras.preprocessing.sequence.pad_sequences(y, maxlen=100, padding='post', truncating='post',
#                                                value=largest_node-1)

xtrain, xtest, ytrain, ytest = train_test_split(x, data["link"],
                                                test_size=0.2,
                                                random_state=SEED)

training_size = len(ytrain)

# hyper_params = {
#     "learning_rate": [0.5, 0.6, 0.7],
#     "max_depth": [3, 4, 5],
#     "min_child_weight": [1, 3, 5, 7],
#     'subsample': [i / 10.0 for i in range(6, 10)],
#     'colsample_bytree': [i / 10.0 for i in range(6, 10)],
#     'gamma': [i / 10.0 for i in range(0,4)],
#     'reg_alpha': [0.1, 1, 20],
# }

# grid = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.68, max_depth=5, min_child_weight=3,
# subsample= 0.9, colsample_bytree=0.8, gamma=0,
# reg_alpha= 0.1), param_grid=hyper_params, scoring="roc_auc", n_jobs=-1, verbose=1, cv=3)
# grid.fit(xtrain, ytrain)
# print(grid.best_params_, grid.best_score_)
# print()
model = xgb.XGBClassifier(n_jobs=-1, verbosity=1)
model.fit(xtrain, ytrain)
# train_generator = regress.DataGenerator(BATCH_SIZE, xtrain, keras.utils.to_categorical(ytrain))
# test_generator = regress.DataGenerator(BATCH_SIZE, xtest, keras.utils.to_categorical(ytest))
#
# model = regress.build_model(NODE2VEC_DIMENSION, weights)
# early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-3, patience=5)
# history = model.fit_generator(
#     generator=train_generator,
#     validation_data=test_generator,
#     callbacks=[early_stopping],
#     epochs=100
# )
# model.save("lstm_on_embedding.h5")
pickle.dump(model, open("xgb3.pickle", "wb"))
preds = model.predict_proba(xtest)
print(roc_auc_score(ytest, preds.argmax(1)))
