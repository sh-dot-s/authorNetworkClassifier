import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

import models
from data import author_graph, edge_list

NODE2VEC_DIMENSION = 100
EPOCHS = 100
BATCH_SIZE = 2 ** 10
SEED = random.randint(0, 100)
PAD_LENGTH = 300


def train():
    if not os.path.exists(os.path.join(os.curdir, "unconnected.npy")):
        node_list = list(author_graph.nodes.keys())
        adjacency = nx.to_numpy_array(author_graph, nodelist=range(0, 4085), dtype=np.int)

        all_unconnected_pairs = list()
        offset = 0
        for i in tqdm(range(adjacency.shape[0])):
            for j in range(offset, adjacency.shape[1]):
                try:
                    if i != j:
                        if adjacency[i, j] == 0:
                            if nx.shortest_path_length(author_graph, i, j) <= 3:
                                all_unconnected_pairs.append([node_list[i], node_list[j]])
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    all_unconnected_pairs.append([node_list[i], node_list[j]])
            offset = offset + 1
        all_unconnected_pairs = np.array(all_unconnected_pairs)
        np.save("unconnected", all_unconnected_pairs)
    else:
        all_unconnected_pairs = np.load("unconnected.npy")
    data = pd.DataFrame({'node_1': all_unconnected_pairs[:, 0],
                         'node_2': all_unconnected_pairs[:, 1]})
    data['link'] = 0
    data = data.sample(frac=2 * (len(edge_list) / len(data)), random_state=SEED)
    fb_df_partial = edge_list.copy()
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())
    fb_df_partial["link"] = 1
    # data = data.append(edge_list_ghost[['node_1', 'node_2', 'link']], ignore_index=True)
    data = data.append(fb_df_partial, ignore_index=True)
    if not os.path.exists(os.path.join(os.curdir, "author.embedding")):
        node2vec = Node2Vec(G_data, dimensions=NODE2VEC_DIMENSION, walk_length=50, num_walks=400)
        n2w_model = node2vec.fit(window=7, min_count=1, workers=12)
        n2w_model.save("author.embedding")
    else:
        n2w_model = Word2Vec.load("author.embedding")

    y = list()
    for i, j in zip(data['node_1'], data['node_2']):
        y.append(list(set(nx.neighbors(author_graph, i)).intersection(set(nx.neighbors(author_graph, j)))))

    largest_node = max(map(int, author_graph.nodes)) + 2
    weights = np.zeros((largest_node, NODE2VEC_DIMENSION))

    for node in author_graph.nodes:
        weights[node] = n2w_model.wv.get_vector(str(node))

    y = keras.preprocessing.sequence.pad_sequences(y, maxlen=PAD_LENGTH, padding='post', truncating='post',
                                                   value=largest_node - 1)
    xtrain, xtest, ytrain, ytest = train_test_split(y, data["link"],
                                                    test_size=0.2,
                                                    random_state=SEED)
    train_generator = models.DataGenerator(BATCH_SIZE, xtrain, keras.utils.to_categorical(ytrain))
    test_generator = models.DataGenerator(BATCH_SIZE, xtest, keras.utils.to_categorical(ytest))

    model = models.build_lstm_model(NODE2VEC_DIMENSION, weights)
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-3, patience=5)
    history = model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        callbacks=[early_stopping],
        epochs=100
    )
    model.save("lstm_on_embedding.h5")
    preds = model.predict_proba(xtest)
    print(roc_auc_score(ytest, preds.argmax(1)))


def test():
    predictor = keras.models.load_model("lstm_on_embedding.h5")
    actuals = pd.read_csv("sample.csv", index_col=0)
    data = pd.read_csv("test-public.csv", index_col=0)
    largest_node = max(map(int, author_graph.nodes)) + 2
    x = list()
    for i, j in zip(data['Source'], data['Sink']):
        try:
            x.append(
                list(set(nx.neighbors(author_graph, str(i))).intersection(set(nx.neighbors(author_graph, str(j))))))
        except (nx.NodeNotFound, nx.NetworkXError, nx.NetworkXNoPath):
            x.append([])
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=PAD_LENGTH, padding='post', truncating='post',
                                                   value=largest_node - 1)
    results = predictor.predict_proba(x)
    actuals["Predicted"] = results[:, 1]
    actuals.to_csv("trial_prediction.csv")
