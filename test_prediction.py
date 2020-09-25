import pickle

import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from node2vec.edges import HadamardEmbedder
import networkx as nx
from tqdm import tqdm
# from tensorflow import keras
from concurrent.futures import ThreadPoolExecutor

from feature_extraction import get_features, normalize, COLUMNS, SKIP
# -------------------------------------------------------------------------------------------------------------
# predictor = keras.models.load_model("lstm_on_embedding.h5")
# -------------------------------------------------------------------------------------------------------------
predictor = pickle.load(open("xgb_on_engineered_features.pickle", "rb"))
# predictor2 = pickle.load(open("rfclf.pickle", "rb"))
# -------------------------------------------------------------------------------------------------------------
actuals = pd.read_csv("sample.csv", index_col=0)
data = pd.read_csv("test-public.csv", index_col=0)
# -------------------------------------------------------------------------------------------------------------

# edge_dict = dict()
# with open("train.txt", "r") as train_file:
#     for line in train_file:
#         src, *dest = line.split()
#         edge_dict.update({src: dest})
#
# G = nx.from_dict_of_lists(edge_dict)
# n2w_model = Word2Vec.load("author.embedding")
# edge_embed = HadamardEmbedder(keyed_vectors=n2w_model.wv)
# largest_node = max(map(int, G.nodes))+2
# x = list()
# for i, j in zip(data['Source'], data['Sink']):
#     try:
#         x.append(list(set(nx.neighbors(G, str(i))).intersection(set(nx.neighbors(G, str(j))))))
#     except (nx.NodeNotFound, nx.NetworkXError, nx.NetworkXNoPath):
#         x.append([])
# x = keras.preprocessing.sequence.pad_sequences(x, maxlen=100, padding='post', truncating='post', value=largest_node-1)
# x = list()
# for i, j in zip(data['Source'], data['Sink']):
#     try:
#         x.append(edge_embed[(str(i), str(j))])
#     except KeyError:
#         x.append(np.zeros(64))
# x = np.array(x)
# -------------------------------------------------------------------------------------------------------------
COLUMNS.remove("label")

if not os.path.exists(os.path.join(os.curdir, "test_features.csv")):
    x = pd.DataFrame(columns=COLUMNS)
    executor = ThreadPoolExecutor()

    def parallel(i):
        features = get_features(*i)
        features.pop("label")
        return pd.DataFrame(features, index=[0])

    parallel_args = list()
    for i, row in tqdm(data.iterrows()):
        parallel_args.append((None, *row.values))
    results = tqdm(executor.map(parallel, parallel_args), total=len(parallel_args))
    x = pd.concat(results)
    x = x.reindex(sorted(x.columns), axis=1)
    x.to_csv("test_features.csv")

else:
    x = pd.read_csv("test_features.csv", index_col=0)

x.drop(columns=SKIP, inplace=True)
# x = normalize(x)
# -------------------------------------------------------------------------------------------------------------

results = predictor.predict_proba(x)
actuals["Predicted"] = results[:, 1]
actuals.to_csv("trial25.csv")
print()
