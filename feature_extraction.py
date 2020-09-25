import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from data import author_graph

COLUMNS = sorted(["label", "common_neighbours", "adjusted_rand",
                  "avg_neighbourhood_degree", "total_neighbours", "path_length", "adamic_index",
                  "resource_allocation", "preferential_attachment", "jaccard", "neighbourhood_distance",
                  "same_community",
                  "u_neighbours", "v_neighbours", "cosine_similarity"])

try:
    resource_allocation = pd.read_csv("resource_allocation.csv", index_col=0)
    adar_adamic_index = pd.read_csv("adamic_adar.csv", index_col=0)
except FileNotFoundError:
    resource_allocation = nx.resource_allocation_index(author_graph)
    resource_allocation = pd.DataFrame(list(resource_allocation))
    resource_allocation.to_csv("resource_allocation.csv")

    adar_adamic_index = nx.adamic_adar_index(author_graph)
    adar_adamic_index = pd.DataFrame(list(adar_adamic_index))
    adar_adamic_index.to_csv("adamic_adar.csv")

average_neighbourhood_degree = nx.average_neighbor_degree(author_graph)
communities = list(nx.community.asyn_lpa_communities(author_graph))

adjacency = nx.to_numpy_array(author_graph, nodelist=range(0, 4085), dtype=np.int)


def _are_of_same_community(nodes):
    nodes = set(nodes)
    for community in communities:
        if community.issuperset(nodes):
            return 1
    return 0


def _get_feature_from_frame(frame, u, v):
    result = frame[(frame["0"] == u) & (frame["1"] == v)].values
    return result[0][-1] if len(result) else 0.0


def normalize(frame):
    scaler = MinMaxScaler()
    scaler.fit(frame)
    pickle.dump(scaler, open("scaler.pickle", "wb"))
    return scaler.transform(frame)


def shortest_path_length(a, b):
    p = -1
    try:
        if author_graph.has_edge(a, b):
            author_graph.remove_edge(a, b)
            p = nx.shortest_path_length(author_graph, a, b)
            author_graph.add_edge(a, b)
        else:
            p = nx.shortest_path_length(author_graph, a, b)
        return p
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return p


def get_features(label, node_1, node_2):
    ai, aj = adjacency[node_1], adjacency[node_2]
    a = ai.dot(aj.T)
    b = ai.dot(1 - aj.T)
    c = (1 - ai).dot(aj.T)
    d = (1 - ai).dot(1 - aj.T)
    try:
        node_1_collabs, node_2_collabs = set(nx.neighbors(author_graph, node_1)), set(
            nx.neighbors(author_graph, node_2))
    except (nx.NodeNotFound, nx.NetworkXNoPath, nx.NetworkXError):
        node_1_collabs = node_2_collabs = set()
    adjusted_rand = (2 * (a * d - b * c)) / (((a + b) * (b + d)) + ((a + c) * (c + d)))
    try:
        neighbourhood_distance = int(a) / math.sqrt(len(node_1_collabs) * len(node_2_collabs))
    except ZeroDivisionError:
        neighbourhood_distance = 0

    total_neighbours = len(node_1_collabs.union(node_2_collabs))
    avg_neighbourhood_degree = np.mean(
        [average_neighbourhood_degree.get(node_1, 0), average_neighbourhood_degree.get(node_2, 0)])
    preferential_attachment = (a + c) * (a + b)
    jaccard_coefficient = a / (a + b + c)
    cosine_similarity = a / preferential_attachment if preferential_attachment else 0
    path_length = shortest_path_length(node_1, node_2)

    features = {
        "label": label,
        "common_neighbours": a,
        "total_neighbours": total_neighbours,
        "u_neighbours": len(node_1_collabs),
        "v_neighbours": len(node_2_collabs),
        "path_length": path_length,
        "adamic_index": _get_feature_from_frame(adar_adamic_index, node_1, node_2),
        "resource_allocation": _get_feature_from_frame(resource_allocation, node_1, node_2),
        "preferential_attachment": preferential_attachment,
        "jaccard": jaccard_coefficient,
        "same_community": _are_of_same_community([node_1, node_2]),
        "neighbourhood_distance": neighbourhood_distance,
        "avg_neighbourhood_degree": avg_neighbourhood_degree,
        "adjusted_rand": adjusted_rand if adjusted_rand != np.nan else 0,
        "cosine_similarity": cosine_similarity
    }
    return features


def run():
    if os.path.exists(os.path.join(os.curdir, "features.csv")):
        feature_df = pd.read_csv("features.csv", index_col=0)
    else:
        from data import edge_list

        all_unconnected_pairs = np.load("unconnected.npy")
        unconnected_df = pd.DataFrame({'node_1': all_unconnected_pairs[:, 0],
                                       'node_2': all_unconnected_pairs[:, 1]})
        unconnected_df = unconnected_df.sample(frac=1.2 * (len(edge_list) / len(unconnected_df)))
        executor = ThreadPoolExecutor()

        parallel_args = list()
        for i in tqdm(edge_list.index.values):
            parallel_args.append((1, *edge_list.loc[i].values))
        for i in tqdm(unconnected_df.index.values):
            parallel_args.append((0, *unconnected_df.loc[i].values))

        results = tqdm(
            executor.map(lambda x: pd.DataFrame(get_features(*x), index=[0], columns=COLUMNS), parallel_args),
            total=len(parallel_args))
        feature_df = pd.concat(results)
        feature_df.reset_index(drop=True, inplace=True)
        feature_df["neighbourhood_distance"][feature_df["neighbourhood_distance"] == np.inf] = 0
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.dropna(0, inplace=True)
        feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
        feature_df.to_csv("features.csv")

    labels = pd.to_numeric(feature_df.pop("label"))
    feature_cols = feature_df.columns
    feature_df = normalize(feature_df)
    train_x, test_x, train_y, test_y = train_test_split(feature_df, labels, test_size=0.2)
    clf = xgb.XGBClassifier()
    hyper_params = {
        "learning_rate": [0.2, 0.23, 0.25, 0.27, 0.3],
        "max_depth": [1, 2, 4, 6, 9, 12],
        "min_child_weight": [1, 3],
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        'gamma': [i / 10.0 for i in range(0, 4)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100, 500, 1000]
    }

    grid = GridSearchCV(estimator=clf,
                        param_grid=hyper_params,
                        scoring="roc_auc",
                        n_jobs=-1, verbose=1,
                        cv=3)
    grid.fit(train_x, train_y)
    print(grid.best_params_, grid.best_score_)
    clf.fit(train_x, train_y)
    print(list(zip(feature_cols, clf.feature_importances_)))
    pickle.dump(clf, open("xgb_on_engineered_features.pickle", "wb"))
    preds = clf.predict_proba(test_x)
    print(roc_auc_score(test_y, preds.argmax(1)))


def test():
    COLUMNS.remove("label")
    actuals = pd.read_csv("./data/sample.csv", index_col=0)
    data = pd.read_csv("./data/test-public.csv", index_col=0)
    predictor = pickle.load(open("xgb_on_engineered_features.pickle", "rb"))

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
    results = predictor.predict_proba(x)
    actuals["Predicted"] = results[:, 1]
    actuals.to_csv("trial_prediction.csv")


if __name__ == '__main__':
    run()
