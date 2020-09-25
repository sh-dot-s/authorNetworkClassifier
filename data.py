import json
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

KEYWORD = "keyword"
VENUE = "venue"


class Author:

    def __init__(self, **kwargs):
        self.venue = np.zeros(348, dtype=np.int)
        self.keyword = np.zeros(53, dtype=np.int)
        self.first = None
        self.last = None
        self.num_papers = None
        self.id = None
        self.collaborators = list()
        self.__dict__.update(**self.parse_nums(kwargs))

    def parse_nums(self, kwargs: dict):
        n_kwargs = dict()
        for k, v in kwargs.items():
            if KEYWORD in k or VENUE in k:
                word, num = k.split("_")
                getattr(self, word)[int(num)] = 1
            else:
                n_kwargs.update({k: v})
        return n_kwargs


class AuthorDict(dict):
    def __init__(self, *args, **kwargs):
        self.callback = kwargs.pop('callback')
        super(AuthorDict, self).__init__(*args, **kwargs)

    def update(self, __m, **kwargs) -> None:
        self.callback(__m)
        super(AuthorDict, self).update(__m, **kwargs)


class AuthorsNetwork:
    nauthors: int = 0
    author_df = pd.DataFrame(columns=["id", "collaborators"])
    adjacency_matrix = pd.DataFrame()
    edge_list = None
    author_graph = None

    def __init__(self):
        self.authors = AuthorDict(callback=self.update_df)

    def update_df(self, *args):
        key, value = [(k, args[0][k]) for k in args[0]][0]
        update = {k: value.__dict__[k] for k in value.__dict__ if
                  k not in ["first", "last", "venue", "keyword", "num_papers"]}
        self.author_df.loc[key] = update
        row = np.zeros(self.nauthors + 1, dtype=np.bool)
        row[value.collaborators] = 1
        self.adjacency_matrix[key] = row

    def calc_similarity(self):
        for v in self.authors.values():
            if v.collaborators:
                setattr(v, "key_similarity",
                        [self.similarity(v.keyword, self.authors.get(c).keyword) for c in v.collaborators])
                setattr(v, "venue_similarity",
                        [self.similarity(v.venue, self.authors.get(c).venue) for c in v.collaborators])

    def to_dict_imp_only(self):
        adict = dict()
        for k, v in self.authors.items():
            adict.update({k: {k1: v1 for k1, v1 in v.__dict__
                         .items() if k1 in ["key_similarity", "venue_similarity", "collaborators", "id"]}})
        return adict

    def to_dict(self):
        return dict(authors={k: v.__dict__ for k, v in self.authors.items()})

    @staticmethod
    def similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if not os.path.exists(os.path.join(os.curdir, "data.pickle")):
    anet = AuthorsNetwork()
    with open("nodes.json", "r") as anet_json, open("train.txt", "r") as relation_map:
        relation_mapping = dict()
        for relation in relation_map:
            aid, *nodes = map(int, relation.split())
            relation_mapping.update({aid: nodes})
        anet.nauthors = aid
        edge_list = [[], []]
        for n1, ln2 in relation_mapping.items():
            for p_ln2 in ln2:
                edge_list[0].append(n1)
                edge_list[1].append(p_ln2)
        edge_list = pd.DataFrame({"node_1": edge_list[0], "node_2": edge_list[1]})
        edge_list["node_1"] = pd.to_numeric(edge_list["node_1"])
        edge_list["node_2"] = pd.to_numeric(edge_list["node_2"])
        for unparsed_a in json.load(anet_json):
            author = Author(**unparsed_a)
            author.collaborators = relation_mapping.get(author.id)
            anet.authors.update({author.id: author})
    author_graph = nx.from_pandas_edgelist(edge_list, "node_1", "node_2", create_using=nx.Graph())

    pickle.dump((anet, edge_list, author_graph), open("data.pickle", "wb"))
else:
    anet, edge_list, author_graph = pickle.load(open("data.pickle", "rb"))
