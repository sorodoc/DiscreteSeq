import os
import csv
import json
import pandas as pd
import random
import numpy as np
from skimage import io, transform
import sys
import skimage
import torch


class SeqData():
    def __init__(self, path, vector_size, query_type):
        self.path = path
        self.query_type = query_type
        self.vector_size = vector_size
        self.training = self.read_data(path + 'training.tsv')
        self.valid = self.read_data(path + 'valid.tsv')
        self.test = self.read_data(path + 'test.tsv')
        self.inp_emb_size = len(self.test[0][0][0])
        self.query_emb_size = len(self.test[0][1])
        print(self.training[0])

    def read_data(self, path):
        fin = open(path, 'r')
        count = 0
        dps = []
        sequence = None
        query = ''
        answer = ''
        for line in fin:
            els = line.strip().split('\t')[1:]
            if count % 3 == 0 and count > 0:
                dps.append((sequence, query, answer))
            if count % 3 == 0:
                sequence = tuple(els)
            elif count % 3 == 1:
                els2 = els[-1].split()
                if self.query_type != '2features':
                    query = els2[-1]
                elif self.query_type == '2features':
                    query = (els2[-3], els2[-1])
            elif count % 3 == 2:
                if els[0] == 'yes':
                    answer = 1
                elif els[0] == 'no':
                    answer = 0
                else:
                    print('wrong')
            count += 1
        dps.append((sequence, query, answer))
        dps = self.create_feature_vect(dps)
        return dps


    def featurize_inp(self, inp):
        feature_vector = torch.zeros(len(inp))
        for i in range(len(inp)):
            if inp[i] == '1':
                feature_vector[i] = 1.0
        return feature_vector

    def featurize_query(self, q):
        q_vector = torch.zeros(self.vector_size)
        if self.query_type != '2features':
            q_vector[int(q)] = 1.0
        else:
            q_vector[int(q[0])] = 1.0
            q_vector[int(q[1])] = 1.0
        return q_vector

    def create_feature_vect(self, inp):
        dps = []
        for dp in inp:
            inp_fts = []
            for inp_ft in dp[0]:
                inp_fts.append(self.featurize_inp(inp_ft))
            dps.append((torch.stack(inp_fts), self.featurize_query(dp[1]), torch.Tensor([dp[2]])))
        return dps
