import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import yaml
import numpy as np
import pandas as pd
from dataloader import Rating_Dataset, ID_Mapper, Contact_Dataset, collate_fn
from models.MatrixFactor import MatrixFactorization
from models.graphrec import GraphRec

# 按用户分组计算NDCG
def compute_ndcg(group):
    true_ratings = group['true'].tolist()
    pred_ratings = group['pred'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 50)

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)


class Basic_Trainer:
    pass