import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import re
import json
import numpy as np
import pandas as pd
import shutup
# shutup.please()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV


from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer, Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn import set_config
#
from sklego.preprocessing import ColumnSelector
#

#
from scipy.sparse import csr_matrix
#
import matplotlib.pyplot as plt
# import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
#
from sklearn_transformers import AutoTokenizer, AutoModel, AutoConfig
#
from tqdm import tqdm