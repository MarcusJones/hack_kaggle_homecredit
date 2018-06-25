#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from MachineLearning import util_other as exergyml_util_other
from MachineLearning import util_sk_transformers as exergyml_transformers
import sklearn_pandas as skpd 
import time
import os
import yaml
import sys

# =============================================================================
# Standard imports
# =============================================================================
import os
import logging
import datetime
import gc

# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
print_imports()

# Import again for code completion!
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
import sklearn
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.grid_search 
#import pickle
#sklearn.__version__
import sklearn.externals
# to make this notebook's output stable across runs
np.random.seed(42)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(action="ignore", module="pandas", message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=warnings)


#%% Logging
import logging
#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.DEBUG)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)-3s - %(module)-10s  %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(levelno)-3s - %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(funcName)-10s: %(message)s"
FORMAT = "%(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
#DATE_FMT = "%H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.debug("Logging started")


#%% Globals

LANDSCAPE_A3 = (16.53,11.69)
PORTRAIT_A3 = (11.69,16.53)
LANDSCAPE_A4 = (11.69,8.27)
TITLE_FONT = {'fontname':'helvetica'}


PATH_DATA_ROOT = r"/home/batman/Dropbox/DATA/Kaggle Housing Credit"
PATH_PROJECT_ROOT = r"/home/batman/git/hack_kaggle_homecredit/03 scripts"
#PATH_OUT = r"/home/batman/git/hack_sfpd1/Out"
#PATH_OUT_KDE = r"/home/batman/git/hack_sfpd1/out_kde"
#PATH_REPORTING = r"/home/batman/git/hack_sfpd1/Reporting"
#PATH_MODELS = r"/home/batman/git/hack_sfpd2/models"

#TITLE_FONT_NAME = "Arial"
#plt.rc('font', family='Helvetica')

#%%
exergyml_transformers.Imputer1D

#%%
exergyml_util_other.target_corrs

#%%
exergyml_util_other.agg_numeric

#%% Categorical counts
exergyml_util_other.count_categorical

#%% Missing vals
exergyml_util_other.missing_values_table

#%% Format a time delta
exergyml_util_other.strfdelta

#%% Column summaries
exergyml_util_other.create_column_summary

#%% Convert object to category
exergyml_util_other.convert_categorical

#%% Print and plot confusion matrix
exergyml_util_other.plot_confusion_matrix

#%% SKLEARN
exergyml_util_other.grid_scores_to_df

#%% Cmap mapper
exergyml_util_other.cmap_map

#%% Get countours
exergyml_util_other.get_contour_verts
