# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
from dask.distributed import LocalCluster

# featuretools for automated feature engineering
import featuretools as ft
ft.entityset.serialization.read_entityset()
#for k in dfs:
#    print(k)

import pickle

#%% Reload all data from pickle
logging.debug("Loading TRAIN entity set".format())
es_train = ft.entityset.serialization.read_entityset(os.path.join(PATH_OFFLINE,"entity set TRAIN.pck"))
logging.debug("Done loading TRAIN entity set".format())

logging.debug("Loading TEST entity set".format())
es_test = ft.entityset.serialization.read_entityset(os.path.join(PATH_OFFLINE,"entity set TEST.pck"))
logging.debug("Done loading TEST entity set".format())


#%% Create DB schema (Entity Set)
# Entity set with id applications
def create_entities(dfs,df_trainortest):
    es = ft.EntitySet(id = 'clients')
    
    ###### Tables
    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id = 'app', dataframe=df_trainortest, index = 'SK_ID_CURR')
    logging.debug("Built tables 1".format())
    es = es.entity_from_dataframe(entity_id = 'bureau', dataframe=dfs.bureau, index = 'SK_ID_BUREAU')
    logging.debug("Built tables 2".format())
    es = es.entity_from_dataframe(entity_id = 'previous', dataframe=dfs.previous_application, index = 'SK_ID_PREV')
    logging.debug("Built tables 3".format())
    
    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe=dfs.bureau_balance,make_index = True, index = 'bureaubalance_index')
    logging.debug("Built tables 4".format())
    es = es.entity_from_dataframe(entity_id = 'cash', dataframe=dfs.POS_CASH_balance, make_index = True, index = 'cash_index')
    logging.debug("Built tables 5".format())
    es = es.entity_from_dataframe(entity_id = 'installments', dataframe=dfs.installments_payments,make_index = True, index = 'installments_index')
    logging.debug("Built tables 6".format())
    es = es.entity_from_dataframe(entity_id = 'credit', dataframe=dfs.credit_card_balance,make_index = True, index = 'credit_index')
    logging.debug("Built tables".format())
    ###### Table Relationships

    # Relationship between app and bureau
    r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
    logging.debug("Built relation 1".format())

    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])
    logging.debug("Built relation 2".format())
    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])
    logging.debug("Built relation 3".format())
    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    logging.debug("Built relation 4".format())
    r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
    logging.debug("Built relation 5".format())
    r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
    logging.debug("Built relation 6".format())
    logging.debug("Built relationships".format())
    # Add in the defined relationships
    es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                               r_previous_cash, r_previous_installments, r_previous_credit])
    logging.debug("Added relations".format())
    return es

es_train = create_entities(dfs,dfs.application_train)
es_test  = create_entities(dfs,dfs.application_test)

# Print out the EntitySet
print(es_train)
print(es_test)
#%%
logging.debug("Writing TRAIN entity set".format())
es_train.to_pickle(os.path.join(PATH_OFFLINE,"entity set TRAIN.pck"))
logging.debug("Done writing TRAIN entity set".format())

logging.debug("Writing TEST entity set".format())
es_test.to_pickle(os.path.join(PATH_OFFLINE,"entity set TEST.pck"))
logging.debug("Done writing TEST entity set".format())


#%% 
n_workers = 12
n_workers = 6
cluster = LocalCluster(n_workers = n_workers,silence_logs=False)
dir(cluster)
print(cluster)
#%% Feature generation
# All features to depth 2
def gen_features_all():
    # Default primitives from featuretools
    default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
    default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]
    
    # DFS with specified primitives
    feature_names = ft.dfs(entityset = es, target_entity = 'app',
                           trans_primitives = default_trans_primitives,
                           agg_primitives=default_agg_primitives, 
                           max_depth = 2, features_only=True)
    
    logging.debug('{} Total Features'.format(len(feature_names)))    
    return feature_matrix_spec, feature_names_spec


# Subset of primatives to depth 2
def gen_features_subset(es):
    subset_primatives=['sum', 'count', 'min', 'max', 'mean', 'mode']
    
    # Specify the aggregation primitives
    feature_matrix_spec, feature_names_spec = ft.dfs(entityset = es, target_entity = 'app',  
                                                     agg_primitives = subset_primatives,
                                                     max_depth = 2, features_only = False, verbose = True,
                                                     dask_kwargs={'cluster': cluster})
    logging.debug('{} Total Features'.format(len(feature_names_spec)))
    return feature_matrix_spec, feature_names_spec


# Subset of primatives to depth 1
def gen_features_subset_d1(es,cluster):
    subset_primatives=['sum', 'count','mean']
    
    # Specify the aggregation primitives
    feature_matrix_spec, feature_names_spec = ft.dfs(entityset = es, target_entity = 'app',  
                                                     agg_primitives = subset_primatives,
                                                     max_depth = 1, features_only = False, verbose = True,
                                                     n_jobs = -1,
                                                     dask_kwargs={'cluster': cluster})
    logging.debug('{} Total Features'.format(len(feature_names_spec)))
    return feature_matrix_spec, feature_names_spec

# Subset of primatives to depth 1
def gen_features_subset_d1_no_cluster(es):
    subset_primatives=['sum', 'count','mean']
    
    # Specify the aggregation primitives
    feature_matrix_spec, feature_names_spec = ft.dfs(entityset = es, target_entity = 'app',  
                                                     agg_primitives = subset_primatives,
                                                     max_depth = 1, features_only = False, verbose = True,
                                                     n_jobs = -1,
                                                     )
    logging.debug('{} Total Features'.format(len(feature_names_spec)))
    return feature_matrix_spec, feature_names_spec


logging.debug("Generating features".format())
start = datetime.datetime.now()
train_X, train_names_spec = gen_features_subset_d1(es_train,cluster)
logging.debug("Done, elapsed H:m:s: {}".format(datetime.datetime.now()-start))

logging.debug("Generating features".format())
start = datetime.datetime.now()
test_X, test_names_spec   = gen_features_subset_d1(es_test,cluster)
logging.debug("Done, elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Ensure alignment again
train_X, test_X = train_X.align(test_X, join = 'inner', axis = 1)








#%%






















