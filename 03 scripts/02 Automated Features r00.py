# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

for k in dfs:
    print(k)


#%% Create DB schema (Entity Set)
# Entity set with id applications
def create_entities(dfs,df_trainortest):
    es = ft.EntitySet(id = 'clients')
    
    ###### Tables
    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id = 'app', dataframe=df_trainortest, index = 'SK_ID_CURR')
    es = es.entity_from_dataframe(entity_id = 'bureau', dataframe=dfs.bureau, index = 'SK_ID_BUREAU')
    es = es.entity_from_dataframe(entity_id = 'previous', dataframe=dfs.previous_application, index = 'SK_ID_PREV')
    
    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe=dfs.bureau_balance,make_index = True, index = 'bureaubalance_index')
    es = es.entity_from_dataframe(entity_id = 'cash', dataframe=dfs.POS_CASH_balance, make_index = True, index = 'cash_index')
    es = es.entity_from_dataframe(entity_id = 'installments', dataframe=dfs.installments_payments,make_index = True, index = 'installments_index')
    es = es.entity_from_dataframe(entity_id = 'credit', dataframe=dfs.credit_card_balance,make_index = True, index = 'credit_index')
    
    ###### Table Relationships

    # Relationship between app and bureau
    r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
    
    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])
    
    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])
    
    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
    r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
    
    # Add in the defined relationships
    es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                               r_previous_cash, r_previous_installments, r_previous_credit])
    
    return es

es_train = create_entities(dfs,dfs.application_train)
es_test  = create_entities(dfs,dfs.application_test)

# Print out the EntitySet
print(es_train)
print(es_test)

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
                                                     max_depth = 2, features_only = False, verbose = True)
    logging.debug('{} Total Features'.format(len(feature_names_spec)))
    return feature_matrix_spec, feature_names_spec


start = datetime.datetime.now()
train_X, train_names_spec = gen_features_subset(es_train)
logging.debug("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

start = datetime.datetime.now()
test_X, test_names_spec = gen_features_subset(es_test)
logging.debug("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Ensure alignment again
train_X, test_X = train_X.align(test_X, join = 'inner', axis = 1)








#%%






















