import pandas as pd 


#%% ===========================================================================
# Bureau
# =============================================================================

#%% AGGREGATE numerical counts, meana, mins, max
# Group by SK_ID_CURR (the customer ID)
# The count of same ID is the count of entries in this previous loans table
# Previous loans come from OTHER INSTITUTIONS
# Get ALL aggregations

bureau_agg = exergyml_util_other.agg_numeric(dfs.bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')

#%% AGGREGATE categorical counts for each customer
bureau_counts = exergyml_util_other.count_categorical(dfs.bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
#bureau_counts.head()


#%% Old

#train_num_prev_loans=len(dfs.application_train.index.intersection(bureau_agg.index))
#test_num_prev_loans=len(dfs.application_test.index.intersection(bureau_agg.index))
#print("Percent clients with a previous loan entry (train):",train_num_prev_loans/len(dfs.application_train))
#print("Percent clients with a previous loan entry (test):",test_num_prev_loans/len(dfs.application_test))

# Fill the missing values with 0 
#dfs.application_train['previous_loan_counts'] = dfs.application_train['previous_loan_counts'].fillna(0)

#%%
if 0:
    kde_target('EXT_SOURCE_3', dfs.application_train)    
    kde_target('previous_loan_counts', dfs.application_train)    
    kde_target('bureau_DAYS_CREDIT_mean', dfs.application_train)

#%%
#exergyml_util_other.target_corrs(dfs.application_train)


#%% ===========================================================================
# Bureau balance
# =============================================================================
# NB: This table is joined not to application, but to bureau!

#%% AGGREGATE categories of loans
bureau_balance_counts = exergyml_util_other.count_categorical(dfs.bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

#%% AGGREATE counts of loans
bureau_balance_agg = exergyml_util_other.agg_numeric(dfs.bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

#%% AGGREGATE super aggregation to clients
# The above aggregates are per LOAN
# Now agg over each CLIENT

# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
bureau_by_loan = dfs.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

# Aggregate the stats for each client
bureau_balance_by_client = exergyml_util_other.agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')

#%%
def merge_dfs(df1, df2, on, how):
    logging.debug("Merging {}, {} to {} on {}".format(how,df1.shape,df2.shape,on))
    newdf = df1.merge(df2, on=on, how=how)
    #num_new_cols = len(newdf.columns)-len(df1)
    logging.debug("Merged into {}".format(newdf.shape))
    return newdf
    
    #print('Training Features new shape: {}, added {} of {} columns, removed unmatched {} rows'.format(m1.shape, m1.shape[1]-original_shape[1], len(bureau_agg.columns),original_shape[0]-len(m1)))


#%% ===========================================================================
#  JOIN ALL
# =============================================================================
dfs.application_train = merge_dfs(dfs.application_train,bureau_agg,'SK_ID_CURR','left')
dfs.application_train = merge_dfs(dfs.application_train,bureau_agg,'SK_ID_CURR','left')
dfs.application_train = merge_dfs(dfs.application_train,bureau_balance_by_client,'SK_ID_CURR','left')

dfs.application_test = merge_dfs(dfs.application_test,bureau_agg,'SK_ID_CURR','left')
dfs.application_test = merge_dfs(dfs.application_test,bureau_agg,'SK_ID_CURR','left')
dfs.application_test = merge_dfs(dfs.application_test,bureau_balance_by_client,'SK_ID_CURR','left')

#%% ALIGN DFs test/train 

print('Training Features shape: ', dfs['application_train'].shape)
print('Testing Features shape: ', dfs['application_test'].shape)

# Align the training and testing data, keep only columns present in both dataframes
dfs['application_train'], dfs['application_test'] = dfs['application_train'].align(dfs['application_test'], join = 'inner', axis = 1)

print('Training Features shape: ', dfs['application_train'].shape)
print('Testing Features shape: ', dfs['application_test'].shape)

#%% DONE HERE - DELETE UNUSED
logging.info("******************************")

del_vars =[
        "bureau_agg",
        "bureau_balance_agg",
        "bureau_balance_by_client",
        "bureau_balance_counts",
        "bureau_by_loan",
        "bureau_counts",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars

gc.collect()
