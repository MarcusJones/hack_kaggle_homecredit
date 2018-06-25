import pandas as pd 

#%% 
df = dfs['bureau']

df['CREDIT_ACTIVE'].value_counts()

df['CREDIT_DAY_OVERDUE'].value_counts()

#%% Count number of previous loans
# Group by SK_ID_CURR (the customer ID)
# The count of same ID is the count of entries in this previous loans table
# Previous loans come from OTHER INSTITUTIONS
previous_loan_counts = dfs['bureau'].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
#previous_loan_counts['previous_loan_counts'].hist()

#%% Join this table
# Join to the training dataframe

train_num_prev_loans=len(dfs.application_train.index.intersection(previous_loan_counts.index))
test_num_prev_loans=len(dfs.application_test.index.intersection(previous_loan_counts.index))
print("Percent clients with a previous loan (train):",train_num_prev_loans/len(dfs.application_train))
print("Percent clients with a previous loan (test):",test_num_prev_loans/len(dfs.application_test))
dfs.application_train = dfs.application_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

# Fill the missing values with 0 
dfs.application_train['previous_loan_counts'] = dfs.application_train['previous_loan_counts'].fillna(0)
#


#%%
if 0:
    kde_target('EXT_SOURCE_3', dfs.application_train)    
    kde_target('previous_loan_counts', dfs.application_train)    

#%% Aggregate all numeric

bureau_agg = dfs.bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'min', 'sum']).reset_index()
bureau_agg.head()


#%% Rename all columns
# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in bureau_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in bureau_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_%s_%s' % (var, stat))

# Assign the list of columns names as the dataframe column names
bureau_agg.columns = columns
bureau_agg.head()

# Merge this new data
# Merge with the training data
dfs.application_train = dfs.application_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

#%% Correlations in these new features
# List of new correlations
new_corrs = []

# Iterate through the columns 
for col in columns:
    # Calculate correlation with the target
    corr = dfs.application_train['TARGET'].corr(dfs.application_train[col])
    
    # Append the list as a tuple

    new_corrs.append((col, corr))

# Sort the correlations by the absolute value
# Make sure to reverse to put the largest values at the front of list
new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs[:15]

#%%
kde_target('bureau_DAYS_CREDIT_mean', dfs.application_train)

#%%
bureau_agg_new = exergyml_util_other.agg_numeric(dfs.bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg_new.head()

#%%
exergyml_util_other.target_corrs(dfs.application_train)


