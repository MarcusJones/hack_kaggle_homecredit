import pandas as pd 

#%% Transformer mappings
TRANSFORMER_MAPPING = {
    'LabelBinarizer'        : sk.preprocessing.LabelBinarizer,
    'LabelEncoder'          : sk.preprocessing.LabelEncoder,
    'OneHotEncoder'         : sk.preprocessing.OneHotEncoder,
    'StandardScaler'        : sk.preprocessing.StandardScaler,
    'CategoricalImputer'    : skpd.CategoricalImputer,
    'Imputer'               : sk.preprocessing.Imputer,
    'Imputer1D'             : exergyml_transformers.Imputer1D,
    }

dir(sk.preprocessing.LabelBinarizer)
sk.preprocessing.LabelBinarizer.__class__


sk.preprocessing.LabelBinarizer.__class__
#%% Get Excel transformer plans
def get_excel_transform_plans(xl_path):
    """
    Iterate over sheets, get all as dataframe
    """
    xl = pd.ExcelFile(xl_path)
    trf_plans = dict()
    for sheet in xl.sheet_names:  # see all sheet names
        trf_plans[sheet] = dict()
        trf_plans[sheet]['df'] = xl.parse(sheet)
        
        kept_col_count = int(trf_plans[sheet]['df']['Keep'].dropna().sum())
        #.astype(int)
        logging.debug("{:>30}, processing {:>4} columns".format(sheet,kept_col_count))
    return trf_plans

trf_plans = get_excel_transform_plans(os.path.join(PATH_PROJECT_ROOT,'all columns.xlsx'))

# Make sure there is alignment betweent the loaded dataframes and the plans
for k in trf_plans:
    assert k in dfs.keys()
#%% Process the plans into pipelines
def convert_plan_pipeline(trf_plan):
    """
    Given a dataframe listing columns and transformers, convert each row into a
    transformer pipeline step, and return the entire list of steps (pipeline)
    """
    logging.debug("Processing plan for {}".format(k))
    
    pipeline = list()

    # Get the df
    this_df = trf_plans[k]['df'] 
    # Get the transformer column indices
    trf_cols = [col for col in this_df if col.startswith('Transformer')]
    # Kept columns indices
    kept_cols = this_df['Keep'] == 1
    # The overall plan, as a DF
    transformation_plan = this_df.loc[kept_cols,['column name']+trf_cols]
    
    # Iterate over each column definition
    for i,transformers in transformation_plan.iterrows():
        
        # This is the template for the column step:
        pipeline_step = dict()
        pipeline_step["column name"] = None
        pipeline_step["transformations"] = list()        
        
        # 
        trfs_list = list(transformers.fillna(value=""))
        pipeline_step["column name"] = trfs_list.pop(0)
        
        if all(t == ''for t in trfs_list):
            pipeline_step["transformations"] = None
        else:
            trfs = [TRANSFORMER_MAPPING[this_t]() for this_t in trfs_list if this_t]
            pipeline_step["transformations"] = trfs
        
        logging.debug("{:30} {}".format(pipeline_step["column name"],pipeline_step["transformations"]))        
        pipeline.append(pipeline_step)
     
    return pipeline

for k in trf_plans:
    trf_plans[k]['pipeline_steps'] = convert_plan_pipeline(trf_plans[k]['df'])

#%%
def convert_pipesteps_datamapper(steps):
    """Given a list of steps, create a data mapper
    """
    if steps:
        step_list = [(step['column name'],step['transformations']) for step in steps]
        data_mapper = skpd.DataFrameMapper(step_list, df_out=True)
        return data_mapper
    else:
        return None

#steps = trf_plans[k]['pipeline_steps']
for k in trf_plans:
    trf_plans[k]['pipeline'] = convert_pipesteps_datamapper(trf_plans[k]['pipeline_steps'])

#%% Copy the plan from TRAIN to TEST
trf_plans['application_test'] = trf_plans['application_train']
logging.debug("Copied transformation plan from application_train to application_test".format())

#%% Overall plan
for k in trf_plans:
    print(k)
    if trf_plans[k]['pipeline']:
        this_pipeline = trf_plans[k]['pipeline']
        for step in this_pipeline.features:
            print(step)
#            #print("\t{:>25} -> ".format(step['column name']),end="")
#            if step['transformations']:
#                for trf in step['transformations']:
#                    #print(trf)
#                    print(trf.__class__.__name__, end="")
#            else:
#                print("(Keep)", end="")
#            print()

#%% Process the pipelines

dfs_transformed = dict()
dfs_transformed_heads = dict()
for k in dfs:
    #logging.debug("Processing {} from {}".format(k, dfs[k].shape))        
    #continue
    if trf_plans[k]['pipeline']:
        #dfs_transformed
        this_df = dfs[k]
        this_pipeline = trf_plans[k]['pipeline']
        dfs_transformed[k] = this_pipeline.fit_transform(this_df.copy())
        dfs_transformed_heads[k] = dfs_transformed[k].head()
        logging.debug("{}: {} cols transformed to {}, over {} rows".format(k, len(dfs[k].columns),len(dfs_transformed[k].columns), len(dfs_transformed[k])))
    else:
        logging.debug("No transformation pipeline found".format())

#%% Augment from other data


        
#%% Custom Imputation for 1D, testing here
        

#dfs['application_test']['AMT_ANNUITY']
#this_imputer = sk.preprocessing.Imputer()
##this_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
#this_imputer.fit([dfs['application_test']['AMT_ANNUITY']])
#
#my_imputer = Imputer1D()
##this_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
#my_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
#res = my_imputer.transform(dfs['application_test']['AMT_ANNUITY'])


#%%
if 0:
    df = trf_plans['application_train']
        
    
    #df_features = pd.read_excel(os.path.join(PATH_PROJECT_ROOT,'all columns.xlsx'))
    df_features["Transformer 1"].fillna("",inplace=True)
    
    pipeline = list()
    for i,row in df_features.iterrows():
        
        if row["Keep"] != 1: # Not in pipeline
            logging.debug("{:3}   {:30}".format(i,row["Column name"]))
            continue    
        
        # New transformer
        pipeline_step = dict()
        pipeline_step["column name"] = row["Column name"]
        pipeline_step["transformations"] = list()
        if row["Transformer 1"]:
            # Get the corresponding class, instantiate a new transformer 
            this_transfomation = TRANSFORMER_MAPPING[row["Transformer 1"]]()
            pipeline_step["transformations"].append(this_transfomation)
        else:
            pipeline_step["transformations"].append(None)
        pipeline.append(pipeline_step)
        logging.debug("{:3} + {:30} {}".format(i,row["Column name"],pipeline_step))
        
    pipe = [[row['column name'], row["transformations"]] for row in pipeline]
    plist1 = list()
    for r in pipe:
        if len(r[1]) == 1:
             r[1] = r[1][0]
        plist1.append(tuple(r))
    
    data_mapper = skpd.DataFrameMapper(plist1, df_out=True)
    
    #dir(data_mapper)
    for step in data_mapper.features:
        print("{:30} {}".format(step[0], step[1]))

#%%
#df_out = data_mapper.fit_transform(df_app.copy())
#df_out_head = df_out.head()


#%%

