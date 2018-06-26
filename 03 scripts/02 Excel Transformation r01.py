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
    'Default'               : "SPECIAL",
    'Keep Unchanged'        : "SPECIAL",
    }

DEFAULT_PIPES = {
        'int64'             : [exergyml_transformers.Imputer1D,sk.preprocessing.StandardScaler],
        'float64'           : [exergyml_transformers.Imputer1D,sk.preprocessing.StandardScaler],
        'category'          : [skpd.CategoricalImputer,sk.preprocessing.LabelBinarizer],
        }

#%% Load Excel transformer plans into DFs
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

trf_plans = get_excel_transform_plans(os.path.join(PATH_PROJECT_ROOT,'all columns r01.xlsx'))

# Make sure there is alignment betweent the loaded dataframes and the plans
for k in trf_plans:
    assert k in dfs.keys()
    
#for row in trf_plans['application_train']['df']:
#    print(row)
    
#%%
# loop over every column in the aligned train and test dfs

def generate_mapper():
    columnsequal = dfs.application_train.columns == dfs.application_train.columns
    assert columnsequal.all()
    
    col = 'NAME_CONTRACT_TYPE'
    col = 'CNT_CHILDREN'
    col = 'FLAG_DOCUMENT_2'
    
    this_plan = list()
    for col in dfs.application_train.columns:
        flg_kept = False
        # Is this column in the original data? 
        flg_original_col = col in trf_plans['application_train']['df']['column name'].values
        
        # Is this column kept?
        if flg_original_col:
            # There must be an easier way here;
            this_col_def = trf_plans['application_train']['df'].loc[trf_plans['application_train']['df']['column name'] == col]
            flg_kept = this_col_def['Keep'].fillna(0).astype(bool).values[0]
    
        # If original and kept, are there transformers applied?
        if flg_original_col and flg_kept:
            # Select only Transformer columsn
            trf_cols = [col for col in this_col_def if col.startswith('Transformer')]
            # Get the transformers as a list
            trf_list = this_col_def.loc[:,trf_cols].fillna(value="").values.tolist()[0]
            trfs_list = [t for t in trf_list if t]
            assert len(trfs_list) >= 1
            if len(trfs_list)==1 and trfs_list[0] == 'Keep Unchanged':
                trfs_list = [None]
            elif len(trfs_list)==1 and trfs_list[0] == 'Default':
                trfs_list = ['Default']
            else:
                trfs_list = [TRANSFORMER_MAPPING[this_t]() for this_t in trfs_list]
        else:
            trfs_list = [None]
    
        # Keep all new columns by default
        if not flg_original_col:
            flg_kept = True
    
        if 'Default' in trfs_list or not flg_original_col:
            # Get the steps
            this_dtype = str(dfs.application_train[col].dtype)
            this_pipe = DEFAULT_PIPES[this_dtype]
            # Instantiate the steps
            trfs_list = [trf() for trf in this_pipe]
        
        #if trfs_list
        
        trfs_list_strings = [t.__class__.__name__ for t in trfs_list]
    
        if None == trfs_list[0]:
            trfs_list = None
        
        print("{:50}  keep: {:1}, dtype: {:10}, original: {:1}, transformers: {}".format(col,
              flg_kept,
              str(dfs.application_train[col].dtype),
              flg_original_col,
              trfs_list_strings,
              )
        
        )
        
        
        if flg_kept:
            this_plan.append([col,trfs_list])
                
    return this_plan

super_plan = generate_mapper()
super_mapper = skpd.DataFrameMapper(super_plan, df_out=True)

#%% Copy the plan from TRAIN to TEST
#trf_plans['application_test'] = trf_plans['application_train']
#logging.debug("Copied transformation plan from application_train to application_test".format())

#%% Overall plan
def print_overall(data_mapper):
    for i,step in enumerate(data_mapper.features):
        col=step[0]
        trfs = step[1]
        if trfs:
            trf_string = " > ".join([t.__class__.__name__ for t in trfs])
        else:
            trf_string = "KEEP"
        logging.debug("{:4} {:50} {}".format(i,col,trf_string))

print_overall(super_mapper)

#%% Apply mapper
train_X = super_mapper.fit_transform(dfs.application_train)
test_X  = super_mapper.fit_transform(dfs.application_test)

#%% Ensure alignment again
train_X, test_X = train_X.align(test_X, join = 'inner', axis = 1)

#%% DONE HERE - DELETE UNUSED
logging.info("******************************")

del_vars =[
        "dfs",
        "k",
        'trf_plans',
        'file_list',
        'super_plan'
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars
gc.collect()

#%% GRAVEYARD

#%% Process the pipelines
if 0:
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


#%% OLD
if 0:
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


#%% OLD Process the plans into pipelines
if 0:
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
            trfs_list = [t for t in trfs_list if t]
            pipeline_step["column name"] = trfs_list.pop(0)
            
            if all(t == ''for t in trfs_list):
                pipeline_step["transformations"] = None
            elif any(t == 'Default' for t in trfs_list):
                #print(pipeline_step["column name"])
                #print(trfs_list)
                assert len(trfs_list) == 1
                assert trfs_list.pop() == 'Default'
                #print()
                #TRANSFORMER_MAPPING[this_t]
            else:
                trfs = [TRANSFORMER_MAPPING[this_t]() for this_t in trfs_list if this_t]
                pipeline_step["transformations"] = trfs
            
            logging.debug("{:30} {}".format(pipeline_step["column name"],pipeline_step["transformations"]))        
            pipeline.append(pipeline_step)
         
        return pipeline
    
    # Loop over each DF, and apply the plan conversion
    #for k in trf_plans:
    #    trf_plans[k]['pipeline_steps'] = convert_plan_pipeline(trf_plans[k]['df'])
    
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#          TESTING
#############################################################
if 0:
    #dfs['application_train']['NAME_TYPE_SUITE'].unique()
    dfs['application_train']['NAME_TYPE_SUITE'].value_counts()
    sum(dfs['application_train']['NAME_TYPE_SUITE'].isnull())
    data_mapper = skpd.DataFrameMapper([
        ("NAME_TYPE_SUITE", [skpd.CategoricalImputer(),sk.preprocessing.LabelBinarizer()]),
    ], df_out=True)
        
    df_trf = data_mapper.fit_transform(dfs.to_frame().copy())
    df_trf_head = df_trf.head()

if 0:
    class Imputer1Dtest(sk.preprocessing.Imputer):
        """
        A simple subclass on Imputer to avoid having to make a single column 2D. 
        """
        def fit(self, X, y=None):
            X2 = np.expand_dims(X, axis=1)
            # Call the Imputer as normal
            return super(Imputer1Dtest, self).fit(X2, y=None) 
            
        def transform(self, X, y=None):
            X2 = np.expand_dims(X, axis=1)
            # Return the result
            return super(Imputer1Dtest, self).transform(X2) 

#        def fit_transform(self, X, y=None,**fit_params):
#            #X2 = np.expand_dims(X, axis=1)
#            # Return the result
#            return self.fit(X, **fit_params).transform(X)
#            #return super(Imputer1Dtest, self).fit_transform(X) 

    this_transformer_test = sk.preprocessing.StandardScaler()
    this_transformer_test = Imputer1Dtest()
    this_transformer_test = exergyml_transformers.Imputer1D()
    r1 = this_transformer_test.fit(dfs['application_train']['CNT_CHILDREN'])
    
    r2 = this_transformer_test.fit_transform(dfs['application_train']['CNT_CHILDREN'])

#%% OLD DELETE!
if 0:
    #
    
    def map_seperate(tr_steps):
        data_mapper1 = skpd.DataFrameMapper([
            ('CNT_CHILDREN', 	    tr_steps),
            ('AMT_INCOME_TOTAL', 	tr_steps),
            ('AMT_CREDIT', 	        tr_steps),
            ('AMT_ANNUITY', 	    tr_steps),
            ('AMT_GOODS_PRICE', 	tr_steps),
        ], df_out=True)
            
        df_trf2 = data_mapper1.fit_transform(dfs['application_train'])
        df_trf2_head = df_trf2.head()
        return df_trf2
        
    
    def map_together(cols,tr_steps):
      
        data_mapperALL = skpd.DataFrameMapper([
            (cols, 	    tr_steps),
        ], df_out=True)
            
        df_trfALL = data_mapperALL.fit_transform(dfs['application_train'])
        df_trfALL_head = df_trfALL.head()        
        return df_trfALL
    
    
    my_cols = ['CNT_CHILDREN',
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',]
    #my_cols = ['CNT_CHILDREN']  
    
    #my_tr_steps1D=[Imputer1Dtest()]
    #my_tr_steps2D=[sk.preprocessing.Imputer(),sk.preprocessing.StandardScaler()]
    my_tr_steps2D=[exergyml_transformers.Imputer1D(),sk.preprocessing.StandardScaler()]
    tr_steps = my_tr_steps2D
    cols=my_cols
    #my_tr_steps2D=[Imputer1D(),sk.preprocessing.StandardScaler()]
    res_seperate = map_seperate(my_tr_steps2D)
    res_together = map_together(my_cols,my_tr_steps2D)


#%% OLD DELETE!
# Basic test
if 0: 
    data_3 = pd.DataFrame({'age': [1, np.nan, 3], 'biscuit' : [10000, 1002342, 2345256]})
    
    mapper3 = skpd.DataFrameMapper([
        (['age'], [sk.preprocessing.Imputer(),sk.preprocessing.StandardScaler()]),
        (['biscuit'], [sk.preprocessing.Imputer(),sk.preprocessing.StandardScaler()]),
        ])
    
    mapper3.fit_transform(data_3)

#%% OLD DELETE!
# Custom Imputation for 1D, testing here
        
if 0:
    dfs['application_test']['AMT_ANNUITY']
    this_imputer = sk.preprocessing.Imputer()
    #this_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
    this_imputer.fit([dfs['application_test']['AMT_ANNUITY']])
    
    my_imputer = Imputer1D()
    #this_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
    my_imputer.fit(dfs['application_test']['AMT_ANNUITY'])
    res = my_imputer.transform(dfs['application_test']['AMT_ANNUITY'])
    

#%% OLD DELETE!
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

