import pandas as pd 

#%% Transformer mappings
TRANSFORMER_MAPPING = {
    'LabelBinarizer'        : sk.preprocessing.LabelBinarizer,
    'LabelEncoder'          : sk.preprocessing.LabelEncoder,
    'OneHotEncoder'         : sk.preprocessing.OneHotEncoder,
    'StandardScaler'        : sk.preprocessing.StandardScaler,
    }


#%% Get Excel transformer plan

xl = pd.ExcelFile(os.path.join(PATH_PROJECT_ROOT,'all columns.xlsx'))
trf_plans = dict()
for sheet in xl.sheet_names:  # see all sheet names
    trf_plans[sheet] = xl.parse(sheet)
    
    kept_col_count = int(trf_plans[sheet]['Keep'].dropna().sum())
    #.astype(int)
    logging.debug("{:>30}, processing {:>4} columns".format(sheet,kept_col_count))
    
# Make sure there is alignment betweent the loaded dataframes and the plans
for k in trf_plans:
    assert k in dfs.keys()

#%%
    
    

df_features = pd.read_excel(os.path.join(PATH_PROJECT_ROOT,'all columns.xlsx'))
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
df_out = data_mapper.fit_transform(df_app.copy())
df_out_head = df_out.head()


#%%

