import pandas as pd 

#%% Summary
df_summary = df_app.describe(include = 'all')
df_summary.loc["count",:].astype(int)
#df_app.info()

#app_train['TARGET'].value_counts()

#%% TARGET default yes/no 
df_app['TARGET'].value_counts()
# 0 = Paid on time      = 282686
# 1 = Default           =  24825

# Imbalance!

#%% Missing values
missing_values = exergyml_util_other.missing_values_table(df_app, sort = False)
missing_values.head(20)

#%% Column types 
df_app.dtypes.value_counts()
#float64    65
#int64      41
#object     16

#%% Unique classes in columns 
unique_classes = df_app.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

entries =dict()
for col in df_app.select_dtypes('object').columns:
    #print(col)
    #entries.append()
    entries[col] = df_app[col].unique().tolist()



#%% Transformer mapping
TRANSFORMER_MAPPING = {
    'LabelBinarizer'        : sk.preprocessing.LabelBinarizer,
    'LabelEncoder'          : sk.preprocessing.LabelEncoder,
    'OneHotEncoder'         : sk.preprocessing.OneHotEncoder,
    'StandardScaler'        : sk.preprocessing.StandardScaler,
    }

#df_app["NAME_TYPE_SUITE"].unique()
#sum(df_app["NAME_TYPE_SUITE"].isnull())
#%% Categorical data to ONE HOT

df_features = pd.read_excel(os.path.join(PATH_PROJECT_ROOT,'features r00.xlsx'))
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

