#%% ===========================================================================
# Standard imports
# =============================================================================
from datetime import datetime

import pandas as pd

#%% ===========================================================================
#  Data source and paths
# =============================================================================
path_data = os.path.join(PATH_DATA_ROOT, r"")
assert os.path.exists(path_data), path_data
logging.info("Data path {}".format(PATH_DATA_ROOT))

# List the directory
files = os.listdir(PATH_DATA_ROOT)
file_list = [(f.split('.')[0], os.path.join(PATH_DATA_ROOT,f)) for f in files]
file_list = [f for f in file_list if os.path.isfile(f[1])]
file_dict = dict(file_list)
#file_dict = 

for file in file_dict:
    print(file)
    file_dict[file]

#%% Column descriptions
df_desc = pd.read_csv(file_dict['HomeCredit_columns_description'], 
                      encoding = "ISO-8859-1",
                      header=0, 
                      sep=',', 
                      quotechar='"',
                      index_col=0)

file_dict.pop('HomeCredit_columns_description')

#%% Load each df in the directory, convert to categorical

#aaaDataSubset = True
#DATA_SAMPLE = 0.3

DATA_SAMPLE = 1

def get_all_data(file_dict,sample_frac):
    dfs = dict()
    overall_start_time = time.time()
    for file in file_dict:
        start_time = time.time()
        logging.debug("Loading {}".format(file))
        #dfs[file] = pd.read_csv(file_dict[file], compression='zip', header=0, sep=',', quotechar='"')
        dfs[file] = pd.read_hdf(file_dict[file],  'data')
        logging.debug("\tLoaded {} with {} rows and {} cols".format(file, dfs[file].shape[0], dfs[file].shape[1]))        
        
        # Sample the data, if needed
        if not(sample_frac == 1):
            dfs[file] = dfs[file].sample(frac=sample_frac)
            logging.debug("\tSubsampled {} to {} rows".format(file, len(dfs[file])))
        
        dfs[file] = exergyml_util_other.convert_categorical(dfs[file])
        logging.debug("\tDone loading {}, {:.0f} seconds".format(file,time.time() - start_time))        

    
    logging.debug("\tLoaded {} dataframes over {:.1f} minutes".format(len(dfs),(time.time() - overall_start_time)/60))
    return dfs

dfs = get_all_data(file_dict,DATA_SAMPLE)

for k in dfs:
    print(k, dfs[k].shape)


#%% Split off the TARGET column

train_Y = dfs['application_train'].TARGET
dfs['application_train'] = dfs['application_train'].drop('TARGET', 1)
    
#%% Write the CSV dataframes to HDF5
if 0:
    for file in dfs:
        #dfs[file]
        file_path = file_dict[file]
        fname,_ = os.path.splitext(file_path)
        fname,_ = os.path.splitext(fname)
        fname += ".h5"
        logging.debug("Writing {}, {} rows, cols, to {}".format(file,dfs[file].shape,fname))
        dfs[file].to_hdf(fname, 'data', format='table')

#%% Create a summary of each dataframe and their columns
if 0:
    df_summaries = dict()
    for df_name in dfs:
        df_summaries[df_name] = exergyml_util_other.create_column_summary(dfs[df_name])
        #print(k)

#%% Print summaries to excel
if 0:
    PATH_PROJECT_ROOT
    excel_file_name = "all columns r00.xlsx"
    
    with pd.ExcelWriter(os.path.join(PATH_PROJECT_ROOT,excel_file_name), engine='xlsxwriter') as writer:
        for df_name in df_summaries:
            df_summaries[df_name].to_excel(writer, sheet_name=df_name)
        #writer.save()


#%% DONE HERE - DELETE UNUSED
logging.info("******************************")

del_vars =[
        "files",
        "file_dict",
        "column_dtypes",
        "column_names",
        "path_data",
        "util_path",
        "k"
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars



#%% Applications (Main data source!)
"""
application_train/application_test: the main training and testing data with 
information about each loan application at Home Credit. Every loan has its own 
row and is identified by the feature SK_ID_CURR. The training application data 
comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not 
repaid.
"""
#
#df_app = pd.read_csv(file_dict['application_train'], compression='zip', header=0, sep=',', quotechar='"')
#df_app = exergyml_util_other.convert_categorical(df_app)
#df_app_head = df_app.head()
#
#df_app_te = pd.read_csv(file_dict['application_test'], compression='zip', header=0, sep=',', quotechar='"')
#df_app_te = exergyml_util_other.convert_categorical(df_app_te)
#
#logging.info(f"Load")
#df_app["TARGET"]
#df_app_head["SK_ID_CURR"]
#column_names = list(df_app_head.columns)
#column_dtypes = [dt.name for dt in df_app_head.dtypes]
##dir(df_app["TARGET"].dtype)
##df_app["TARGET"].dtype.name
#
#df_app_te = pd.read_csv(file_dict['application_test'], compression='zip', header=0, sep=',', quotechar='"')
#


    
#%% Bureau
"""
bureau: data concerning client's previous credits from other financial 
institutions. Each previous credit has its own row in bureau, but one loan in 
the application data can have multiple previous credits.
"""

#df1_bureau = pd.read_csv(file_dict['bureau'], compression='zip', header=0, sep=',', quotechar='"')
#df1_bureau = exergyml_util_other.convert_categorical(df1_bureau)
#df1_head = df1_bureau.head()
#
#df2_bureau_balance = pd.read_csv(file_dict['bureau_balance'], compression='zip', header=0, sep=',', quotechar='"')
#df2_bureau_balance = exergyml_util_other.convert_categorical(df2_bureau_balance)
#df2_head = df2_bureau_balance.head()
#
#df3_credit_card_balance = pd.read_csv(file_dict['credit_card_balance'], compression='zip', header=0, sep=',', quotechar='"')
#df3_credit_card_balance = exergyml_util_other.convert_categorical(df3_credit_card_balance)
#df3_head = df3_credit_card_balance.head()
#
#df4_installments_payments = pd.read_csv(file_dict['installments_payments'], compression='zip', header=0, sep=',', quotechar='"')
#df4_installments_payments = exergyml_util_other.convert_categorical(df4_installments_payments)
#df4_head = df4_installments_payments.head()
#
#df5_POS_CASH_balance = pd.read_csv(file_dict['POS_CASH_balance'], compression='zip', header=0, sep=',', quotechar='"')
#df5_POS_CASH_balance = exergyml_util_other.convert_categorical(df5_POS_CASH_balance)
#df5_head = df5_POS_CASH_balance.head()
#
#df6_previous_application = pd.read_csv(file_dict['previous_application'], compression='zip', header=0, sep=',', quotechar='"')
#df6_previous_application = exergyml_util_other.convert_categorical(df6_previous_application)
#df6_head = df6_previous_application.head()


