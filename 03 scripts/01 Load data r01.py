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
    #file_dict[file]


#%% Column descriptions, load then remove from data file list
if 0:
    df_desc = pd.read_csv(file_dict['HomeCredit_columns_description'], 
                          encoding = "ISO-8859-1",
                          header=0, 
                          sep=',', 
                          quotechar='"',
                          index_col=0)
    with pd.ExcelWriter(os.path.join(PATH_PROJECT_ROOT,"Descriptions.xlsx"), engine='xlsxwriter') as writer:
            df_desc.to_excel(writer)
            #writer.save()
    
file_dict.pop('HomeCredit_columns_description')

#%% Attribute access by dot operator - convenience!
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

#%% Load each df in the directory, convert to categorical

#aaaDataSubset = True
#DATA_SAMPLE = 0.3

DATA_SAMPLE = 1
def get_all_data(file_dict,sample_frac):
    dfs = AttrDict()
    overall_start_time = time.time()
    for file in file_dict:
        start_time = time.time()
        logging.debug("Loading {}".format(file))
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
    print("dfs['{}'] {}".format(k, dfs[k].shape))


#%% Split off the TARGET column
train_Y = dfs['application_train'].TARGET
dfs['application_train'] = dfs['application_train'].drop('TARGET', 1)
    
#%% DONE HERE - DELETE UNUSED
logging.info("******************************")

del_vars =[
        "file",
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
gc.collect()