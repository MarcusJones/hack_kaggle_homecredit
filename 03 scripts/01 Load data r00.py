#%% ===========================================================================
# Standard imports
# =============================================================================
import os
import yaml
import sys

from datetime import datetime

#%%
import logging
#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.DEBUG)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)-3s - %(module)-10s  %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(levelno)-3s - %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(funcName)-10s: %(message)s"
FORMAT = "%(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
#DATE_FMT = "%H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.debug("Logging started")

#%% ===========================================================================
#  Data source and paths
# =============================================================================
path_data = os.path.join(PATH_DATA_ROOT, r"")
assert os.path.exists(path_data), path_data
logging.info("Data path {}".format(PATH_DATA_ROOT))

files = os.listdir(PATH_DATA_ROOT)
file_dict = dict([(f.split('.')[0], os.path.join(PATH_DATA_ROOT,f)) for f in files])

#%% Column descriptions

df_desc = pd.read_csv(file_dict['HomeCredit_columns_description'], header=0, sep=',', quotechar='"')


#%% Applications
"""
application_train/application_test: the main training and testing data with 
information about each loan application at Home Credit. Every loan has its own 
row and is identified by the feature SK_ID_CURR. The training application data 
comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not 
repaid.
"""

df_app = pd.read_csv(file_dict['application_train'], compression='zip', header=0, sep=',', quotechar='"')
df_app_head = df_app.head()
logging.info(f"Load")
df_app["TARGET"]
df_app_head["SK_ID_CURR"]
column_names = list(df_app_head.columns)
column_dtypes = [dt.name for dt in df_app_head.dtypes]
#dir(df_app["TARGET"].dtype)
#df_app["TARGET"].dtype.name

df_app_te = pd.read_csv(file_dict['application_test'], compression='zip', header=0, sep=',', quotechar='"')

#%% Bureau
"""
bureau: data concerning client's previous credits from other financial 
institutions. Each previous credit has its own row in bureau, but one loan in 
the application data can have multiple previous credits.
"""

df_bureau = pd.read_csv(file_dict['bureau'], compression='zip', header=0, sep=',', quotechar='"')
df_bureau_head = df_bureau.head()






#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        "path_data",
        "sfpd_head",
        "sfpd_kag_all",
        "sfpd_kag_head",
        "df_summary",
        "util_path",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars





