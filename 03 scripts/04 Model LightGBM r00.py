#%% ===========================================================================
# Grid search
# =============================================================================
import lightgbm as lgb
###############################################################################
MODEL_STRING = "Random Forest"
###############################################################################

# Grid serach

clf = lgb.LGBMClassifier(n_estimators=10000, 
                         objective = 'binary', 
                         class_weight = 'balanced', 
                         learning_rate = 0.05, 
                         reg_alpha = 0.1, 
                         reg_lambda = 0.1, 
                         subsample = 0.8, 
                         n_jobs = -1, 
                         random_state = 50)


############ Parameter grid #################
param_grid = {
    'n_estimators':[10000],
    'learning_rate':[0.05],
    #'min_samples_split':[5,10],
    #'min_samples_leaf':[3,6],
}
############################################


cv_folds=5
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,
                                           cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='roc_auc',
                                           verbose=11)

###############################################################################
