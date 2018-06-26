#%% ===========================================================================
# Grid search
# =============================================================================
###############################################################################
MODEL_STRING = "Random Forest"
###############################################################################

# Grid serach
clf = sk.ensemble.RandomForestClassifier(n_estimators=10, 
                                             criterion='gini', 
                                             max_depth=None, 
                                             min_samples_split=2, 
                                             min_samples_leaf=1, 
                                             min_weight_fraction_leaf=0.0, 
                                             max_features='auto', 
                                             max_leaf_nodes=None, 
                                             min_impurity_decrease=0.0, 
                                             min_impurity_split=None, 
                                             bootstrap=True, 
                                             oob_score=False, 
                                             n_jobs=1, 
                                             random_state=None, 
                                             verbose=0, 
                                             warm_start=False, 
                                             class_weight=None)

############ Parameter grid #################
param_grid = {
    'n_estimators':[1000,1500,3000,6000,10000],
    'max_depth':[15,20,30,60],
    'min_samples_split':[10,20,50,100],
    'min_samples_leaf':[16,32,64],
}
############################################


cv_folds=5
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='roc_auc',
                                           verbose=11)

###############################################################################
