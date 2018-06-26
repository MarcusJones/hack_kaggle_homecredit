#%% ===========================================================================
# Grid search
# =============================================================================
import lightgbm as lgb
###############################################################################
MODEL_STRING = "Random Forest"
###############################################################################

# Grid serach

clf = sklearn.svm.SVC(C=1.0, 
                      kernel='rbf', 
                      degree=3, 
                      gamma='auto', 
                      coef0=0.0, 
                      shrinking=True, 
                      probability=True, 
                      tol=0.001, 
                      cache_size=200, 
                      class_weight='balanced', 
                      verbose=False, 
                      max_iter=-1, 
                      decision_function_shape='ovr', 
                      random_state=None)



############ Parameter grid #################
param_grid = {
        'C' : [0.1,1,10,50,100],
        #'gamma' : [0.00001,0.0001,0.001,0.01,0.1],
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
