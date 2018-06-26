#%% ===========================================================================
# Grid search
# =============================================================================
###############################################################################
MODEL_STRING = "Logistic Regression"
###############################################################################

# Grid serach
clf = sk.linear_model.LogisticRegression(penalty='l2', 
                                         dual=False, 
                                         tol=0.0001, 
                                         C=1.0, # Regularization strength (inverse)
                                         fit_intercept=True, 
                                         intercept_scaling=1, 
                                         class_weight=None, 
                                         random_state=None, 
                                         #solver='sag', 
                                         solver='liblinear', 
                                         max_iter=100, 
                                         multi_class='ovr', 
                                         verbose=0, 
                                         warm_start=False, 
                                         n_jobs=1)




############ Parameter grid #################
param_grid = {
    #'penalty': ['l2','l1'],
    'penalty': ['l2'],
    'class_weight': ['balanced'], #[None,'balanced'],
    #'fit_intercept': [True, False],
    'C':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0]
    #'C':np.linspace(0.05,1,20).tolist()
    #'C':[0.05,0.5]
    #'alpha': [10 ** x for x in range(-4, -2)],
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    #'max_iter': [1000],
    #'tol': [1e-3],
}
############################################

cv_folds=5
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='roc_auc',
                                           verbose=11)

###############################################################################