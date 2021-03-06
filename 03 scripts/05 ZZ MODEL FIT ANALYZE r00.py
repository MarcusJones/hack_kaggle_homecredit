#%% Smaller subset for gridsearch
SAMPLE_FRAC = 0.20
SAMPLE_SIZE = int(len(train_X)*SAMPLE_FRAC)
#train_X.sample(SAMPLE_SIZE)
#sample_rows = pd.Series(train_X.index).sample(SAMPLE_SIZE).astype(str).values
sample_rows = pd.Series(train_X.index).sample(SAMPLE_SIZE)
logging.debug("Subset for Grid Search, {} rows".format(len(sample_rows)))

#%% Final sizing

print("train_X",train_X.shape,"Reduced for grid:", train_X.loc[sample_rows,:].shape)
print("train_Y",train_Y.shape,"Reduced for grid:->", train_Y.loc[sample_rows].shape)
print("test_X",test_X.shape)

#%% Fit
start = datetime.datetime.now()
clf_grid.fit(train_X.loc[sample_rows,:], train_Y.loc[sample_rows])
grid_search_elapsed = datetime.datetime.now()-start
grid_search_elapsed = exergyml_util_other.strfdelta(grid_search_elapsed, "{hours:02d}:{minutes:02d}:{seconds:02d}")
#grid_search_elapsed.strptime
logging.debug("Elapsed H:m:s: {}".format(grid_search_elapsed))
print("best score:", clf_grid.best_score_)
print("Bast parameters:", clf_grid.best_params_)

#%% Save the grid search
if 0:
    path_grid_search_out = os.path.join(PATH_MODELS,MODEL_STRING+" grid search fit"+".pkl")
    sk.externals.joblib.dump(clf_grid, path_grid_search_out)

#clf_grid_old = clf_grid
#clf_grid = sk.externals.joblib.load(path_confusionmatrix_out)

#%% Analysis of grid search
# Get the paramater space
df_grid = exergyml_util_other.grid_scores_to_df(clf_grid.grid_scores_)
df_grid.fillna('None',inplace=True)

# Add the string labels (hack)
grid_labels = dict(df_grid.dtypes)
string_labels = list()
for k in grid_labels: 
    if grid_labels[k] == 'object': string_labels.append(k)
    
# Aggregate over each parameter set
df_grid_agg = df_grid.groupby(['param_set']+string_labels).agg(['mean','var']).reset_index()
df_grid_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_agg.index = df_grid_agg['param_set']
df_grid_agg.drop('param_set',1,inplace=True)

# The BAST model is: 
clf_grid_BEST = clf_grid.best_estimator_

# Write to Excel
if 0:
    path_grid_params = os.path.join(PATH_MODELS,MODEL_STRING+" grid space"+".xlsx")
    with pd.ExcelWriter(path_grid_params) as writer:
        df_grid.to_excel(writer,'Grid Space over Folds')
        df_grid_agg.to_excel(writer,'Grid Space aggregate')
        writer.save()

# Save the best
if 0:
    path_best_out = os.path.join(PATH_MODELS,MODEL_STRING+" best fit"+".pkl")
    sk.externals.joblib.dump(clf_grid_BEST, path_best_out)


#%% Do the final fit on the BEST estimator
start = datetime.datetime.now()
predicted = clf_grid_BEST.fit(train_X, train_Y)
logging.debug("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))


#%% Predict on Test set
# NB we only want the defaulters column! 
start = datetime.datetime.now()
predicted = clf_grid_BEST.predict_proba(test_X)[:,1]
logging.debug("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Submission dataframe
submit = test_X[['SK_ID_CURR']].astype(int)
submit['TARGET'] = predicted

submit.head()

# Save the submission to a csv file
submit.to_csv('log_reg_baseline10.csv', index = False)

#%% Results on TEST

#predicted_cat_num = predicted.argmax(axis=1)
#predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))
test_logloss = sk.metrics.log_loss(test_Y, predicted)
print(test_logloss)
test_roc_auc = sk.metrics.roc_auc_score(test_Y, predicted)

#predicted_cat_num = predicted.argmax(axis=1)
#predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))

test_accuracy = sk.metrics.accuracy_score(test_Y, predicted_cat_num)
print(test_accuracy)
#print(sk.metrics.log_loss(res, predicted))

confusion_matrix = pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, predicted_cat_num))
confusion_matrix.columns = selected_crimes
confusion_matrix.index = selected_crimes


#%% Confusion
plot_confusion_matrix(confusion_matrix,
                          selected_crimes,
                          normalize=False,
                          title=MODEL_STRING + ' Confusion matrix',
                          cmap=plt.cm.inferno,
                          )

path_confusionmatrix_out = os.path.join(PATH_MODELS,MODEL_STRING+" confusion"+".pdf")
plt.savefig(path_confusionmatrix_out,dpi=300,format='pdf',papertype='A4')
# Save the confusion matrix

#dir(plt.cm)
#%% One example
sample_selection = range(0,100)
sample1 = test_X.iloc[sample_selection,:]
print(sample1)
predict1 = clf_grid_BEST.predict_proba(sample1)
predicted_cat_num1 = predict1.argmax(axis=1)
predicted_cat_str1 = pd.Series(le_cat.inverse_transform(predicted_cat_num1))
#print()
predicted_values = predicted_cat_str1.values

actual_values = test_Y.iloc[sample_selection].astype(int)
actual_values = le_cat.inverse_transform(actual_values)

for pred,act in zip(predicted_values,actual_values):
    print("{:50} {}".format(pred,act))

#%% Save summary text
    
path_text_summary_out = os.path.join(PATH_MODELS,MODEL_STRING+" summary"+".txt")
with open(path_text_summary_out, 'a') as f:
    print(MODEL_STRING, file=f)
    print("Subset for Grid Search; {} rows (Fraction {})".format(len(sample_rows),SAMPLE_FRAC), file=f)
    print("Grid search completed over {} folds, {} parameter sets, over {} (H:M:S)".format(cv_folds,len(df_grid_agg),grid_search_elapsed))
    print("Best grid score:", clf_grid.best_score_, file=f)
    print("Best grid parameters:", clf_grid.best_params_, file=f)
    print("Predicted on test set with {} records".format(len(test_X)), file=f)
    print("Test set log-loss",test_logloss, file=f)
    print("Test set accurcacy",test_accuracy, file=f)
    print("{:50} {}".format('pred','act'),file=f)
    for pred,act in zip(predicted_values,actual_values):
        this_pred = "Wrong"
        if pred == act:
            this_pred = "Correct"
        print("{:10} {:50} {}".format(this_pred,pred,act),file=f)

