import numpy as np
import pandas as pd

from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold   
from sklearn.metrics import accuracy_score

#-----------------------------------------------------------------------------

def append_dummies(df, columns, drop_first=True):
    ''' Append dummy versions of columns to a dataframe '''
    for col in columns:
        dummy_col = pd.get_dummies(df[col], drop_first=drop_first)
        df[dummy_col.columns.tolist()] = dummy_col
    return df

#-----------------------------------------------------------------------------

def hyperparam_search(base_clf, parameters, scores, X_train, X_test, y_train, 
                      y_test, strategy='grid'):
    ''' 
    Do a grid search and return the best classifer if only evaluating on one 
    metric.

    Args:
        base_cls -- the base classifier to train
        parameters -- the hyperparameters and their bounds to search through
        scores -- the relevant metrics for picking a winner
        X_train -- training features
        X_test  -- testing features
        y_train -- training targets
        y_test  -- testing targets
        strategy -- whether to do a grid or randomized parameter search

    Returns:
        clf -- the best classifier by one metric or the results from the search
    '''

    # loop over the different scores
    for score in scores:

        # pick the strategy for searching parameters
        if strategy == 'grid':
            clf = GridSearchCV(base_clf, parameters, cv=KFold(5), scoring=score)
        else:
            clf = RandomizedSearchCV(base_clf, parameters, cv=KFold(5), 
                                     scoring=score, n_iter=20)
        clf.fit(X_train, y_train)

        print("Best parameters set found:\n")
        print(clf.best_params_)
        print()
        print("Grid scores:\n")
        means = clf.cv_results_['mean_test_score']
        stds  = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
        
    # train the final model on the full training set
    if len(scores) == 1:
        clf = clf.best_estimator_.fit(X_train, y_train)
        
    return clf

#-----------------------------------------------------------------------------

def nested_cv(base_model, parameters, train_df, features, target, 
              strategy='grid'):
    ''' 
    Do nested CV and return the mode of the most common set of parameters that 
    win out in the training 
    '''
    
    # loop over outer folds
    params, scores = [], []
    outer_cv = KFold(10, shuffle=True)
    for train_cv_ind, test_ind in outer_cv.split(train_df[features], 
                                                 train_df[target]):

        # pull out the training/cv and the test sets
        train_cv = train_df.iloc[train_cv_ind]
        test     = train_df.iloc[test_ind]

        # do parameter search
        if strategy == 'grid':
            model = GridSearchCV(base_model, parameters, 
                                 cv=KFold(5, shuffle=True))
        else:
            model = RandomizedSearchCV(base_model, parameters, 
                                       cv=KFold(5, shuffle=True), n_iter=20)
        model.fit(train_cv[features], train_cv[target])

        # score the model
        scores.append(accuracy_score(test[target], 
                                     model.predict(test[features])))
        
        # save the parameters
        params.append(model.best_params_)
    
    # get the mode of the list of best parameters
    param_list  = [str(p) for p in params]
    best_params = eval(stats.mode(param_list)[0].tolist()[0])

    print("Best parameter set found:\n")
    print(best_params)
    print()
    print("Grid scores:\n")
    for score, param in zip(scores, params):
        print('{:.3f} for {}'.format(score, param))
    
    return best_params

#-----------------------------------------------------------------------------

class Pipeline(Pipeline):
    '''
    Wrapper Class for sklearn's Pipeline so the summary can be printed
    '''

    def summary(self, features, alpha=0.05):
        '''
        Print out summary statistics for the trained model
        '''       
        
        if self._final_estimator is not None:
            self._final_estimator.summary(features)
            
#-----------------------------------------------------------------------------



