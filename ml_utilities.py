# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold   
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def plot_distribution(series):
    ''' Plot data compared and compare it to a normal distribution '''

    # make sure it's a pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
  
    # print summary statistics
    mu, sigma, skew = series.mean(), series.std(), series.skew()
    print('mu = {:.2f}, sigma = {:.2f}, skew = {:.2f}'.format(mu, sigma, skew))
    
    # plot the distribution
    sns.distplot(series, fit=stats.norm)
    
def hyperparam_search(base_clf, parameters, scores, X_train, X_test, y_train, 
                      y_test, strategy='grid'):
    ''' 
    Do a grid search and return the final classifer if only evaluating 
    on one metric 
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

def plot_roc(models, X_test, y_test):
    ''' Plot ROC curves for multiple models '''
   
    # loop over all models in the dictionary models 
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    for name, clf in models.items():
    
        # get the fpr and tpr
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
        roc_auc = auc(fpr, tpr)

        # plot curves
        plt.plot(fpr, tpr,lw=2, label='{:4.3f}: {}'.format(roc_auc, name))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # format plot
    plt.xlim([-.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()

class LogisticRegression(LogisticRegression):
    '''
    Wrapper Class for sklearn's LogisticRegression class  
    Calculates std errors, z-scores, p-values, and CIs for each coefficient
    '''
           
    def fit(self, X, y, alpha=0.05):
        ''' 
        Fit the model and also calculate some important statistics 
        Code taken from: https://stats.stackexchange.com/questions/89484/how-
          to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
        '''
        
        # fit the model
        super().fit(X, y)
               
        # design matrix: add column of 1's if there's an intercept
        if self.get_params()['fit_intercept']:
            X_design      = np.hstack((np.ones((X.shape[0], 1)), X))  
            self.params   = np.hstack((self.intercept_, self.coef_[0]))
        else:
            X_design    = X
            self.params = self.coef_[0]
        
        # fill diagonal of V  with each predicted observation's variance
        predProbs = np.matrix(self.predict_proba(X))
        V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
        np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)

        # covariance matrix
        self.covLogit = np.linalg.inv(np.dot(np.dot(X_design.T,V),X_design))
        
        # standard errors
        self.stderr = np.sqrt(np.diag(self.covLogit))
        
        # z-scores
        self.z_scores = self.params/self.stderr 
        
        # p-values
        self.p_values = [stats.norm.sf(abs(x))*2 for x in self.z_scores] 
        
        # confidence interval
        q = stats.norm.ppf(1 - alpha / 2)
        lower = self.params - q * self.stderr
        upper = self.params + q * self.stderr
        self.conf_int = np.dstack((lower, upper))[0]
        
    def summary(self, features, alpha=0.05):
        '''
        Print out summary statistics for the trained model
        '''
        
        # check if there's an intercept and save the features
        if self.get_params()['fit_intercept']:
            features = ['Intercept'] + features
        self.features = features

        # print everything
        print("{:>11s}|{:>9s}|{:>9s}|{:>9s}|{:>9s}|{:>9s} {:>9s}|{:>9s}".format("Name", "Coef.", \
                "Std.Err.", "z-score", "p-value", "[0.025", "0.975]", "Signif."))
        print(85*'-')
        for i in range(len(self.p_values)):
            print("{:10.10s} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | [{:7.4f}  {:7.4f}]| {:>6s}".format(\
                    self.features[i], self.params[i], self.stderr[i], self.z_scores[i], self.p_values[i], \
                    self.conf_int[i][0], self.conf_int[i][1], str((self.p_values[i] < alpha))))
            
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
            
class AveragingModels():
    ''' Average several models together '''
    def __init__(self, models):
        self.models = models
        
    # fit to the data
    def fit(self, X, y):      
        # Train base models
        for model in self.models:
            model.fit(X, y)
        return self
    
    # return the mode of the predictions for each model
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return stats.mode(predictions, axis=1)[0]
    
    # return the average probabilities from each model
    def predict_proba(self, X):
        predictions = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models
        ])
        return np.mean(predictions, axis=1)
