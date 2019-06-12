import numpy as np

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#-----------------------------------------------------------------------------

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
            

#-----------------------------------------------------------------------------

class AverageClassifierModels():
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
        predictions = [model.predict_proba(X) for model in self.models]
        return np.mean(predictions, axis=0)

#-----------------------------------------------------------------------------

class BoostedClassifierModels():
    ''' Average several models together '''
    def __init__(self, models):
        self.models = models
        
    # fit to the data
    def fit(self, X, y):      
        # Train base models
        for i, model in enumerate(self.models):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3, random_state=i)
            model.fit(X_train, y_train)
        return self
    
    # return the mode of the predictions for each model
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return stats.mode(predictions, axis=1)[0]
    
    # return the average probabilities from each model
    def predict_proba(self, X):
        predictions = [model.predict_proba(X) for model in self.models]
        return np.mean(predictions, axis=0)

#-----------------------------------------------------------------------------
