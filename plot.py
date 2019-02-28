import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

#-----------------------------------------------------------------------------

def plot_distribution(series):
    ''' Plot data and compare it to a normal distribution '''

    # make sure it's a pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
  
    # print summary statistics
    mu, sigma, skew = series.mean(), series.std(), series.skew()
    print('mu = {:.2f}, sigma = {:.2f}, skew = {:.2f}'.format(mu, sigma, skew))
    
    # plot the distribution
    sns.distplot(series, fit=stats.norm)

#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
