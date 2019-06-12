import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import roc_curve, auc

#-----------------------------------------------------------------------------

def plot_distribution(series):
    ''' 
    Plot data and compare it to a normal distribution 
    
    Args:
        series -- a pandas Series or a list that is turned into a series
    '''

    # make sure it's a pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
  
    # print summary statistics
    mu, sigma, skew = series.mean(), series.std(), series.skew()
    print('mu = {:.2f}, sigma = {:.2f}, skew = {:.2f}'.format(mu, sigma, skew))
    
    # plot the distribution
    sns.distplot(series, fit=stats.norm)

#-----------------------------------------------------------------------------

def plot_roc(models, X_test, y_test, name='Classifier'):
    ''' 
    Plots ROC curves for a single model or an ensemble of models
    
    Args:
        models -- the already trained model objects (or ensemble)
        X_test -- the test features
        y_test -- the test targets
    '''
   
    if not isinstance(models, dict):
        models = {name: models}

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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    fig.set_size_inches(4, 4)
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
#-----------------------------------------------------------------------------
def confusion_matrix_summary(cm):
    ''' Print out summary statistics of the classifier '''
    tn = cm[0, 0] # true negative
    tp = cm[1, 1] # true positive
    fn = cm[1, 0] # false negative
    fp = cm[0, 1] # false positive

    print('Sensitivity/Recall/True Positive Rate: {}'.format(round(tp / (tp+fn), 3)))
    print('       Specificity/True Negative Rate: {}'.format(round(tn / (tn+fp), 3)))
    print('                            Precision: {}'.format(round(tp / (tp+fp), 3)))
    print('            Negative Predictive Value: {}'.format(round(tn / (tn+fn), 3)))
    print('                  False Negative Rate: {}'.format(round(fn / (fn+tp), 3)))
    print('          Fallout/False Positive Rate: {}'.format(round(fp / (fp+tn), 3)))
    print('                 False Discovery Rate: {}'.format(round(fp / (fp+tp), 3)))
    print('                  False Omission Rate: {}'.format(round(fn / (fn+tn), 3)))
    print('                             Accuracy: {}'.format(round((tp+tn) / (tp+tn+fp+fn), 3)))
    print('                             F1 Score: {}\n'.format(round(2*tp / (2*tp + fp+fn), 3)))

#-----------------------------------------------------------------------------
    
def plot_feature_importance(trained_models):
    
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    for i, (name, pipe) in enumerate(trained_models.items()):

        clf = pipe.named_steps.clf
        # save the feature importance and sort
        feature_importance = {'%s' % features[i]: coef for i, coef in enumerate(clf.coef_[0])}
        ordered_features = sorted(feature_importance, key=lambda dict_key: abs(feature_importance[dict_key]))
        coefficients = [feature_importance[f] for f in ordered_features]
        
        # plot everything
        ax = fig.add_subplot(3,3,i+1) 
        sns.barplot(x=ordered_features, y=coefficients, ax=ax)
        plt.title("Log Odds")
        for item in ax.get_xticklabels():
            item.set_rotation(45)

    plt.show()
