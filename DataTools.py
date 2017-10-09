## New with MinMaxScaler as step in transform method

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from scipy.stats import boxcox, skew, skewtest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
mM = MinMaxScaler(feature_range=( .5, 1.5))

class Deskew(BaseEstimator, TransformerMixin):
    #mM = MinMaxScaler(feature_range=( .5, 1.5))
    def __init__ (self,alpha=1):
        self.alpha = alpha
    def _reset(self):
        pass
    def fit(self,X,y=None):
        self
    def transform(self,X):
        #mM = MinMaxScaler(feature_range=( .5, 1.5))
        #mM.fit(X)
        #X_minMax = pd.DataFrame( mM.fit_transform(X))
        #X_minMax = mM.fit_transform(X)
        #X_minMax = pd.DataFrame( mM.fit_transform(X))
        #box_cox_df = pd.DataFrame()
        boxed = list()
        lambdas = list()
        #for col in X_minMax.T:
        for col in X.T:
            boxcoxed, lam = boxcox( col)
            lambdas.append(lam)
            boxed.append( boxcoxed)
               

            #box_cox_df[col] = pd.Series( boxcoxed)
            #box_cox = box_cox_df.as_matrix()
        #return box_cox_df, lambdas
        box_cox = np.array( boxed).T
        return box_cox#, lambdas
    def fit_transform(self,X,y=None):
        return self.transform(X)
    def inverse_transform(self, X, lambdas):## Needs work
        X_s = pd.DataFrame()  ## Original skewed, non 0 centered or variance scaled
        for col, lam in zip(X.columns, lambdas):
            
            if lam != 0:
                reskewed_col = (lam*X[col] + 1)**(1/lam)
            else:
                reskewed_col = np.exp(X[col])
            X_s[col] = pd.Series( reskewed_col)
        X_o_s = pd.DataFrame( mM.inverse_transform(X_s))  ## descaled, deskewed data
        return X_o_s
        #return np.exp(X) - self.alpha
    def score(self,X,y):
        pass
        
        
        
from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification, make_multilabel_classification

data_sets = { 
    'blobs':make_blobs(n_samples=1000, n_features = 5, centers=3, cluster_std=0.5, random_state=42), 
    'circles': make_circles(n_samples = 1000, shuffle = True, factor = .5, noise = 0, random_state=42), 
    'moons': make_moons(n_samples = 1000, shuffle = True, noise = 0, random_state=42),
    'classification': make_classification(n_samples = 1000, n_features=20, n_redundant=3, n_informative=2,
                             n_clusters_per_class=1, n_classes=4, random_state=42)#,
    #'multilabel_classification': make_multilabel_classification( random_state=42) 
}

## Mean, Variance, and Skew normalization 
from sklearn.preprocessing import StandardScaler ## Mean translation and variance scaling 
from sklearn.preprocessing import MinMaxScaler ## scaling features to lie between a given minimum and maximum value
from sklearn.preprocessing import MaxAbsScaler ## maximum absolute value of each feature is scaled to unit size
from sklearn.preprocessing import RobustScaler ## scaling with MANY OUTLIERS
from sklearn.preprocessing import Normalizer ##  scaling individual samples to have unit norm
from sklearn.preprocessing import KernelCenterer
from scipy.stats import boxcox  ## Skew normalizer, No negative or zero values  


scalers = {
    'standard': StandardScaler(), ## Mean translation and variance scaling 
    'min_max': MinMaxScaler(),  ## scaling features to lie between a given minimum and maximum value
    'max_abs': MaxAbsScaler(), ## maximum absolute value of each feature is scaled to unit size
    'robust': RobustScaler(), ## scaling with MANY OUTLIERS
    'normalizer': Normalizer(), ##  scaling individual samples to have unit norm
    'k_center': KernelCenterer(),
    'box_cox': Deskew() ## return box-coxed, lambda (if not provided as input) - Skew normalizer, No negative or zero.
}

## Generators (Features)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem

generators = {
    'poly': PolynomialFeatures(),
    'fourier': RBFSampler(random_state = 42),
    'rbf': Nystroem(random_state=42)
}

## Dimension Reduction (feature_selection)

from sklearn.feature_selection import VarianceThreshold ## removes all features whose variance below threshold, default 0 variance removed
from sklearn.feature_selection import SelectFromModel ## any estimator that has a coef_ or feature_importances_ attribute after fitting
from sklearn.feature_selection import SelectPercentile ## Percent of features to keep.
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif ## removes all but the k highest scoring features
from sklearn.feature_selection import RFE ## select features by recursively considering smaller and smaller sets of features
from sklearn.feature_selection import GenericUnivariateSelect ## allows to perform univariate feature selection with a configurable strategy. 


f_selectors = {
    # ANOVA
    'var': VarianceThreshold(), ## removes all features whose variance below threshold, default 0 variance removed
    'sp': SelectPercentile(), ## Percent of features to keep.
    'skb': SelectKBest(), ## removes all but the k highest scoring features

    'sfm': SelectFromModel( estimator=None), ## any estimator that has a coef_ or feature_importances_ attribute after fitting 
    'rfe': RFE( estimator=None), ## select features by recursively considering smaller and smaller sets of features
    'gus': GenericUnivariateSelect() ## allows to perform univariate feature selection with a configurable strategy. 
}


#  Dimension Reduction (decomposition)

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import SparseCoder
from sklearn.decomposition import FastICA 

decomposers = {
    'pca': PCA(random_state=42),  ## decompose NxM dataset in a set of successive orthogonal components, explain a maximum amount of the variance, whiten = True to project onto singular space                  
    'kpca': KernelPCA(random_state=42), ##  non-linear dimensionality reduction - denoising, compression
    'svd': TruncatedSVD(n_components=2, random_state=42), ## works with sparse matrix, use tf-idf for nlp (LSA), if 0 mean scaled - equivalent to PCA
    'fa': FactorAnalysis(random_state=42),  ## generate latent variables, hidden in the noise
    'nmf': NMF(random_state=42), ## decomposition of samples X into two matrices W and H of non-negative elements
    'lda': LatentDirichletAllocation(random_state=42), ## generative probabilistic model for collections of discrete dataset, text corpora    
    
    'ipca': IncrementalPCA(), ## partial fit ## random_state=42
    'spca': SparsePCA(), ## extract sparse components that best reconstruct the data  random_state=42
    'fica': FastICA(), ## separates a multivariate signal into additive subcomponents that are maximally independent random_state=42
    #'scoder': SparseCoder(dictionary = dict({'0':0})),    
}

pre_processors = {
    'scalers': scalers,
    'f_selectors': f_selectors,
    'decomposers': decomposers,
    'generators': generators
}



## Models: classification estimators 

## .linear_model estimators
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier

## .tree estimators
from sklearn.tree import DecisionTreeClassifier

## .svm estimators
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

## Nearest Neighbors Clasifier

from sklearn.neighbors import KNeighborsClassifier
## Naive Bayes
from sklearn.naive_bayes import MultinomialNB

models = { # Classifiers
    'logit': LogisticRegression(random_state=42, n_jobs = -1),
    'sgd_c': SGDClassifier(random_state=42),
    
    'svc': SVC(random_state=42),
    #'nu_svc': NuSVC(),
    'l_svc': LinearSVC(random_state=42),   
    'dt_c': DecisionTreeClassifier(random_state=42),
    'prcpt': Perceptron(random_state=42),
    
    'multi_mb': MultinomialNB(),
    'knn': KNeighborsClassifier(n_jobs = -1)
    #'pa_c': PassiveAggressiveClassifier(),
}

## Ensemble methods 
from sklearn.ensemble import BaggingClassifier ## for .linear_model or .tree estimators
from sklearn.ensemble import AdaBoostClassifier  ## for .linear_model or .tree estimators 

from sklearn.ensemble import RandomForestClassifier ## .tree estimators
from sklearn.ensemble import ExtraTreesClassifier ## .tree estimators
from sklearn.ensemble import GradientBoostingClassifier ## .tree estimators


ensembles = {
    'rf_c': RandomForestClassifier(), ## Tree
    'et_c': ExtraTreesClassifier(), ## Tree
    'grb_c': GradientBoostingClassifier(),  ## Tree
    'bag_c': BaggingClassifier(),  ## Linear, Tree
    'adab_c': AdaBoostClassifier()  ## Linear, Tree
}


data_tools = {
    'pre_processing': pre_processors,
    'models': models,
    'ensembles':ensembles
}

## steps need to be in the for

def key_inversion( step_dict = dict()):
    match_dict = dict()
    for step_type, steps in step_dict.items():
        for step in steps:
            match_dict[step] = step_type
    return match_dict
            
#match_dict = { '' }  # dictionary lookup for methods in each type of pre_processor
match_dict = key_inversion( pre_processors)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 

def build_pipelines( *steps): #, pre_processors={}, models={}):  ## *params,  model = None,
    '''Steps need to be in the form "step:step_type" e.g. "standard:scalers".'''
    list_of_pipes = []
    #print(models)
    for model, estimator in models.items():
        #print( model)
        list_of_steps = []
        for step in steps:  ## steps are estimators from sets above
            step_type = match_dict[step]
            step_tuple = (step, pre_processors[step_type][step])
            list_of_steps.append( step_tuple)
        model_tuple = (model, models[model])
        list_of_steps.append( model_tuple )
        pipe = Pipeline( list_of_steps )
        #print( pipe)
        list_of_pipes.append( pipe )
    return list_of_pipes
    
    
    
