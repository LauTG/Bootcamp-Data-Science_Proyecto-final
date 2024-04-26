import pandas as pd
import numpy as np
import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

from prepro import load_data,FE,nulls

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

def seleccion(df):
    todrop=list(df.columns[df.dtypes=='datetime64[ns]'])+['t_evento','exitus','CF','edad',
                                                         'disfagia', 'ERGE','PAS','PAD']
    df.drop(columns=todrop,inplace=True)
    df.dropna(axis=0,inplace=True)
    print(f'El dataset contiene {df.shape[0]} filas y {df.shape[1]} columnas')
    return df

##############################################################################
def best_grid(grid,xtrain,ytrain):
    """Parametros del mejor modelo grid y lo
    devuelve ajustado"""
    print("Mejor score:", grid.best_score_)
    print("Mejores parametros:",grid.best_params_)
    print('Ajustando el mejor modelo...')
    return grid.best_estimator_.fit(xtrain,ytrain)
#################################################################################3
def coefi(pipe,nombre):
    """Devuelve un df con los coef/pesos ordenados en valor absoluto
    (nombre=nombre del modelo dentro del pipe)."""
    if (nombre=='logit') | (nombre=='svm'):
        coef=[round(x,2) for x in pipe.named_steps[nombre].coef_[0]]
    elif (nombre=='rf') | (nombre=='gb'):
        coef=[round(x,2) for x in pipe.named_steps[nombre].feature_importances_]
    pesos=pd.DataFrame(zip(pipe.named_steps['prepro'].get_feature_names_out(),coef),
                 columns=['Variable','Coeficientes']).set_index('Variable')
    pesos=pesos[pesos['Coeficientes']!=0]
    pesos=pesos.reindex(pesos['Coeficientes'].abs().sort_values(ascending=False).index)
    return pesos
###################################################################################
def metricas(modelo,xtrain,ytrain,xtest,ytest):
    """Sensibilidad, report classification del test y 
    matrices de confusión de train vs test"""
    print("{:=^50}".format(" Report classification "))
    print('')
    ptrain=modelo.predict(xtrain)
    print(f'Sensibilidad del train: {metrics.recall_score(ytrain,ptrain):.2f}')
    ptest=modelo.predict(xtest)
    print(f'Sensibilidad del test: {metrics.recall_score(ytest,ptest):.2f}')
    print('')
    print(classification_report(ytest,ptest,target_names=['No','Si']))
    print('')
    print("{:=^50}".format(" Confusion matrix train vs test "))
    print('')
    cfm1 = ConfusionMatrixDisplay(confusion_matrix(ytrain, ptrain)) 
    cfm2 = ConfusionMatrixDisplay(confusion_matrix(ytest, ptest)) 
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    cfm1.plot(ax=axs[0])
    cfm2.plot(ax=axs[1])
##########################################################################################3
