import pandas as pd
import numpy as np
import datetime 
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path,info):
    """Carga y limpieza del df. es una fx específica de este df. info puede ser True o False"""
    df = pd.read_excel(path)
    if info==True:
        print(f"- El dataset original contiene {df.shape[0]} filas y {df.shape[1]} variables")
    ## Columnas eliminadas
    todrop=['id','fc2','fc3','visit','dur_fu','cause_die_all','death_other_01','dur_die','fuless5y','dead5y']
    df.drop(columns=todrop, inplace=True)
    ## Cambiar a castellano variables
    vars=['sexo','f_nacimiento','PSNR','f_debut','f_fin','peso','PA','subtipo','mRSS','CF','raynaud','ulceras',
       'discromias','roces_tendinosos','contracturas_flex','artritis','EPID','HAP','FEVI','derrame_per','Acardiaca',
      'crisis_renal','disfagia','ERGE','Aesofagica','Agastrica','Aintestinal','perdida_peso','anemia','exitus',
       'f_exitus']
    df = df.set_axis(vars, axis=1)
    ## Categorias PSNR a castellano
    PSNRv=dict(zip(list(df.PSNR.unique()),['Endurecimiento piel','Puffy hand','Otros','Debilidad','Articular','Disfagia','Disnea']))
    df.PSNR = df.PSNR.apply( lambda x : PSNRv[x])
    ## Duplicados
    if (df.duplicated().sum()==0) & (info==True):
        print('- El dataset no tiene filas duplicadas')
    elif (df.duplicated().sum()>0) & (info==True):
        df.drop_duplicates(keep="first", inplace=True)
        print(f"- El dataset tiene {df.duplicated().sum()} filas duplicadas que han sido eliminadas")
    elif (df.duplicated().sum()>0) & (info==False): 
        df.drop_duplicates(keep="first", inplace=True)
    else:
        pass
    return df
#####################################################################
def nulls(df,n,info):
    """Proporción de missings, si es mayor a n% la var se elimina. info=True/False"""
    if info==True:
        print(f'- Proporción de missings (variable se elimina si >{n}%):')
    for i in df.columns[df.isnull().any()]:
        if (df.isnull().sum()[i]/len(df)*100>n) & (info==True):
            if (i == 'f_exitus') or (i == 't_evento'):
               continue
            print(f"   > {i} : {df.isnull().sum()[i]/len(df)*100:.1f}% --> Esta variable ha sido eliminada")
            df.drop(columns=i,inplace=True)
        elif (df.isnull().sum()[i]/len(df)*100>n) & (info==False):
            if (i == 'f_exitus') or (i == 't_evento'):
               continue
            df.drop(columns=i,inplace=True)
        elif (df.isnull().sum()[i]/len(df)*100<=n) & (info==True):
            print(f"   > {i} : {df.isnull().sum()[i]/len(df)*100:.1f}%")
        else:
            pass
    return df
#########################################################################
def FE(df):
    """Creación de nuevas variables, fx específica de este df"""
    ## Variables datetime
    df['f_exitus']=np.where(df['f_exitus']=='0/0/0000',np.nan,df['f_exitus'])
    df['f_exitus'] = pd.to_datetime(df['f_exitus'], format='%m/%d/%Y',exact=False)
    ## Nueva variable edad y edad al debut
    df['edad']=((df['f_fin'] - df['f_nacimiento'])/365).apply(lambda x: x.days)
    df['edad_debut']=((df['f_debut'] - df['f_nacimiento'])/365).apply(lambda x: x.days)
    ## Nueva variable tiempo hasta la muerte
    df['t_evento']=(df['f_exitus']-df['f_debut']).apply(lambda x: x.days/365.25)
    ## Target (muerte prematura <= 5.9 años)
    df['exitus5'] = np.where(pd.isnull(df['t_evento']) | (df['t_evento']>5.5), 0, 1)
    ## Divisón de PA en sistolica y diastolica
    df[['PAS', 'PAD']] = df['PA'].str.split('/', n=1, expand=True).astype('Int64')
    df.PAD.replace(0, np.nan, inplace=True)
    df.PAS.replace(0, np.nan, inplace=True)
    df.drop('PA',axis = 1,inplace = True)
    ## categorización CF
    df['CFc']=np.where(df['CF']==1,'CF1',
                                np.where(df['CF']==2,'CF2',
                                         np.where(df['CF']>2,'CF3-4',None)))
    df['CFc'].fillna(np.nan, inplace=True)
      #np.where(df['CF'].isnull(),np.nan,
    ## Cambio nivel ref en sexo y subtipo
    df.sexo=np.where(df.sexo==0,1,0)
    df.subtipo=np.where(df.subtipo==0,1,0)
    ## Convertir a categ
    tocat=['sexo','subtipo','raynaud','ulceras','discromias','roces_tendinosos','contracturas_flex','artritis',
           'EPID','HAP','crisis_renal','disfagia','ERGE','Aesofagica','Agastrica','Aintestinal','perdida_peso',
           'anemia','exitus','exitus5']
    df[tocat]=df[tocat].apply(lambda x: x.astype('Int64').astype('category'))
    df['PSNR']=df['PSNR'].astype('category')
    df['CFc']=df['CFc'].astype('category')
    return df  

################################################################33
def prepro(path,n,info):
    df=(load_data(path,info)
        .pipe(FE)
        .pipe(nulls,n,info))
    if info==True:
        print(f"Dimensiones finales tras preprocesamiento: {df.shape[1]} variables y {df.shape[0]} filas")
    return df
##########################################################################
def pprint(df,capt,decimales):
    """Tablas con formato (df), título (capt)
    y precisión de decimales (decimales)"""
    from IPython.display import display
    display(df.style \
    .format(precision=decimales, thousands=".", decimal=",") \
    .format_index(str.upper,axis=1) \
    .set_caption(capt).set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'white'),
        ('font-size', '16px'),
        ('font-weight', 'bold')]}]))
#########################################################################3
def set_levels(df):
    """Crea o modifica un df con los niveles de las categoricas con los nombres
    adecuados. es una fx espcífica de este df"""
    df=df.copy()
    for i in df.columns[df.dtypes=='category']:
        if i == 'sexo':
            df[i]=df[i].cat.rename_categories({0:'Mujer',1:'Hombre'})
        if (i == 'PSNR') | (i=='CFc'):
            continue
        if i == 'subtipo':
            df[i]=df[i].cat.rename_categories({0:'Limitada',1:'Difusa'})
        else:
            df[i]=df[i].cat.rename_categories({0:'No',1:'Si'})
    return df
############################################################################
def countplots(df,grupo):
    """Countplots de todas las v.categoricas del df. por grupo (hue)"""
    sns.set_theme()
    plt.figure(figsize=(10, 15),layout='constrained')
    plt.suptitle(f'Frecuencias relativas de las variables categóricas por {grupo} (%)',fontweight="bold")
    
    for i, column in enumerate(df.columns[df.dtypes=='category'], start=1):
        plt.subplot(6, 4, i)
        if len(df[column].unique())>4:
            sns.countplot(x=df[column],stat='percent',hue=df[grupo],alpha=0.5)
            plt.xticks(rotation=90)
            plt.ylabel('')
            plt.xlabel('')
            plt.title(column,fontweight="bold")
        else:
            sns.countplot(x=df[column],stat='percent',hue=df[grupo],alpha=0.5)
            plt.ylabel('')
            plt.xlabel('')
            plt.title(column,fontweight="bold")
    plt.show()
################################################################################
def histograms(df,grupo):
    """ Histogramas de todas las numericas del df por grupo (hue)"""
    sns.set_theme()
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'Histogramas de las variables numéricas por {grupo}',fontweight="bold")
    
    for i, column in enumerate(df._get_numeric_data().columns.tolist() , start=1):
        plt.subplot(2, 4, i)
        sns.histplot(x=df[column], kde=True,hue=df[grupo],alpha=0.5)
        plt.ylabel('')
        plt.xlabel('')
        plt.title(column,fontweight="bold")
    plt.show()
################################################################################
def chi2(df,a,h):
    """test chi-cuadrado de independencia entre v.categóricas, representación
    en un heatmap de los p.valores. a=ancho figura, h=alto figura"""
    from scipy.stats import chi2_contingency
    res=pd.DataFrame([[round(chi2_contingency(pd.crosstab(df[x],df[y])).pvalue,2) for y in 
                       df.columns[df.dtypes=='category']] for x in df.columns[df.dtypes=='category']],
                     columns=df.columns[df.dtypes=='category'], index=df.columns[df.dtypes=='category'])
    mask = np.zeros_like(res)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
     f, ax = plt.subplots(figsize=(a,h),layout='constrained')
     f.suptitle('Asociaciones entre v.categóricas: p-valores del test chi-cuadrado',fontweight="bold",x=0.6)
     ax = sns.heatmap(res, mask=mask, center=0, square=True, linewidths=2, annot=True, cbar_kws={'shrink': .5})
    plt.show()
#############################################################################
def corr(df,a,h):
    """ Heatmap de la matriz de correlaciones, a=ancho, h=alto"""
    corr=df._get_numeric_data().corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
     f, ax = plt.subplots(figsize=(a,h),layout='constrained')
     f.suptitle('Asociaciones entre v.numéricas: coeficientes de correlación',fontweight="bold",x=.6)
     ax = sns.heatmap(corr, mask=mask, center=0, square=True, linewidths=2, annot=True, cbar_kws={'shrink': .5})
    plt.show()
################################################################################
def ttest(df,grupo,n1,n2):
    """ttest para 2 muestras independientes y var desigual de todas las v.numericas
    del df vs una variable de agrupación dicotomica (grupo),n1=nivel 1 y n2=nivel 2"""
    from scipy.stats import ttest_ind
    res={}
    for i in df._get_numeric_data().columns.tolist():
        g0 = df.where(df[grupo]== n1).dropna()[i]
        g1 = df.where(df[grupo]== n2).dropna()[i]
        res[i] = round(ttest_ind(g0,g1,equal_var=False).pvalue,2)
    res = pd.DataFrame.from_dict(res,orient='index')
    res.columns = ['ttest pval']
    return res
  
