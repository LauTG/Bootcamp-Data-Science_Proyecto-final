# Bootcamp-Data-Science_Proyecto-final
Proyecto final del bootcamp en Data Science titulado "Predicción de mortalidad temprana en pacientes con esclerodermia"

Este proyecto ha sido realizado con unos datos de acceso público que provienen de una investigación realizada en Tailandia y publicada en la revista científica Nature "Development and validation of machine learning for early mortality in systemic sclerosis".

La esclerodermia es una enfermedad minoritaria, crónica y de origen autoinmune, cuya etiología es aún desconocida, aunque se supone multifactorial. Es una enfermedad sistémica grave, que cursa con una afectación multiorgánica y cuya mortalidad es la mas alta entre las enfermedades autoinmunes sitémicas.

El objetivo de este proyecto es encontrar y entrenar el mejor modelo de machine learning supervisado que clasifique a los pacientes acorde a la variable objetivo "exitus5", definida como mortalidad temprana (dentro de los 5 años desde el inicio de la enfermedad) a partir de 34 variables clínicas. La problemática de este conjunto de datos es su tamaño reducido y el desequilibrio de clases en la variable objetivo, teniendo en cuenta que la métrica objetivo es la sensibilidad de la clase minoritaria y por lo tanto es la que se intenta maximizar.

Para poder afrontar esta problemática, se proponen diferentes algoritmos de machine learning que son son entrenados con distintas estrategias para mejorar y afrontar estas problemáticas:

1. Modelos:  
   a. Regresión logistica con regularización Ridge (Rlogistica.ipynb)  
   b. Support vector machine con regularización Ridge (SVM.ipynb)  
   c. Métodos de ensembling (Ensembling.ipynb)  
      - Random Forest
      - XGBoost (sin método oversampling)

3. Estrategias:  
   a. Estrategias básicas: estratificar la divisón train-test y elección de métrica más acorde al objetivo  
   b. Con ponderación de pesos de clase  
   c. Con Oversampling de clase minoritaria   

Para todos ellos se realiza una búsqueda de hiperparámetros y cross-validation con GridSearchCV.

Conclusiones:
- El modelo que mejor captura la relación entre las variables y  mejor clasifica a los pacientes teniendo en cuenta la sensibilidad de la clase minoritaria es la regresión logística, tanto con la estrategia de ponderación de pesos de clase como con oversampling, con una sensibilidad de la clase minoritaria en el test de 0.89
- El modelo SVM también alcanza una buena recall de 0.83 con ambos métodos.
- Los modelos con peor métricas son los métodos de ensembling.
