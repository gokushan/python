from uu import Error
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
import random

########################################################################
# Categoriza una columna de texto
def categorizar_datos(valor):
    if valor.dtype == 'object':
        return valor.astype("category").cat.codes
    else:
        return valor


########################################################################
# Devuelve la serie sin valores nulos y convertidos al valor indicado en valor
def eliminar_nulos(serie, valor):
    if isinstance(serie, pd.Series):
        return serie.fillna(valor)
    else:
        raise TypeError("El parametro de entrada no es una serie Panda.")


######################################################################3
# Genera un array numpy con todas las combinaciones posibles de 1s y 0s para todas las caracteristicas de los datos
def generar_array_caracteristicas_presentes(numColumnas):
    if isinstance(numColumnas, int) and numColumnas > 0:
         raise TypeError("El parametro de entrada no es un entero mayor que cero.")
    
    combinaciones = np.array(list(product([0, 1], repeat=numColumnas)))
    # filtro y obtengo solo las filas con mas de un elemento de la matriz a 1. Las filas con todo 0s o con un solo 1 no me aportan nada
    return combinaciones[np.sum(combinaciones,axis=1)>1]


###########################################################################
# Genera un dataframe con todas las combinaciones posibles de 1s y 0s 
# para todas las caracteristicas de los datos y con las etiquetas de columnas

def generar_df_caracteristicas_etiquetados(nombreColumnas):

    if isinstance(nombreColumnas,pd.core.indexes.base.Index) == False:
        raise TypeError("El parametro de entrada no es un índice de una serie Panda.")
    combinaciones = np.array(list(product([0, 1], repeat=nombreColumnas.size)))
    # filtro y obtengo solo las filas con mas de un elemento de la matriz a 1. Las filas con todo 0s o con un solo 1 no me aportan nada para entrenar los modelos
    miarray = combinaciones[np.sum(combinaciones,axis=1)>1]
    return pd.DataFrame(miarray, columns=nombreColumnas)



########################################################################################
# Lanza el modelo las veces indicadas en num_iteraciones usando el dataframe de combinaciones de features
# Devuelve:
#       dfResultados => DataFrame con los resultado del lanzamiento del modelo con esos hiperparametros. Se guardan todos los indices y el score de cada iteración
#       indiceMaximo => Devuelve el indice del dataframe dfResultados que ha conseguido el valor máximo
#       valorMaximo => Devuelve el score máximo correspondiente con el indiceMaximo
#
# Parámetros
#       model => Modelo que se esta entrenando
#       df_combinaciones => Columnas de un dataframe Panda a partir de la cual se generan la matriz de 1 y 0s con todas las combinaciones de features posibles
#       df_features_normal => valores X (features)
#       y => valores y  
#       hiperParametros => diccionario con configuración de hiperparametros
##########################################################################################################################33

def GetScoreModel(model, df_combinaciones, df_features_normal, y, hiperParametros):
    
    if model is None:
        raise TypeError("El modelo no esta instanciado.")
    
    if isinstance(df_combinaciones, pd.core.frame.DataFrame) == False:
        raise TypeError("El data Frame de combinaciones no es del tipo Data Frame.")
    
    if isinstance(df_features_normal, pd.core.frame.DataFrame) == False:
        raise TypeError("El data Frame de X no es del tipo Data Frame.")
    
    if isinstance(y, pd.core.series.Series) == False:
        raise TypeError("La clase y no es una serie Panda.")
    
    if (isinstance(hiperParametros["numIteraciones"],int)) == False or (hiperParametros["numIteraciones"]<=0) :
        raise TypeError("El hiperparametro numero de iteraciones no es un entero válido.")
    
    if (isinstance(hiperParametros["crossValue"],int)) == False or (hiperParametros["crossValue"]<=0) :
        raise TypeError("El hiperparametro cross_value no es un entero válido.")
    
    if (isinstance(hiperParametros["valorInicial"],int)) == False or (hiperParametros["valorInicial"]<0) :
        raise TypeError("El hiperparametro valor inicial no es un entero válido.")
    
    if (isinstance(hiperParametros["valorFinal"],int)) == False or (hiperParametros["valorFinal"]<=0) :
        raise TypeError("El hiperparametro valor final no es un entero válido.")

    if (hiperParametros["valorFinal"] <= hiperParametros["valorInicial"]) :
        raise TypeError("El hiperparametro valor final es mayor o igual que valor inicial.")
    
    if (hiperParametros["iteraciones_secuencia"] not in [True, False]) :
        raise TypeError("El hiperparametro iteraciones_secuencia tiene que ser True o False.")


    # Obtengo una lista de indices de columnas aleatorios para entrenar el modelo solo si las iteraciones no van en secuencia
    if (hiperParametros["iteraciones_secuencia"]) == False:
        bucle = [random.randint(hiperParametros["valorInicial"], hiperParametros["valorFinal"]) for _ in range(hiperParametros["numIteraciones"])]
    # almaceno las features usadas y scores obtenidos en la iteracion n
    lista_columnas = []

    # Ejecuto el modelo tantas veces como se indique en num_iteraciones
    for i in range(0, hiperParametros["numIteraciones"]): 
        # Obtengo la fila de la iteracion i
        if (hiperParametros["iteraciones_secuencia"]) == False:
            fila = df_combinaciones.iloc[bucle[i],:]
        else:
            fila = df_combinaciones.iloc[hiperParametros["valorInicial"]+i,:]
        # Obtengo las columnas con el valor 1
        columnas_df = fila[fila==1]
        X = df_features_normal[columnas_df.index]
        score = cross_val_score(model,X.values,y,cv=hiperParametros["crossValue"]).mean()
        lista_columnas.append([columnas_df.index,score])

    # Convierto la lista con las columnas usadas y los scores a un dataframe
    dfResultados = pd.DataFrame(lista_columnas)
    
    # Utiliza el método idxmax() para encontrar el índice del valor máximo
    indiceMaximo = dfResultados[1].idxmax()
    
    # Utiliza el valor máximo y el índice máximo para obtener los resultados
    valorMaximo = dfResultados.at[indiceMaximo, 1]
    
    return dfResultados, indiceMaximo, valorMaximo

# ----------------------------------------------------------------------------------------------------------------------------------
# Clase para obtener los valores extremos de una serie Panda numérica.
# Se usa para obtener un valor máximo y mínimo por el cual sustituir los valores extremos de una serie
class ExtremeValues:

    _MAXMULTIPLICADOR = 5

    def __init__(self, factorMultiplicador:float = 1.5, q1:float = 0.25, q2:float = 0.5, q3:float=0.75):

        if (factorMultiplicador <=0):
             raise TypeError("El factorMultiplicador no puede ser menor o igual a 0.")
        if (q1 <= 0 or q2 <=0 or q3 <=0):
            raise TypeError("Los cuartiles no pueden ser inferiores o iguales a 0.")
        if (q1 >= 1 or q2 >=1 or q3 >=1):
            raise TypeError("Los cuartiles no pueden ser mayores que 1.")
        if (q1>q2 or q2>q3 or q1>q3):
            raise TypeError("Los cuartiles q1, q2 y q3 no guardan una gradación de mayor a menor.")
        if (q1==q2 or q2==q3 or q1==q3):
            raise TypeError("Los cuartiles q1, q2 y q3 no pueden ser iguales 2 o 2 o los 3.")


        self._factorMultiplicador = factorMultiplicador
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3

    @property
    def factorMultiplicador(self):
        return self._factorMultiplicador

   
    @factorMultiplicador.setter
    def factorMultiplicador(self, value):
        if value > self._MAXMULTIPLICADOR:
            raise TypeError("El valor del factor multiplicador no puede ser mayor que 5.") 
        self._factorMultiplicador = value

    @property
    def q1(self):
        return self._q1

    @property
    def q2(self):
        return self._q2

    @property
    def q3(self):
        return self._q3

    # Este método recibe una serie panda y si la serie no es un objeto (categorica) devuelve:
    # ValorMAX del IQR -> percentil 75% + (IQR *1.5)
    # ValorMIN del IQR -> percentil 25% - (IQR *1.5)
    # No devuelve el IQR sino el valor extremo a sustituir por la parte superior de la serie panda y por la parte inferior
    def GetExtremeValueMaxMin(self, seriePanda):

        if isinstance(seriePanda, pd.core.series.Series) == False:
            raise TypeError("El parámetro indicado no es una serie Panda.")
        
        if seriePanda.dtype == 'object':
            raise TypeError("La serie Panda es de tipo objeto y no se puede calcular el valor extremo superior e inferior")

        q1, q2, q3 = seriePanda.quantile([self._q1, self._q2, self._q3])
        IQRMax = q3 - q2
        IQRMin = q2 - q1
        vMax = q3 + (self._factorMultiplicador * IQRMax)
        vMin = q1 - (self._factorMultiplicador * IQRMin)
        return  vMax, vMin
    
#########################################################################################################################
# ------------------------------------------------------------------------------------------------------------------------  
class TransformerData:
    
    def __init__(self, vAnomalos:ExtremeValues, numValoresUnicos: int=4):
        
        self._valoresExtremo = vAnomalos
        self._numValoresUnicos = numValoresUnicos
    
    # Cambia los valores por encima del valor extremo superior e inferior en una de las series panda del DataFrame
    # Solo modifica estas series si son numéricas o bien contienen un número de valores únicos superiores al parámetro
    # especificado en numValoresUnicos. Por defecto, 
    def _ApplyToSeriesExtremeValues(self, seriePanda:pd.core.series.Series):
    
        if seriePanda.dtype == 'object':
            return seriePanda
       
        if len(seriePanda.unique()) <= self._numValoresUnicos:
            return seriePanda

        vExtreme = self._valoresExtremo.GetExtremeValueMaxMin(seriePanda)
        
        # Obtengo primero los valores extremos de rango superior
        serieValoresExtremosSup = seriePanda.loc[seriePanda > vExtreme[0]].copy()
        if (serieValoresExtremosSup.count()>0):
            serieValoresExtremosSup[:] = vExtreme[0]
            seriePanda.loc[serieValoresExtremosSup.index] = serieValoresExtremosSup
        
        # Obtengo los valores extremos de rango inferior
        serieValoresExtremosInf = seriePanda.loc[seriePanda < vExtreme[1]].copy()
        if (serieValoresExtremosInf.count()>0):
            serieValoresExtremosInf[:] = vExtreme[1]
            seriePanda.loc[serieValoresExtremosInf.index] = serieValoresExtremosInf
        
        return seriePanda

    # Solo aplico el cambio de los valores extremos a las columnas del dataframe
    # que no son categoricas y son numericas    
    def ConvertDframeApplyExtremeValues(self, dataFrame:pd.core.frame.DataFrame):
        
        if isinstance(dataFrame, pd.core.frame.DataFrame) == False:
            raise TypeError("El data Frame pasado no es del tipo Data Frame.") 
        
        return dataFrame.apply(self._ApplyToSeriesExtremeValues)
        
  
        
