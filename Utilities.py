import numpy as np
import pandas as pd

class NullUtilities:
    

    def __init__(self):
        return
    
    @staticmethod
    def ConvertNullsToFrecuencyShow(df:pd.DataFrame,columnNameWithNulls: str=''):

        if isinstance(df, pd.core.frame.DataFrame) == False:
            raise TypeError("There isn't valid dataframe.")

        if (len(columnNameWithNulls) == 0):
            raise TypeError("columnNameWithNulls must be a value not null.")
        
        if columnNameWithNulls not in df.columns:
            raise TypeError("The column called columnNameWithNulls doesn't exist in dataframe.")

        # Obtengo el numero de filas con nulos y las filas con nulos       
        df_aux = df[df.pickup_zone.isna() == True]
        numRowNull = df_aux.shape[0]
        # Obtengo la frecuencia de aparicion de cada valor junto con su indice
        frecuency = df[columnNameWithNulls].value_counts(normalize=True, sort=True, ascending=False)
        
        return frecuency
    