import random
import pandas as pd
import numpy as np
import seaborn as sns
import Utilities as util


#ruta_del_archivo = 'taxis_categorizado.pkl'
#df_taxis = pd.read_pickle(ruta_del_archivo)
df_taxis = sns.load_dataset('taxis')
print(df_taxis.head())
datos = util.NullUtilities.ConvertNullsToFrecuencyShow(df_taxis,"pickup_zone")
print(datos)


