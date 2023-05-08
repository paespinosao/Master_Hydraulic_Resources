# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:57:41 2018

@author: Unalmed
"""

###codigo para editar el archivo del awac 600 y dividirlo por burts



from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
import scipy.stats
import xlsxwriter as xlsxwl # Crear archivos de Excel
import pandas as pd
import scipy.stats as scp
from datetime import datetime 


path='./AW101.wad'
Data = np.genfromtxt(path,delimiter='',dtype=str,skip_header=0)




Datos=[]
for i in range(len(Data)):
    Datos.append(' '.join(Data[i]))
    
NDatos=1024
inicio=0
for i in range ((len(Datos))/NDatos):  
    
    with open('AW600'+str(i+100)+'.WAD','a') as archivo:
        for j in range(inicio,NDatos):
            line = Datos[j]
            archivo.write(line)
            archivo.write("\n")
    inicio+=1024
    NDatos+=1024
            
   