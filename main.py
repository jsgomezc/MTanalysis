# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:54:53 2020

@author: Juancho
"""
import time
t=time.time()
print("Loading libraries...")

from obspy import UTCDateTime
import numpy as np
import MTprocessing as MTp
import os

secs=time.time()-t
print("Libraries loaded in %d seconds.\n" % (secs))

network="BT"
stations="USME"			#["USME","TUNJ","VCIO"]
channel1="HQE"			#["HFN","HFE","HFZ","BH*"]
channel2="HFN"			#["HFN","HFE","HFZ","BH*"]
LocIds="*"				#["02","02","02"]

GeneralFolder= "/home/jsgomezc/Documentos/UN/TrabajoGrado-Pregrado/ARCHIVE_USME_2019"
FieldType = "D"

DateStart_s = "2019-12-24T00:00:00"
DateEnd_s   = "2019-12-25T00:00:00"

#DateStart_s2 = "2019-12-22T00:00:00"
#DateEnd_s2   = "2019-12-24T23:00:00"

interval_rho_hours = 1
interval_rho_secs  = interval_rho_hours * 3600

interval_sumspec   = 1000

overlap = 0.13

New_Folder = stations+' '+channel1+'-'+channel2+' '+DateStart_s+' - '+DateEnd_s+' ('+str(interval_sumspec)+'s)'+' ('+str(interval_rho_hours)+'h)'
New_Folder = New_Folder.replace(':','.')

DateStart = UTCDateTime(DateStart_s)
DateEnd   = UTCDateTime(DateEnd_s)

#DateStart2 = UTCDateTime(DateStart_s2)
#DateEnd2   = UTCDateTime(DateEnd_s2)

# Saber cuántos días se procesa
secs = DateEnd - DateStart
days = np.ceil(secs/(60*60*24))
'''
try:
    os.mkdir(New_Folder)
except OSError:
    print ("Creation of the directory %s failed" % New_Folder)
else:
    print ("Successfully created the directory %s " % New_Folder)
'''

# Just to know the starttime
st = MTp.read_signal(network, stations, channel2, LocIds, GeneralFolder, FieldType, DateStart, DateEnd)


start_time = st[0].times('matplotlib')[0]

i_inicial = DateStart.timestamp
i         = DateStart.timestamp
i_final   = DateEnd.timestamp

#i_aux     = DateStart1.timestamp

windows   = 0
times     = [start_time]

while(i < i_final):
    t1 = i
    t2 = i + interval_rho_secs
    
    #t1_aux = i_aux
    #t2_aux = i_aux + interval_rho_secs
    
    print()    
    print('From', UTCDateTime(t1), 'to', UTCDateTime(t2))
    
    t=time.time()
    print("Loading data...")
    
    rho_a, T , finish_time, max_d, min_d, skin_depth = MTp.main_window(network, stations, channel1, channel2, LocIds, GeneralFolder, FieldType, UTCDateTime(t1), UTCDateTime(t2), interval_sumspec, New_Folder, overlap=overlap)
    
    # Definir la matriz RHO_a
    if(t1 == DateStart.timestamp):
        RHO_a = rho_a
        SKIN_depth = skin_depth
    else:
        RHO_a = np.concatenate((RHO_a, rho_a), axis=1)
        SKIN_depth = np.concatenate((SKIN_depth, skin_depth), axis=1)
        
        
    
    # Definir máxima y mínima profundidad
    if(i == i_inicial):
        max_depth = max_d
        min_depth = min_d
    else:
        if(max_d > max_depth):
            max_depth = max_d
        
        if(min_d < min_depth):
            min_depth = min_d
            
    secs=time.time()-t
    print("Data loaded and saved in %d seconds." % (secs))
    
    times.append(finish_time)
    windows = windows + 1
    
    i = t2
    
    #i_aux = t2_aux
    
    
times = np.array(times)

# Pasar de profundidad en m a km
max_depth = max_depth/1000
min_depth = min_depth/1000
SKIN_depth = SKIN_depth/1000


# Comprobar si existe el archivo, leer el catálogo sísmico
fname = '/home/juancho/Documentos/TrabajoGrado-Pregrado/Coding/Catalogo_Sismico(4.5).csv'
if os.path.isfile(fname):
    event_data = MTp.event_data(fname, stations, min_depth, max_depth)
else:
    event_data = []

# Comprobar si existe el archivo, leer dato del índice Kp
fname = '/home/juancho/Documentos/TrabajoGrado-Pregrado/Coding/kp2019.csv'
if os.path.isfile(fname):
    kp_data = MTp.kp_index(fname)
else:
    kp_data = None





# Mapa de anomalías de resistividad

RHO_a = 10**RHO_a

av = np.sum(RHO_a, axis=1)
av = av/RHO_a.shape[1]          # av = arreglo de promedio rho_a
av = av[:, np.newaxis]

# Crear la matriz de anomalias

delta_rho = np.zeros((RHO_a.shape[0], RHO_a.shape[1]))

for col in range(0, RHO_a.shape[1]):
    for row in range(0, RHO_a.shape[0]):
        delta_rho[row,col] = (np.abs(RHO_a[row,col] - av[row])/av[row])*100

# Save original delta_rho data
delta_rho_original = np.zeros((RHO_a.shape[0], RHO_a.shape[1]))
for col in range(0, RHO_a.shape[1]):
    for row in range(0, RHO_a.shape[0]):
        delta_rho_original[row,col] = delta_rho[row,col]

# Limit the scale to 300%
delta_rho[delta_rho > 300] = 300

RHO_a = np.log10(RHO_a)

MTp.rho_map(T, times, RHO_a, delta_rho, days, max_depth, min_depth, event_data, kp_data, New_Folder)

