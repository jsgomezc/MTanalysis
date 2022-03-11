# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:54:00 2020

@author: Juancho
"""
def read_signal(network, stations, channels, LocIds, GeneralFolder, FieldType, DateStart, DateEnd, interpolate=True):

    #import time
    #t=time.time()
    #print("Loading libraries...")
    
    from obspy.clients.filesystem.sds import Client
    #from obspy import UTCDateTime

    #secs=time.time()-t
    #print("Libraries loaded on %d seconds." % (secs))
    #print()
    
    #print("Loading data from ",DateStart," to ",DateEnd,"for: ")
    #print()
    #print("        Network: ",network)
    #print("       Stations: ",stations)
    #print("       Channels: ",channels)
    #print("         LocIds: ",LocIds)
    #print(" General Folder: ",GeneralFolder)
    #print()
    #ts=UTCDateTime.now()
    #print("---Data loading started at",ts)
    
    # Load Seiscomp Data Structure (SDS) for RSSB
    client_RSSB = Client(GeneralFolder)
    
    # Load Traces
    st = client_RSSB.get_waveforms(network, stations, LocIds, channels, DateStart, DateEnd)
    
    #delta=UTCDateTime.now()-ts
    #delta=UTCDateTime(delta)
    #print("---Time spent loading data: ",delta.strftime("%H:%M:%S"))
    #print()
    
    # if the user want to merge the traces
    if(interpolate):
        st.merge(method=1, fill_value='interpolate')
    
    st.filter('lowpass', freq=9.99)
    #print(st)
    
    # Electrode's distance for HQ* values
    if (stations == 'USME'):
        d_N = 97
        d_E = 98
    elif (stations == 'TUNJ'):
        d_N = 85
        d_E = 67
    elif (stations == 'VCIO'):
        d_N = 130
        d_E = 70
    
    # Data modified to real values of Amplitude
    for i in range (0,len(st)):
        tr_id = st[i].id
        if(tr_id[11] == 'H'):
            if(tr_id[12] == 'F'):
                original_data = st[i].data
                # print('Original data HF: ', original_data)
                # Conversión
                factorHF = (1+180/82)
                new_data_F = (((original_data-(2**15))*(20/(2**16))/factorHF)-1.65)/(0.00005) #nT
                # print('New data HF: ', new_data_F)
                
                st[i].data = new_data_F/1e9
            
            elif(tr_id[12] == 'Q'):
                original_data = st[i].data
                # print('Original data HQ: ', original_data)
                # Conversión
                tr = st[i]
                tr_new = tr.detrend(type='constant')
                factorHQ = 2/(2**16)
                new_data_Q = tr_new.data * factorHQ
                # print('New data HQ: ', new_data_Q)
                st[i].data = new_data_Q
                
                if(tr_id[13] == 'N'):
                    st[i].data = st[i].data/d_N
                elif(tr_id[13] == 'E'):
                    st[i].data = st[i].data/d_E
                    
    
                    
    return st


def amp_spec(data,dt):
   """
   Compute the power spectrum of a given signal
   Uses numpy
   Assumes data is a numpy array, column vector
   """
   import scipy.signal as sci
   import numpy as np

   ndim = data.ndim
   if ndim==1 :
      x = data 
      x = x[:,np.newaxis]
   
   # Detrend signal
   x = sci.detrend(x,axis=0)

   npts = x.shape[0]
   nfft = npts

   if np.all(np.isreal(data)):
      if (npts%2 == 0):
         nf = round(nfft/2.) + 1
         fvec_p  = np.arange(0,nfft/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft/2),0)/(nfft*dt)
      else:
         nf = round((nfft+1)/2)
         fvec_p  = np.arange(0,(nfft-1)/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft-1)/2,0)/(nfft*dt)
    
      # Create vector as in FFT sampling 
      # except with nyquist in positive freq
      # (matlab and Python numpy.fft)
      fvec = np.concatenate((fvec_p,fvec_n))
      freq = np.zeros((nf,1))
      spec = np.zeros((nf,1))

      fdata = np.fft.fft(x,nfft,0)
      #spec  = abs(fdata[0:nf])
      spec  = fdata[0:nf]
      freq  = fvec[0:nf]

   #print(npts, nfft, nf)
   #print('freq ', freq)

   return freq, spec, nf

def sum_spec(tr, intervalo, overlap=0.1):
    
    import numpy as np
    #import time
    
    # Inicio ciclo for que suma espectros
    #Dt = st[0].stats.endtime - st[0].stats.starttime

    win = int(intervalo)                            # Ventana de tiempo deseada segundos
    
    freq_muestreo = int(tr.stats.sampling_rate)     # Frecuencia de Muestreo de Datos
    i             = tr.stats.starttime.timestamp      # Inicio de la ventana completa en segundos
    i_final       = tr.stats.endtime.timestamp        # Final de la ventana completa en segundos
    
    cont = 0                                          # Número de espectros que se suman
    j    = 0                                          # index del arreglo de datos de la traza
    
    dt = tr.stats.delta
    # print(dt)
    
    tr_win = tr.data[j:j+(win*freq_muestreo)]
    npts = len(tr_win)
    
    if (npts%2 == 0):
        nf = round(npts/2.) + 1
    else:
        nf = round((npts+1)/2)
        
    sum_spec = np.zeros((nf,1))
    
    #t=time.time()
    #print("Processing freq data...")
    
    while i <= i_final:
        t1 = i
        t2 = t1 + win
        #print('Tiempo inicial',UTCDateTime(t1), 'hasta ',UTCDateTime(t2))
        #Procedimiento espectro
        #print('j inicial: ',j, 'hasta ', j+(win*freq_muestreo))
        tr_temp = tr
        tr_temp = tr.taper(0.01)
        tr_win = tr_temp.data[j:j+(win*freq_muestreo)]
        #print('Cantidad de datos ',len(tr_win))
    
        
        freq, spec, nf = amp_spec(tr_win, dt)
       
        
        if (spec.shape[0] == sum_spec.shape[0]):
            sum_spec = sum_spec + spec
            freq_final = freq
            #nf_final = nf
        '''
        else:
            temp = np.zeros((nf,1))
            for i in range(spec.shape[0]):
                temp[i][0] = spec[i][0]
            sum_spec = sum_spec + temp
        '''
        
        i = t2 - win*overlap
        cont = cont + 1
        j = j + (int(i - t1)*freq_muestreo)
        #print('j final: ',j)
        #print('Tiempo inicial nuevo ', UTCDateTime(i))
        #print(type(spec),spec.shape)
        #print('spec: ',spec)
        #print()
    
    #secs=time.time()-t
    #print("Data processed in %d seconds." % (secs))
    #print()
    #print('Final spec ', spec)
    #print('Spec sum ', sum_spec)
    #print('Shape: ', sum_spec.shape)
      
    #sum_spec = sum_spec/(cont-1)
    '''
    # Matriz con los datos obtenidos
    data_fs = np.zeros((nf_final,2), dtype=complex)
    for i in range(0,nf_final):
        data_fs[i][0] = freq_final[i]
        data_fs[i][1] = sum_spec[i]
    '''
    freq_final = freq_final[:, np.newaxis]
        
    return freq_final, sum_spec


def search_freq(interval, sample_rate):
    '''
    input:  
        window_lenght: length of the window chosen, must be in seconds
        sample_rate: Sample rate of the signal, must be in (Hz)
    
    output: 
        freq: list object with frequencies that should be used
    '''
    
    import numpy as np
    
    fmax = sample_rate/2
    ideal_freq = []
    #freq.append(fmax)
    
    fmin = 1 / interval
    
    i = 2
    f = fmax
    while(f > fmin):
        f = fmax * (1/(1.8**((i-1)/2)))
        ideal_freq.append(f)
        i = i+1
        
    ideal_freq = np.array(ideal_freq)
    ideal_freq = ideal_freq[:, np.newaxis]
    
    #print('Ideal frequencies: \n', ideal_freq)
        
    return ideal_freq



def rho_a(Z, freq):
    
    import numpy as np
    
    pi  = np.pi
    m_0 = 1.2566e-6
    
    rho_a     = np.zeros((len(Z),1))
    
    for i in range (0,len(Z)):
        if(freq[i] != 0):
            w            = 2*pi*freq[i]
            c_temp       = Z[i]/w
            rho_a[i]     = (np.abs(c_temp*np.conjugate(c_temp)))*m_0*w      
        
    # Pasar de frecuencia a Periodo
    T = np.zeros(len(freq))
    for i in range(0,len(freq)):
        if (freq[i] != 0):
            T[i] = 1/freq[i]
        
    T = T[:, np.newaxis]
    
    np.seterr(divide = 'ignore') 
    rho_a = np.log10(rho_a)
    
    return rho_a, T

def func(x, e, f):
    import numpy as np
    
    return e*np.log10(x) + f
    

def lsqr(rho_a, freq, ideal_freq):
    
    import numpy as np
    from scipy.optimize import curve_fit
    
    delta_freq = freq[1]-freq[0]
    
    rho_final  = np.zeros(len(ideal_freq))
    #perr       = np.zeros(len(ideal_freq))
    
    j_k        = np.zeros(len(ideal_freq))
    pos        = []
    
    
    for i in range(0, len(ideal_freq)):
        j_k[i]  = int(ideal_freq[i]/(2*delta_freq))
        m_k     = int(ideal_freq[i]/delta_freq)
        pos.append(m_k)
        obs = int(j_k[i])
        
        if(obs > 1):
            l_limit = int(m_k-obs)
            if((m_k+obs) >= len(rho_a)):
                u_limit = len(rho_a)
            else:
                u_limit = int(m_k+obs)
            
            temprho  = rho_a[l_limit:u_limit]
            tempfreq = freq[l_limit:u_limit]
            
            popt, pcov = curve_fit(func, tempfreq, temprho)
            #perr[i]    = np.sqrt(np.diag(pcov))                  # Standard deviation error
            
            rho_final[i] = func(freq[m_k], *popt)
            
        elif(obs <= 1):
            rho_final[i] = rho_a[m_k]
        
        
    freq = np.array(freq[pos])
    
    # Pasar de frecuencia a Periodo
    T = np.zeros(len(freq))
    for i in range(0,len(freq)):
        if (freq[i] != 0):
            T[i] = 1/freq[i]
            
    #rho_final = abs(rho_final)
            
    return rho_final, T

def plot(T, rho_a, T_lsqr, rho_a_lsqr, title, New_Folder):
    
    import matplotlib.pyplot as plt
     
    fig = plt.figure(figsize = (15,10))

    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    #ax1.loglog(T, rho_a, 'ko')
    ax1.semilogx(T, rho_a, 'ko', markersize=0.2)
    #ax1.loglog(T_lsqr, rho_a_lsqr, 'ro', label='Fitted curve')
    ax1.semilogx(T_lsqr, rho_a_lsqr, 'ro', label='Fitted curve')
    #ax1.errorbar(T, rho_a, yerr=delta_rho, capsize=5, fmt='ko')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylabel('$log_{10}(\\rho_a)$ $({\Omega}$ $m)$')
    ax1.set_xlabel('$Period$ $(s)$')
    ax1.set_ylim(-3, 5)
    ax1.legend(loc=4, fontsize=15, markerscale=2)
    #ax2.errorbar(T, phase, yerr=delta_p, capsize=5, fmt='o')
    #ax2.set_xscale('log')
    
    #fname = title.replace(':','.')
    #fname = New_Folder+'/'+fname+'.png'
    
    #plt.savefig(fname);
    #print('Image saved')
    
    plt.close()
    
    
def main_window(network, stations, channel1, channel2, LocIds, GeneralFolder, FieldType, DateStart, DateEnd, interval, New_Folder, overlap=0.0):
    
    import MTprocessing as MTp
    import numpy as np
    
    title = stations+' '+channel1+'-'+channel2+' '+str(DateStart)+' - '+str(DateEnd)+' (Intervalo '+str(interval)+'s)'

    st_Ex = MTp.read_signal(network, stations, channel1, LocIds, GeneralFolder, FieldType, DateStart, DateEnd)
    st_By = MTp.read_signal(network, stations, channel2, LocIds, GeneralFolder, FieldType, DateStart, DateEnd)
    
    sample_rate = st_By[0].stats.sampling_rate
    
    freq_Ex, spec_Ex = MTp.sum_spec(st_Ex[0], interval, overlap)
    freq_By, spec_By = MTp.sum_spec(st_By[0], interval, overlap)
    
    Freq = freq_Ex
    
    Ex   = spec_Ex
    By   = spec_By
    
    Z = Ex/By

    ideal_freq = MTp.search_freq(interval, sample_rate)
    
    rho_a, T   = MTp.rho_a(Z, Freq)
    
    rho_a = rho_a.flatten()
    Freq = Freq.flatten()
    
    rho_a_lsqr, T_lsqr = MTp.lsqr(rho_a, Freq, ideal_freq)
    
    rho_a      = rho_a[:, np.newaxis]
    rho_a_lsqr = rho_a_lsqr[:, np.newaxis]
    T_lsqr     = T_lsqr[:, np.newaxis]
    
    cut_data   = int(len(T_lsqr)*0.35*-1)
    
    rho_a_lsqr = rho_a_lsqr[:cut_data]
    T_lsqr     = T_lsqr[:cut_data]
    
    MTp.plot(T, rho_a, T_lsqr, rho_a_lsqr, title, New_Folder)

    skin_depth = 503*(((10**(rho_a_lsqr))*T_lsqr)**(1/2))
    
    max_depth  = max(skin_depth)
    min_depth  = min(skin_depth)

        

    finish_time = st_By[0].times('matplotlib')[-1]
    
    return rho_a_lsqr, T_lsqr, finish_time, max_depth, min_depth, skin_depth

def event_data(fname, stations, min_depth, max_depth):
    
    import numpy as np
    import matplotlib.dates as mpd
    from obspy.geodetics.base import gps2dist_azimuth
    
    cat = np.loadtxt(fname, dtype='str', delimiter=',', skiprows=1, usecols=0)

    for i in range(0, len(cat)):
        cat[i] = cat[i].replace(' ','T')
        
    cat = np.array(cat, dtype='datetime64[s]')
    
    plt_dates = mpd.date2num(cat)
    plt_dates = plt_dates[:, np.newaxis]
    
    depth = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=3)
    mag   = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=4)
    
    # Read location of each earthquake
    lat = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=1)
    lon = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=2)
    
    # Discard >max_depth km and <min_depth km depth earthquakes
    pos = []
    for i in range(0, len(depth)):
        # Change negative depths to 0.1 km
        if (depth[i] < 0):
            depth[i] = 0.1
        if(depth[i] < max_depth and depth[i] < max_depth):
            pos.append(i)
            
            
    depth     = depth[pos]
    plt_dates = plt_dates[pos]
    mag       = mag[pos]
    lat       = lat[pos]
    lon       = lon[pos]
    
    if(stations == 'USME'):
        # Station coordinates (USME)
        lat_0 = 4.480951
        lon_0 = -74.126777   
    elif(stations == 'TUNJ'):
        # Station coordinates (TUNJ)
        lat_0 = 5.533368
        lon_0 = -73.357760
    elif(stations == 'VCIO'):
        # Station coordinates (VCIO)
        lat_0 = 4.111264
        lon_0 = -73.592480
    else:
        raise ValueError("Station is not valid")
    
    dist  = []
    
    # Calculate distance from epicenter to station
    for i in range(0, len(lat)):
        d, azi, azi_2 = gps2dist_azimuth(lat_0, lon_0, lat[i], lon[i])
        d = d/1000      # Distance in km
        dist.append(d)
        
    dist = np.array(dist)

    
    # Concatenate data
    depth = depth[:, np.newaxis]
    mag   = mag[:, np.newaxis]
    dist  = dist[:, np.newaxis]

    data = np.concatenate((plt_dates, depth), axis=1)
    data = np.concatenate((data, mag), axis=1)
    data = np.concatenate((data, dist), axis=1)
    
    return data


def kp_index(fname):
    
    import numpy as np
    import matplotlib.dates as mpd
    
    day_date = np.loadtxt(fname, dtype='str', delimiter=',', skiprows=0, usecols=(0,1,2))

    kp_total = np.loadtxt(fname, dtype='float', delimiter=',', skiprows=0, usecols=(3,4,5,6,7,8,9,10))
    
    dates = []
    kp    = []
    
    for i in range (0, len(day_date)):
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T01:30:00')
        kp.append(kp_total[i,0])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T04:30:00')
        kp.append(kp_total[i,1])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T07:30:00')
        kp.append(kp_total[i,2])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T10:30:00')
        kp.append(kp_total[i,3])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T13:30:00')
        kp.append(kp_total[i,4])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T16:30:00')
        kp.append(kp_total[i,5])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T19:30:00')
        kp.append(kp_total[i,6])
        
        dates.append(day_date[i,0]+'-'+day_date[i,1]+'-'+day_date[i,2]+'T22:30:00')
        kp.append(kp_total[i,7])
    
    dates = np.array(dates, dtype='datetime64[s]')
    kp    = np.array(kp)
        
    
    plt_dates = mpd.date2num(dates)
    
    plt_dates = plt_dates[:, np.newaxis]
    kp        = kp[:, np.newaxis]
    
    
    kp_data   = np.concatenate((plt_dates, kp), axis=1)
    
    return kp_data


    
def rho_map(T, times, RHO_a, delta_rho, days, max_depth, min_depth, event_data, kp_data, New_Folder):
    
    import matplotlib.pyplot as plt
    #import matplotlib.colors as colors
    import numpy as np
    from matplotlib.dates import DateFormatter
    from matplotlib.patches import Patch
    
    # Method pcolormesh or contourf   
    x = times
    y = T[:,0]
    add = '(pcolormesh)'
    
        
    # Figure's xsize depends on the days to be plotted
    xsize = 6.4*days
    
    #cmap = plt.get_cmap('rainbow')
    
    fig = plt.figure(figsize=(xsize,15))
    ax = fig.add_subplot(211)
    fig.suptitle(New_Folder+'                 ', fontsize=30)
    
    
    #pcm1 = ax.pcolormesh(x, y, RHO_a,
     #                  norm=colors.LogNorm(vmin=RHO_a.min(), vmax=RHO_a.max()),
      #                  cmap='rainbow')
    pcm1 = ax.pcolormesh(x, y, RHO_a, cmap='rainbow', vmin=-2, vmax=4)

    #ax.xaxis.tick_top()
    ax2 = ax.twinx()
    
    if(len(event_data) != 0):
        # Plot event data
        
        # Rectify dates
        pos = []
        for i in range(0, len(event_data)):
            if(event_data[i,0] > x[0] and event_data[i,0] < x[-1]):
                pos.append(i)
        event_data = event_data[pos]
    
        # Increasing difference in Magnitude
        s = [1*3**n for n in event_data[:,2]]
        
        
        if(len(event_data) <= 4 and len(event_data) > 0):
        
        
            scatter = ax2.scatter(event_data[:,0], event_data[:,1], s=s, c=event_data[:,3], edgecolor='black', vmin=0, vmax=500, cmap='hot')
            # produce a legend with the unique colors from the scatter
            legend1 = ax2.legend(*scatter.legend_elements(),
                                loc="center left", title="Distance (km)")
            ax2.add_artist(legend1)
            
            # produce a legend with a cross section of sizes from the scatter
            # If you want a color in the color-distance scale: color=scatter.cmap(0.5)
            kw = dict(prop="sizes", color='black',
                      func=lambda s: np.log(s)/np.log(3))
            legend2 = ax2.legend(*scatter.legend_elements(**kw),
                                loc="lower left", title="Mag.", ncol=5)
            
        elif(len(event_data) > 4):
            
            
            scatter = ax2.scatter(event_data[:,0], event_data[:,1], s=s, c=event_data[:,3], edgecolor='black', vmin=0, vmax=500, cmap='hot')
            # produce a legend with the unique colors from the scatter
            legend1 = ax2.legend(*scatter.legend_elements(num = 4),
                                loc="center left", title="Distance (km)")
            ax2.add_artist(legend1)
            
            # produce a legend with a cross section of sizes from the scatter
            # If you want a color in the color-distance scale: color=scatter.cmap(0.5)
            kw = dict(prop="sizes", num = 4, color='black',
                      func=lambda s: np.log(s)/np.log(2.5))
            legend2 = ax2.legend(*scatter.legend_elements(**kw),
                                loc="lower left", title="Mag.", ncol=5)
        
        
    ax2.spines['left'].set_position(('outward', 70))
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')       
    # no x-ticks                 
    ax2.xaxis.set_ticks([])
    ax2.set_yscale('log')
    ax2.set_ylim(max_depth, min_depth)
    ax2.set_ylabel('$Depth$ $(km)$', fontsize=20)
    ax2.yaxis.set_tick_params(labelsize=15)
    
    ax.xaxis_date()
    ax.set_ylim(max(T), min(T))
    ax.set_yscale('log')
    ax.set_ylabel('$Period$ $(T)$ $(s)$', fontsize=20)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel('$Time$')
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, step=0.250))
    ax.set_title('$Apparent$ $Resistivity$ $map$\n', fontsize=20)
    formatter = DateFormatter('%y-%m-%d %H')
    ax.xaxis.set_major_formatter(formatter)
    
    # Anomaly map
    ax3 = fig.add_subplot(212)
    ax3.set_title('$Anomaly$ $map$\n', fontsize=20)
    

    pcm2 = ax3.pcolormesh(x, y, delta_rho, cmap='Greys', vmin=0, vmax=300)
    ax3.yaxis.set_tick_params(labelsize=15)
    
    
    ax4 = ax3.twinx()

    if(len(event_data) <= 4 and len(event_data) > 0):
        
        
        scatter = ax4.scatter(event_data[:,0], event_data[:,1], s=s, c=event_data[:,3], edgecolor='black', vmin=0, vmax=500, cmap='hot')
        # produce a legend with the unique colors from the scatter
        legend1 = ax4.legend(*scatter.legend_elements(),
                            loc="center left", title="Distance (km)")
        ax4.add_artist(legend1)
        
        # produce a legend with a cross section of sizes from the scatter
        # If you want a color in the color-distance scale: color=scatter.cmap(0.5)
        kw = dict(prop="sizes", color='black',
                  func=lambda s: np.log(s)/np.log(3))
        legend2 = ax4.legend(*scatter.legend_elements(**kw),
                            loc="lower left", title="Mag.", ncol=5)
        
    elif(len(event_data) > 4):
        
        
        scatter = ax4.scatter(event_data[:,0], event_data[:,1], s=s, c=event_data[:,3], edgecolor='black', vmin=0, vmax=500, cmap='hot')
        # produce a legend with the unique colors from the scatter
        legend1 = ax4.legend(*scatter.legend_elements(num = 4),
                            loc="center left", title="Distance (km)")
        ax4.add_artist(legend1)
        
        # produce a legend with a cross section of sizes from the scatter
        # If you want a color in the color-distance scale: color=scatter.cmap(0.5)
        kw = dict(prop="sizes", num = 4, color='black',
                  func=lambda s: np.log(s)/np.log(2.5))
        legend2 = ax4.legend(*scatter.legend_elements(**kw),
                            loc="lower left", title="Mag.", ncol=5)
        
    ax4.spines['left'].set_position(('outward', 70))
    ax4.spines["left"].set_visible(True)
    ax4.yaxis.set_label_position('left')
    ax4.yaxis.set_ticks_position('left')       
    # no x-ticks                 
    ax4.xaxis.set_ticks([])
    ax4.set_yscale('log')
    ax4.set_ylim(max_depth, min_depth)
    ax4.set_ylabel('$Depth$ $(km)$', fontsize=20)
    ax4.yaxis.set_tick_params(labelsize=15)
        
    ax3.xaxis_date()
    ax3.set_ylim(max(T), min(T))
    ax3.set_yscale('log')
    ax3.set_ylabel('$Period$ $(T)$ $(s)$', fontsize=20)
    ax3.set_xlabel('$Time$', fontsize=20)
    start, end = ax3.get_xlim()
    ax3.xaxis.set_ticks(np.arange(start, end, step=0.250))
    formatter = DateFormatter('%y-%m-%d %H')
    ax3.xaxis.set_major_formatter(formatter)
    ax3.xaxis.set_tick_params(labelsize=15)
    
    
    
    # Plot kp index if the file exists
    if kp_data is not None:
        #ax5 = fig.add_subplot(313)
        ax5 = ax.twinx()
           
        # Rectify dates
        pos = []
        for i in range(0, len(kp_data)):
            if(kp_data[i,0] > x[0] and kp_data[i,0] < x[-1]):
                pos.append(i)
        kp_data = kp_data[pos]
        
        for kp in kp_data:
            if(kp[1] < 4):
                ax5.bar(kp[0], kp[1], width=0.1, align='center', color='green', edgecolor='black', alpha=0.5)
            elif(kp[1] == 4):
                ax5.bar(kp[0], kp[1], width=0.1, align='center', color='yellow', edgecolor='black', alpha=0.5)
            elif(kp[1] < 4):
                ax5.bar(kp[0], kp[1], width=0.1, align='center', color='red', edgecolor='black', alpha=0.5)
        
        legend_elements = [Patch(facecolor='green', edgecolor='k', label='Low activity', alpha=0.5),
                           Patch(facecolor='yellow', edgecolor='k', label='Medium activity', alpha=0.5),
                           Patch(facecolor='red', edgecolor='k', label='High activity', alpha=0.5)]
        
        ax5.legend(handles=legend_elements, loc="lower right", title='Kp index')
        ax5.set_ylim(9, 0)
        ax5.xaxis.tick_top()
        ax5.spines['left'].set_position(('outward', 150))
        ax5.spines["left"].set_visible(True)
        ax5.yaxis.set_label_position('left')
        ax5.yaxis.set_ticks_position('left')       
        # no x-ticks                 
        ax5.xaxis.set_ticks([])
        
        ax5.set_ylabel('$K_p$ $index$', fontsize=20)
        ax5.yaxis.set_tick_params(labelsize=15)
    
    '''
    ax5.xaxis_date()
    ax5.set_ylim(7, 0)
    ax.xaxis.tick_top()
    ax5.set_ylabel('$K_p$ $index$')
    ax5.set_xlabel('$Time$')
    start, end = ax5.get_xlim()
    ax5.xaxis.set_ticks(np.arange(start, end, step=0.250))
    formatter = DateFormatter('%y-%m-%d %H')
    ax5.xaxis.set_major_formatter(formatter)
    '''
    
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.3)
    
    #ax3 = fig.add_subplot(122)
    cb1 = fig.colorbar(pcm1, ax=ax, pad=0.02)
    cb2 = fig.colorbar(pcm2, ax=ax3, pad=0.02)

    cb1.set_label(label='$log_{10}(\\rho_a)$ $({\Omega}$ $m)$', size=15)
    cb1.ax.tick_params(labelsize=15)
    cb2.set_label(label='${\Delta}{\\rho_a}$ $/$ ${\\bar{\\rho_a}}$ $({\%})$', size=15)
    cb2.ax.tick_params(labelsize=15)
             
    #pcm = ax[1].pcolor(x, y, RHO_a, cmap='PuBu_r')
    #fig.colorbar(pcm, ax=ax[1], extend='max')
    #fname = New_Folder+'/Final Image '+add+'.png'
    #plt.savefig(fname)
    print('Final Image saved')
    plt.show()