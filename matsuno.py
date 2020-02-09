# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:17:47 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pylab as pl


def E_comp(s_mod,s_obs,theta):
    
    return np.sqrt( (s_mod)**2 + s_obs**2 -2* s_mod *s_obs* np.cos(theta))




#%% Model 1.14

dt = 10  #time step
t= np.arange(0,5*24*3600,dt)        #time length
N=len(t)
A=1e-3
phi=0  #phase
omega = 2*np.pi/(24*3600)       #angular velocity of earth
lat = 52.457
f = 2*omega*np.sin(lat)
rho = 1.225 #kg/m3
alpha = 6371*1000 #radius of earth



v=np.zeros(N)       #empty array for v velocity values
u=np.zeros(N)       #empty array for u velocity values
u[0] = 0            #initial value (boundary condition)
v[0] = 0             #initial value (boundary condition)
t[0] = 0             #initial value (boundary condition)







#%%Matsuno

v1=np.zeros(N)       #empty array for v velocity values
u1=np.zeros(N)       #empty array for u velocity values
u1[0] = 0            #initial value (boundary condition)
v1[0] = 0             #initial value (boundary condition)
t1[0] = 0             #initial value (boundary condition)
    
def du_dt(v,t):
    return f*v-A*np.cos(omega*t+phi)/rho
    
def dv_dt(u):
    return -f*u

#Predictor step
for n in range(N-1):
    u[n+1] = u[n] + dt* du_dt(v[n],t[n])
    v[n+1] = v[n] + dt * dv_dt(u[n])

#Corrector step
for n in range(N-1):
    u1[n+1] = u1[n] + dt* du_dt(v[n+1],t[n+1])
    v1[n+1] = v1[n] + dt * dv_dt(u[n+1])
    

#%% 
plt.figure()
plt.plot(t,v,label=v)
plt.plot(t,v1,label=v1)
plt.legend()


