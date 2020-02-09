# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:30:25 2019

@author: user
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

def f(t,u,v):
    return fc*v - A*np.cos(omega*t+phi) / rho

def g(t,u,v):
    return -fc*u

#%%
dt = 10 #time step
t= np.arange(0,5*24*3600,dt)        #time length
N=len(t)
A=1e-3
phi=0  #phase
omega = 2*np.pi/(24*3600)       #angular velocity of earth
lat = 52.457
fc = 2*omega*np.sin(lat)
rho = 1.225 #kg/m3
alpha = 6371*1000 #radius of earth

u = np.zeros(N)
v = np.zeros(N)

for n in range(N-1):
    
    k1=dt*f(t[n],u[n],v[n])
    l1=dt*g(t[n],u[n],v[n])
    k2=dt*f(t[n]+dt/2,u[n]+k1/2,v[n]+l1/2)
    l2=dt*g(t[n]+dt/2,u[n]+k1/2,v[n]+l1/2)
    k3=dt*f(t[n]+dt/2,u[n]+k2/2,v[n]+l2/2)
    l3=dt*g(t[n]+dt/2,u[n]+k2/2,v[n]+l2/2)
    k4=dt*f(t[n]+dt,u[n]+k3,v[n]+l3)
    l4=dt*g(t[n]+dt,u[n]+k3,v[n]+l3)
    
    u[n+1]=u[n]+1/6 *(k1+2*k2+2*k3+k4)
    v[n+1]=v[n]+1/6 *(l1+2*l2+2*l3+l4)
    
plt.figure()
plt.plot(t,u,label='u')
plt.plot(t,v,label='v')
plt.legend()

plt.figure()
plt.plot(u,v)






