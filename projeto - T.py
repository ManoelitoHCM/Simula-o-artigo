# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:50:02 2019

@author: Manoelito
"""
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#estequiometria da reaçao
#A + 3B <-> 3C + D
#m, k, Keq, K2, K4, u1, r, mod_thiele, n
#@m = 22.73, LHSV = 0.25 h^-1

def model(X, L, T):    
    d_cat = 1*10**(-3)                                                         #m    
    h_cat = 2*10**(-3)                                                         #m  
    Vp = ((math.pi*d_cat**2)/4)*h_cat                                          #m^3
    Sp = (math.pi*d_cat**2)/2 + math.pi*d_cat*h_cat                            #m^2    
    epsilon = 0.5192
    Ltotal = 0.6                                                               #m
    rho_b = 1109.0                                                             #kg/m^3
    D_1 = (T*2.26*10**(-10))/338                                               #m^2/s
    theta = 0.0713
    delta = 4.1
    Deff_1 = D_1*theta/delta                                                   #m^2/s
    A = math.exp(28.46759622)                                                  #m^6/kmol.kg.s
    EaR = 13180.06819237                                                       #K
    k = A*math.exp(-EaR/T)                                                     #m^6/kmol.kg.s
    Keq = (7.97E-13)*math.exp((0.0788)*T)    
    K2 = 703727.709*math.exp(-0.0420215266*T)                                  #m^3/kmol   
    K4 = 10336808100*math.exp(-0.0685249768*T)                                 #m^3/kmol    
    Cb_0 = 1                                                                   #kmol/m^3
    m = 22.73
    LHSV = 0.25/3600                                                           #s^-1
    u1 = LHSV*Ltotal/(1-epsilon)                                               #m/s
    r = k*((Cb_0**2)*(1-X)*(m-3*X) - (27*X**4)*(Cb_0**2)/(Keq*(m-3*X)**2))/(1 + K2*Cb_0*(m-3*X) + K4*Cb_0*X)
    mod_thiele = (Vp/Sp)*math.sqrt(abs(r*rho_b/(Cb_0*Deff_1)))
    n = (1/mod_thiele)*(1/math.tanh(3*mod_thiele) - 1/(3*mod_thiele))
    dXdL = r*n*rho_b/(u1*Cb_0)
    return dXdL
    
#initial condition
X0 = 0

#length points
L = np.linspace(0, 0.6)

#solve ODE
T = 347
X1 = odeint(model, X0, L, args=(T,))
T = 338
X2 = odeint(model, X0, L, args=(T,))
T = 328
X3 = odeint(model, X0, L, args=(T,))
T = 318
X4 = odeint(model, X0, L, args=(T,))
T = 298
X5 = odeint(model, X0, L, args=(T,))

#plot results
plt.plot(L, X1, 'r-', linewidth=2,label='T=347 K')
plt.plot(L, X2, 'b--', linewidth=2,label='T=338 K')
plt.plot(L, X3, 'g:', linewidth=2,label='T=328 K')
plt.plot(L, X4, 'y+', linewidth=2,label='T=318 K')
plt.plot(L, X5, 'c+-', linewidth=2,label='T=298 K')
plt.xlabel('Length (m)')
plt.ylabel('Conversion')
plt.legend()
plt.show()


#T (K)	k (m^6/kmol.kg.s)	K	K2 (m^3/kmol)	K4 (m^3/kmol)	Deff
#347	7.37146324E-05	5.97906007E-01	0.33	0.4871	4.03484529E-12
#343	4.73350949E-05	4.36257892E-01	0.39	0.6408	3.98833410E-12
#338	2.68112662E-05	2.94192279E-01	0.48	0.9026	3.93019512E-12
#328	8.16594159E-06	1.33784937E-01	0.73	1.7910	3.81391716E-12
#318	2.30793100E-06	6.08391541E-02	1.11	3.5539	3.69763920E-12
#298	1.42942907E-07	1.25815765E-02	2.56	13.9927	3.46508327E-12

#propriedades do palm oil (A ou 1)
#Cb_1 = Cb_0*(1-X)                                                             #mol/m^3

#propriedades do methanol (B ou 2)
#Cb_2 = Cb_0*(m-3*X)                                                           #mol/m^3

#propriedades do biodiesel/FAME (C ou 3)
#Cb_3 = 3*Cb_0*X                                                               #mol/m^3

#propriedades do glycerol (D ou 4)
#Cb_4 = Cb_0*X                                                                 #mol/m^3

#efeito de difusão externa neglicenciado
#o que limita é a difusão interna