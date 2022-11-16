#%% EXPONENTIAL THEORETICAL FITTING
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt("task 17 activity.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt("task 17 activity.csv", delimiter=",", unpack=True, skiprows=5)

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B


params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2.8,-2.45,11.2])
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]))
print(type(u_restricted))


#%% EXPONENTIAL THEORETICAL FITTING (MARTIN'S VERSION)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'Kinnari'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt("Task 17.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt("Task 17.csv", delimiter=",", unpack=True, skiprows=5)

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B


params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2.8,-2.45,11.2])
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]))
print(type(u_restricted))




#%% INVERSE SQUARE FITTING
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, n, t, unc_n = np.loadtxt("count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, n_restricted, t_restricted, unc_n_restricted = np.loadtxt("count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=9)

deadtime = 1.1e-06
area,unc_area = 1.97e-4, 1.4e-5
def inverse_square(d,A,k,B):                                                                                 
     return A*area/(4*np.pi*(d+k)**2) + B

params_rate, cov_params_rate = curve_fit(inverse_square,d,true_count_rate,p0=[1400000,0.007,-100], sigma = unc_true_count_rate, absolute_sigma = False, maxfev=100000) # -0.575
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("Count Rate ($s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
#plt.ylim(-0.2e04, 0.05e06)
plt.title("Task 17: Count rate $s^{-1}$) vs. d (m)", **font)
plt.plot(x, (params_rate[0]*area)/(4*np.pi*(x+params_rate[1])**2) + params_rate[2], color = 'orange') # 11.8 , ls='-'
#plt.plot(x, exponential_decay(x, *params),'r')    
plt.errorbar(d, true_count_rate, yerr=unc_true_count_rate, xerr=0.003, ls='', mew=1, ms=0.5, capsize=3, color = 'blue') # Plots uncertainties in points          
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params_rate)
print("Activity",inverse_square(0,params_rate[0],params_rate[1],params_rate[2]))
print(np.sqrt(cov_params_rate[0][0]), np.sqrt(cov_params_rate[1][1]), np.sqrt(cov_params_rate[2][2]))


#%% INVERSE SQUARE FITTING (MARTIN'S VERSION)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'Kinnari'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, n, t, unc_n = np.loadtxt("count rate vs. distance martin.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, n_restricted, t_restricted, unc_n_restricted = np.loadtxt("count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=9)

deadtime = 1.1e-06

true_count_rate = (n/t)/(1-(n/t)*deadtime)
unc_true_count_rate = np.sqrt(n)/t

# count_rate_restr = n_restricted/t_restricted
# unc_count_rate_restr = np.sqrt(n_restricted)/t_restricted

area,unc_area = 1.97e-4, 1.4e-5
def inverse_square(d,A,k,B):                                                                                 
     return A*area/(4*np.pi*(d+k)**2) + B

params_rate, cov_params_rate = curve_fit(inverse_square,d,true_count_rate,p0=[1400000,0.007,-100], sigma = unc_true_count_rate, absolute_sigma = False, maxfev=100000) # -0.575
x = np.linspace(0,0.85,10000)
plt.xlabel("Separation, d (m) [Source Detector]", **font)                                            # Label axes, add titles and error bars
plt.ylabel("Count Rate ($s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
#plt.ylim(-0.2e04, 0.05e06)
plt.title("Task 17: n ($s^{-1}$) vs. d (m)", **font)
plt.plot(x, (params_rate[0]*area)/(4*np.pi*(x+params_rate[1])**2) + params_rate[2], color = 'orange', label="Inverse Square Curve") # 11.8 , ls='-'
#plt.plot(x, exponential_decay(x, *params),'r')
plt.legend()
plt.errorbar(d, true_count_rate, yerr=unc_true_count_rate, xerr=0.003, ls='', mew=1, ms=0.5, capsize=3, color = 'blue') # Plots uncertainties in points          
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params_rate)
print("Activity",inverse_square(0,params_rate[0],params_rate[1],params_rate[2]))
print(np.sqrt(cov_params_rate[0][0]), np.sqrt(cov_params_rate[1][1]), np.sqrt(cov_params_rate[2][2]))

# %%
