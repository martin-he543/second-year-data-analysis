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



#%% INVERSE SQUARE FITTING
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'Kinnari'}
fontAxesTicks = {'size':7}

d, n, t, count_rate, val, err_val,rel_err_val = np.loadtxt("data/Task 16.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, n_restricted, t_restricted, count_rate_restricted, val_restricted, err_val_restricted,rel_err_val_restricted = np.loadtxt("data/Task 16.csv", delimiter=",", unpack=True, skiprows=5)

# deadtime = 1.1e-06
# true_count_rate = (n/t)/(1-(n/t)*deadtime)
# unc_true_count_rate = np.sqrt(n)/t
# print(true_count_rate)
# count_rate_restr = n_restricted/t_restricted
# unc_count_rate_restr = np.sqrt(n_restricted)/t_restricted

area,unc_area = 1.97e-4, 1.4e-5
def inverse_square(d,A,k,B):
     return A*area/(4*np.pi*(d+k)**2) + B

params_rate, cov_params_rate = curve_fit(inverse_square,d,count_rate,p0=[0.0144,1033000,-1.72], sigma = err_val, absolute_sigma = False, maxfev=100000) # -0.575

x = np.linspace(0,0.85,10000)
plt.xlabel("Separation d (m) [Source-detector]", **font)                                            # Label axes, add titles and error bars
plt.ylabel("Count Rate (($s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
#plt.ylim(-0.2e04, 0.05e06)
plt.title("Task 17: Count rate ($s^{-1}$) vs. d (m)", **font)
# plt.plot(x, (params_rate[0]*area)/(4*np.pi*(x+params_rate[1])**2) + params_rate[2], color = 'orange') # 11.8 , ls='-'
#plt.plot(x, exponential_decay(x, *params),'r')    
plt.errorbar(d, count_rate, yerr=err_val, xerr=0.003, ls='', mew=1, ms=0.5, capsize=3, color = 'blue') # Plots uncertainties in points          

plt.plot(x, (1033000*area)/(4*np.pi*(x+0.0144)**2) -1.72, color = 'orange') # 11.8 , ls='-'

plt.show()      

print(params_rate)
print("Activity",inverse_square(0,params_rate[0],params_rate[1],params_rate[2]))
print(np.sqrt(cov_params_rate[0][0]), np.sqrt(cov_params_rate[1][1]), np.sqrt(cov_params_rate[2][2]))

# %%
