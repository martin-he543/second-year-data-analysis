
# Author: Hiroki Kozuki

#%%

# exp(mu*d+k)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=9)

# u_restr_new = 



area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B


params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2.8,-2.45,11.2], sigma = unc_u_restricted, absolute_sigma = False) # -2.45
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
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]), np.sqrt(cov_params[3][3]))


#%%

# exp(mu*(d+k))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=9)

# deadtime = 1.1e-06
# u_restr_new = u_restricted/



area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*(d+k))/(4*np.pi) + B

params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2.8,-0.575,11.2], sigma = unc_u_restricted, absolute_sigma = False, maxfev=100000) # -0.575
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
plt.plot(x, params[0]*area*np.exp(-params[1]*(x+params[2]))/(4*np.pi) + params[3], ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]), np.sqrt(cov_params[3][3]))

#%%                            

# BEST VERSION WITH BEST UNCERTAINTIES.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=9)

# deadtime = 1.1e-06
# u_restr_new = u_restricted/


area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,B):                                                                                 
     return A*area*np.exp(-mu*d)/(4*np.pi) + B


params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[2000000,5,-69], sigma = unc_u_restricted, absolute_sigma = False, maxfev=100000) # 2.8
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
plt.plot(x, params[0]*area*np.exp(-params[1]*x)/(4*np.pi) + params[2], ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]))

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'CMU Serif'}                                                                             # Assign font parameters
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\task 17 activity.csv", delimiter=",", unpack=True, skiprows=8)

# deadtime = 1.1e-06
# u_restr_new = u_restricted/


area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,B):                                                                                 
     return A*area*np.exp(-mu*d)/(4*np.pi) + B*d**2


params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2,10], sigma = unc_u_restricted, absolute_sigma = False, maxfev=100000) # 2.8
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
plt.plot(x, params[0]*area*np.exp(-params[1]*x)/(4*np.pi) + params[2]*x, ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]))


#%%
# d, n, time = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=1)

# count_rate = n/time
# unc_count_rate = np.sqrt(n)/time

# def inverse_square_exponential(A,mu,k,B):                                                                                 
#     return A*np.exp(-mu*d_restricted+k)/(4*np.pi)-B


# plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
# plt.ylabel("count rate ($s^{-1}$)", **font)
# plt.grid()
# #plt.ylim(5,25)
# plt.xticks(**font, **fontAxesTicks)
# plt.yticks(**font, **fontAxesTicks)
# plt.title("Task 17: Count Rate ($s^{-1}$) vs. d (m)", **font)
# plt.errorbar(d, count_rate, yerr=unc_count_rate, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3)
# plt.show() 


#%%