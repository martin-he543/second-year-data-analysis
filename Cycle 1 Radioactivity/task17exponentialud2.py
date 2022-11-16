#%% EXPONENTIAL THEORETICAL FITTING
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'fontname':'Kinnari'}
fontAxesTicks = {'size':7}

d, u, unc_u = np.loadtxt("Task 17.csv", delimiter=",", unpack=True, skiprows=1)
d_restricted, u_restricted, unc_u_restricted = np.loadtxt("Task 17.csv", delimiter=",", unpack=True, skiprows=5)

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B


# params, cov_params = curve_fit(exponential_decay,d_restricted,u_restricted,p0=[5555555,2.8,-2.45,11.2])
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. d (m)", **font)
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', mew=0.6, ms=0.5, capsize=3) # Plots uncertainties in points
# plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-') # 11.8
#plt.plot(x, exponential_decay(x, *params),'r')              
plt.show()      
# plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)

print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]))
print(type(u_restricted))

# %%
