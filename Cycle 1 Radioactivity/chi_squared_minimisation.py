#%% TASK 19

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import uncertainties as unc
import itertools as iter
from scipy.optimize import curve_fit
from scipy import integrate

# Font Formatting Styles
titleFont = {'fontname': 'Kinnari', 'size': 13}
axesFont = {'fontname': 'Kinnari', 'size': 9}
ticksFont = {'fontname': 'SF Mono', 'size': 7}
errorStyle = {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle = {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle = {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle = {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}

x19,count19,time,y19,countErr19,yErr19 = \
    np.loadtxt("Task 19 Al.csv",unpack=True,delimiter=",",skiprows=1)

plt.plot(x19,y19,'x')
plt.errorbar(x=x19,xerr=0.01,y=y19,yerr=yErr19, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("ln (Count)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Range of Decay Particles through Al", **titleFont)
# plt.savefig('Task 16.jpg', dpi=1000)

poly19A,cov_poly19A = np.polyfit(x19[:6],y19[:6],1,cov=True)
x= np.linspace(0,3.5,1000)
plt.plot(x,poly19A[0]*x + poly19A[1],label="2.27MeV β-,0.51MeV β-,\n0.54MeV β-,γ Rays")
poly19B,cov_poly19B = np.polyfit(x19[9:17],y19[9:17],1,cov=True)
x= np.linspace(1,6,1000)
plt.plot(x,poly19B[0]*x + poly19B[1],color="red",label="0.54MeV β-,γ Rays")
poly19C,cov_poly19C = np.polyfit(x19[18:],y19[18:],1,cov=True)
x= np.linspace(3,6,1000)
plt.plot(x,poly19C[0]*x + poly19C[1],color="green",label="γ Rays")

x_intersection = -(poly19B[1] - poly19A[1])/(poly19B[0] - poly19A[0])
print("first intersection: [",x_intersection,poly19A[0]*x_intersection+poly19A[1],"]")
x_intersection = -(poly19C[1] - poly19B[1])/(poly19C[0] - poly19B[0])
print("second intersection: [",x_intersection,poly19B[0]*x_intersection+poly19B[1],"]")

plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()

print("Linear Fit Parameters")
print("m₀ = ",poly19A[0],"+/-",np.sqrt(cov_poly19A[0][0]))
print("c₀ = ",poly19A[1],"+/-",np.sqrt(cov_poly19A[1][1]))
print("m₁ = ",poly19B[0],"+/-",np.sqrt(cov_poly19B[0][0]))
print("c₁ = ",poly19B[1],"+/-",np.sqrt(cov_poly19B[1][1]))
print("m₂ = ",poly19C[0],"+/-",np.sqrt(cov_poly19C[0][0]))
print("c₂ = ",poly19C[1],"+/-",np.sqrt(cov_poly19C[1][1]))

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B
params, cov_params = curve_fit(exponential_decay,x19,count19, p0=[555000,1.6,10,10])

x=np.linspace(0,5.2,10000)
plt.plot(x19,count19,'x')
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-')
plt.errorbar(x=x19,xerr=0.01,y=count19,yerr=countErr19, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("Count, u / s¯¹", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Range of Decay Particles through Al", **titleFont)

print("\nExponential Fit Parameters")
print("A = ",params[0],"+/-",np.sqrt(cov_params[0][0]))
print("µ = ",params[1],"+/-",np.sqrt(cov_params[1][1]))
print("k = ",params[2],"+/-",np.sqrt(cov_params[2][2]))
print("B = ",params[3],"+/-",np.sqrt(cov_params[3][3]))

#%% CHI SQUARED SECTION

combinations = [[0,4,13,21],[0,5,13,21],[0,6,13,21],[0,7,13,21],[0,8,13,21],[0,9,13,21],
                [0,4,14,21],[0,5,14,21],[0,6,14,21],[0,7,14,21],[0,8,14,21],[0,9,14,21],
                [0,4,15,21],[0,5,15,21],[0,6,15,21],[0,7,15,21],[0,8,15,21],[0,9,15,21],
                [0,4,16,21],[0,5,16,21],[0,6,16,21],[0,7,16,21],[0,8,16,21],[0,9,16,21],
                [0,4,17,21],[0,5,17,21],[0,6,17,21],[0,7,17,21],[0,8,17,21],[0,9,17,21],
                [0,4,18,21],[0,5,18,21],[0,6,18,21],[0,7,18,21],[0,8,18,21],[0,9,18,21]
                ]
# for i in range(1,len(x19)-1):
#     for j in range(i+1,len(x19)-1):
#         print([0,i,j,len(x19)-1])

chi_squared_values = []
red_chi_squared_values = []
intersection_values = []

for i in range(len(combinations)-1):
    poly19A,cov_poly19A = np.polyfit(x19[:combinations[i][1]],y19[:combinations[i][1]],1,cov=True)
    x= np.linspace(0,3.5,1000)
    plt.plot(x,poly19A[0]*x + poly19A[1],color="orange")
    poly19B,cov_poly19B = np.polyfit(x19[combinations[i][1]:combinations[i][2]],y19[combinations[i][1]:combinations[i][2]],1,cov=True)
    x= np.linspace(1,4.5,1000)
    plt.plot(x,poly19B[0]*x + poly19B[1],color="red")
    poly19C,cov_poly19C = np.polyfit(x19[combinations[i][2]:],y19[combinations[i][2]:],1,cov=True)
    x= np.linspace(3,6,1000)
    plt.plot(x,poly19C[0]*x + poly19C[1],color="green")
    
    estimated_values = []
    for k in range(0,combinations[i][1]+1):
        estimated_values.append(poly19A[0]*x19[k] + poly19A[1])
    for l in range(combinations[i][1]+1,combinations[i][2]+1):
        estimated_values.append(poly19B[0]*x19[l] + poly19B[1])
    for m in range(combinations[i][2]+1,combinations[i][3]):
        estimated_values.append(poly19C[0]*x19[m] + poly19C[1])
        
    chi_squared = 0
    red_chi_squared = 0
    for j in range(len(x19) - 1):
        chi_squared += (x19[j] - estimated_values[j])**2/estimated_values[j]
    red_chi_squared = chi_squared / (len(x19) - 9)    
    print(combinations[i],": χ² = ",chi_squared,", red χ² = ",red_chi_squared)
    chi_squared_values.append(chi_squared)
    red_chi_squared_values.append(red_chi_squared)
    
    x_intersection = -(poly19B[1] - poly19A[1])/(poly19B[0] - poly19A[0])
    intersection_values.append([x_intersection,poly19A[0]*x_intersection+poly19A[1]])
    x_intersection = -(poly19C[1] - poly19B[1])/(poly19C[0] - poly19B[0])
    intersection_values.append([x_intersection,poly19B[0]*x_intersection+poly19B[1]])
    

i = len(combinations) - 1
poly19A,cov_poly19A = np.polyfit(x19[:combinations[i][1]],y19[:combinations[i][1]],1,cov=True)
x= np.linspace(0,3.5,1000)
plt.plot(x,poly19A[0]*x + poly19A[1],color="orange",label="2.27MeV β-,0.51MeV β-,\n0.54MeV β-,γ Rays")
poly19B,cov_poly19B = np.polyfit(x19[combinations[i][1]:combinations[i][2]],y19[combinations[i][1]:combinations[i][2]],1,cov=True)
x= np.linspace(1,4.5,1000)
plt.plot(x,poly19B[0]*x + poly19B[1],color="red",label="0.54MeV β-,γ Rays")
poly19C,cov_poly19C = np.polyfit(x19[combinations[i][2]:],y19[combinations[i][2]:],1,cov=True)
x= np.linspace(3,6,1000)
plt.plot(x,poly19C[0]*x + poly19C[1],color="green",label="γ Rays")#

estimated_values = []
for k in range(0,combinations[i][1]+1):
    estimated_values.append(poly19A[0]*x19[k] + poly19A[1])
for l in range(combinations[i][1]+1,combinations[i][2]+1):
    estimated_values.append(poly19B[0]*x19[l] + poly19B[1])
for m in range(combinations[i][2]+1,combinations[i][3]):
    estimated_values.append(poly19C[0]*x19[m] + poly19C[1])
chi_squared = 0
red_chi_squared = 0
for j in range(len(x19) - 1):
    chi_squared += (x19[j] - estimated_values[j])**2/estimated_values[j]
red_chi_squared = chi_squared / (len(x19) - 9)
print(combinations[i],": χ² = ",chi_squared,", red χ² = ",red_chi_squared)
chi_squared_values.append(chi_squared)
red_chi_squared_values.append(red_chi_squared)
x_intersection = -(poly19B[1] - poly19A[1])/(poly19B[0] - poly19A[0])
intersection_values.append([x_intersection,poly19A[0]*x_intersection+poly19A[1]])
x_intersection = -(poly19C[1] - poly19B[1])/(poly19C[0] - poly19B[0])
intersection_values.append([x_intersection,poly19B[0]*x_intersection+poly19B[1]])

plt.plot(x19,y19,'x')
plt.errorbar(x=x19,xerr=0.01,y=y19,yerr=yErr19, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("ln (Count)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Range of Decay Particles through Al", **titleFont)
plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()


plt.plot(combinations,chi_squared_values,'x')
plt.xlabel("4 Distinct Linear Regression Ranges [0,x,y,21]", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Chi Squared Parametrisation", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()

plt.plot(combinations,red_chi_squared_values,'x')
plt.xlabel("4 Distinct Linear Regression Ranges [0,x,y,21]", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Reduced Chi Squared Parametrisation", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()


plt.axvline(4.17,linestyle="dotted")
plt.axvline(10.7,linestyle="dotted")
plt.plot(intersection_values[1::2],chi_squared_values,'x')
plt.plot(intersection_values[0::2],chi_squared_values,'x')
plt.xlabel("ln(Count) Value at Intersection", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Chi Squared Intersection Values", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()

maximum_energy = []
for n in range(len(intersection_values)):
    maximum_energy.append(intersection_values[n][0]*2.7)
    
print(maximum_energy)

 
# %% TASK 20
x20,count20,time,y20,count20Err,yErr20 = \
    np.loadtxt("Task 20 Cu.csv",unpack=True,delimiter=",",skiprows=1)

plt.plot(x20,y20,'x')
plt.errorbar(x=x20,xerr=0.01,y=y20,yerr=yErr20, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("ln (Count)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Range of Decay Particles through Cu", **titleFont)
# plt.savefig('Task 16.jpg', dpi=1000)

poly20A,cov_poly20A = np.polyfit(x20[:8],y20[:8],1,cov=True)
x= np.linspace(0,0.8,1000)
plt.plot(x,poly20A[0]*x + poly20A[1])
poly20B,cov_poly20B = np.polyfit(x20[5:12],y20[5:12],1,cov=True)
x= np.linspace(0.3,1.1,1000)
plt.plot(x,poly20B[0]*x + poly20B[1],color="red")
poly20C,cov_poly20C = np.polyfit(x20[14:],y20[14:],1,cov=True)
x= np.linspace(0.8,1.6,1000)
plt.plot(x,poly20C[0]*x + poly20C[1],color="green")

x_intersection = -(poly20B[1] - poly20A[1])/(poly20B[0] - poly20A[0])
print("first intersection",x_intersection,poly20A[0]*x_intersection+poly20A[1])
x_intersection = -(poly20C[1] - poly20B[1])/(poly20C[0] - poly20B[0])
print("second intersection",x_intersection,poly20B[0]*x_intersection+poly20B[1])

plt.show()

print("Linear Fit Parameters")
print("m₀ = ",poly20A[0],"+/-",np.sqrt(cov_poly20A[0][0]))
print("c₀ = ",poly20A[1],"+/-",np.sqrt(cov_poly20A[1][1]))
print("m₁ = ",poly20B[0],"+/-",np.sqrt(cov_poly20B[0][0]))
print("c₁ = ",poly20B[1],"+/-",np.sqrt(cov_poly20B[1][1]))
print("m₂ = ",poly20C[0],"+/-",np.sqrt(cov_poly20C[0][0]))
print("c₂ = ",poly20C[1],"+/-",np.sqrt(cov_poly20C[1][1]))

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B
params, cov_params = curve_fit(exponential_decay,x20,count20)

x=np.linspace(0,1.55,10000)
plt.plot(x20,count20,'x')
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-')
plt.errorbar(x=x20,xerr=0.01,y=count20,yerr=count20Err, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("Count, u / s¯¹", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Range of Decay Particles through Cu", **titleFont)

print("\nExponential Fit Parameters")
print("A = ",params[0],"+/-",np.sqrt(cov_params[0][0]))
print("µ = ",params[1],"+/-",np.sqrt(cov_params[1][1]))
print("k = ",params[2],"+/-",np.sqrt(cov_params[2][2]))
print("B = ",params[3],"+/-",np.sqrt(cov_params[3][3]))

#%% CHI SQUARED SECTION

combinations = [[0,2,8,18],[0,3,8,18],[0,4,8,18],[0,5,8,18],[0,6,8,18],
                [0,2,9,18],[0,3,9,18],[0,4,9,18],[0,5,9,18],[0,6,9,18],
                [0,2,10,18],[0,3,10,18],[0,4,10,18],[0,5,10,18],[0,6,10,18],
                [0,2,11,18],[0,3,11,18],[0,4,11,18],[0,5,11,18],[0,6,11,18],
                [0,2,12,18],[0,3,12,18],[0,4,12,18],[0,5,12,18],[0,6,12,18],
                [0,2,13,18],[0,3,13,18],[0,4,13,18],[0,5,13,18],[0,6,13,18],
                [0,2,14,18],[0,3,14,18],[0,4,14,18],[0,5,14,18],[0,6,14,18],
                [0,2,15,18],[0,3,15,18],[0,4,15,18],[0,5,15,18],[0,6,15,18],
                ]


chi_squared_values = []
red_chi_squared_values = []
intersection_values = []

for i in range(len(combinations)-1):
    poly20A = np.polyfit(x20[:combinations[i][1]],y20[:combinations[i][1]],1)
    x= np.linspace(0,0.9,1000)
    plt.plot(x,poly20A[0]*x + poly20A[1],color="orange")
    poly20B= np.polyfit(x20[combinations[i][1]:combinations[i][2]],y20[combinations[i][1]:combinations[i][2]],1)
    x= np.linspace(0.3,1.1,1000)
    plt.plot(x,poly20B[0]*x + poly20B[1],color="red")
    poly20C = np.polyfit(x20[combinations[i][2]:],y20[combinations[i][2]:],1)
    x= np.linspace(0.8,1.6,1000)
    plt.plot(x,poly20C[0]*x + poly20C[1],color="green")
    
    estimated_values = []
    for k in range(0,combinations[i][1]+1):
        estimated_values.append(poly20A[0]*x20[k] + poly20A[1])
    for l in range(combinations[i][1]+1,combinations[i][2]+1):
        estimated_values.append(poly20B[0]*x20[l] + poly20B[1])
    for m in range(combinations[i][2]+1,combinations[i][3]):
        estimated_values.append(poly20C[0]*x20[m] + poly20C[1])
        
    chi_squared = 0
    red_chi_squared = 0
    for j in range(len(x20) - 1):
        chi_squared += (x20[j] - estimated_values[j])**2/estimated_values[j]
    red_chi_squared = chi_squared / (len(x20) - 9)    
    print(combinations[i],": χ² = ",chi_squared,", red χ² = ",red_chi_squared)
    chi_squared_values.append(chi_squared)
    red_chi_squared_values.append(red_chi_squared)
    
    x_intersection = -(poly20B[1] - poly20A[1])/(poly20B[0] - poly20A[0])
    intersection_values.append([x_intersection,poly20A[0]*x_intersection+poly20A[1]])
    x_intersection = -(poly20C[1] - poly20B[1])/(poly20C[0] - poly20B[0])
    intersection_values.append([x_intersection,poly20B[0]*x_intersection+poly20B[1]])
    

i = len(combinations) - 1
poly20A,cov_poly20A = np.polyfit(x20[:combinations[i][1]],y20[:combinations[i][1]],1,cov=True)
x= np.linspace(0,0.9,1000)
plt.plot(x,poly20A[0]*x + poly20A[1],color="orange",label="2.27MeV β-,0.51MeV β-,\n0.54MeV β-,γ Rays")
poly20B,cov_poly20B = np.polyfit(x20[combinations[i][1]:combinations[i][2]],y20[combinations[i][1]:combinations[i][2]],1,cov=True)
x= np.linspace(0.3,1.1,1000)
plt.plot(x,poly20B[0]*x + poly20B[1],color="red",label="0.54MeV β-,γ Rays")
poly20C,cov_poly20C = np.polyfit(x20[combinations[i][2]:],y20[combinations[i][2]:],1,cov=True)
x= np.linspace(0.8,1.6,1000)
plt.plot(x,poly20C[0]*x + poly20C[1],color="green",label="γ Rays")#

estimated_values = []
for k in range(0,combinations[i][1]+1):
    estimated_values.append(poly20A[0]*x20[k] + poly20A[1])
for l in range(combinations[i][1]+1,combinations[i][2]+1):
    estimated_values.append(poly20B[0]*x20[l] + poly20B[1])
for m in range(combinations[i][2]+1,combinations[i][3]):
    estimated_values.append(poly20C[0]*x20[m] + poly20C[1])
chi_squared = 0
red_chi_squared = 0
for j in range(len(x20) - 1):
    chi_squared += (x20[j] - estimated_values[j])**2/estimated_values[j]
red_chi_squared = chi_squared / (len(x20) - 9)
print(combinations[i],": χ² = ",chi_squared,", red χ² = ",red_chi_squared)
chi_squared_values.append(chi_squared)
red_chi_squared_values.append(red_chi_squared)
x_intersection = -(poly20B[1] - poly20A[1])/(poly20B[0] - poly20A[0])
intersection_values.append([x_intersection,poly20A[0]*x_intersection+poly20A[1]])
x_intersection = -(poly20C[1] - poly20B[1])/(poly20C[0] - poly20B[0])
intersection_values.append([x_intersection,poly20B[0]*x_intersection+poly20B[1]])

plt.plot(x20,y20,'x')
plt.errorbar(x=x20,xerr=0.01,y=y20,yerr=yErr20, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("ln (Count)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Range of Decay Particles through Al", **titleFont)
plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()


plt.plot(combinations,chi_squared_values,'x')
plt.xlabel("4 Distinct Linear Regression Ranges [0,x,y,18]", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Chi Squared Parametrisation", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()

plt.plot(combinations,red_chi_squared_values,'x')
plt.xlabel("4 Distinct Linear Regression Ranges [0,x,y,18]", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Reduced Chi Squared Parametrisation", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()


plt.axvline(4.68,linestyle="dotted")
plt.axvline(8.85,linestyle="dotted")
plt.plot(intersection_values[1::2],chi_squared_values,'x')
plt.plot(intersection_values[0::2],chi_squared_values,'x')
plt.xlim(0,10)
plt.xlabel("ln(Count) Value at Intersection", **axesFont)
plt.ylabel("Chi Squared Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Chi Squared Intersection Values", **titleFont)
# plt.legend(loc="upper right",prop={'family':'SF Mono', 'size':8})
plt.show()

maximum_energy = []
for n in range(len(intersection_values)):
    maximum_energy.append(intersection_values[n][0]*2.7)
    
print(maximum_energy)

# %%
