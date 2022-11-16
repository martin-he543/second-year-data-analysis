import numpy as np
import matplotlib.pyplot as plt

titleFont = {'fontname': 'Kinnari', 'size': 13}
axesFont = {'fontname': 'Kinnari', 'size': 9}
ticksFont = {'fontname': 'SF Mono', 'size': 7}
errorStyle = {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle = {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle = {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle = {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}


#%% TASK 6
# ______________________________________________

E_deposited, SourceZ, SIAngleOffAxis, AbsorberThickeness = \
    np.loadtxt("6_data.txt",skiprows=1,unpack=True,delimiter=",")

plt.hist(E_deposited,bins=25)
plt.xlabel("Energy Deposited in SI", **axesFont)
plt.ylabel("Number of Particles", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 6: Histogram of Particle Simulation", **titleFont)
# plt.savefig('Task 6 25 bins.jpg', dpi=1000)
plt.show()


#%% TASK 7
# ______________________________________________

E_deposited300, SourceZ, SIAngleOffAxis, AbsorberThickeness = \
    np.loadtxt("7_data_0.3MeVOllie.txt",skiprows=1,unpack=True,delimiter=",")

E_deposited2000, SourceZ, SIAngleOffAxis, AbsorberThickeness = \
    np.loadtxt("7_data_2MeV.txt",skiprows=1,unpack=True,delimiter=",")
# H:\Lab\
plt.hist(E_deposited300,bins=50)
plt.xlabel("Energy Deposited in SI", **axesFont)
plt.ylabel("Number of Particles", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 7: Histogram of Energy Deposited by 300KeV Electrons", **titleFont)
# plt.savefig('Task 7 0.3MeV.jpg', dpi=1000)
plt.show()

plt.hist(E_deposited2000,bins=50)
plt.xlabel("Energy Deposited in SI", **axesFont)
plt.ylabel("Number of Particles", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 7: Histogram of Energy Deposited by 2MeV Electrons", **titleFont)
# plt.savefig('Task 7 2MeV.jpg', dpi=1000)
plt.show()

#%% TASK 8
# ______________________________________________

E_deposited60, SourceZ, SIAngleOffAxis, AbsorberThickeness = \
    np.loadtxt("8_data_0.06MeV_new.txt",skiprows=1,unpack=True,delimiter=",")

E_deposited2000, SourceZ, SIAngleOffAxis, AbsorberThickeness = \
    np.loadtxt("8_data_2MeV_new.txt",skiprows=1,unpack=True,delimiter=",")

plt.hist(E_deposited60,bins=50)
plt.xlabel("Energy Deposited in SI", **axesFont)
plt.ylabel("Number of Particles", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 8: Histogram of Energy Deposited by 60KeV Gamma Rays", **titleFont)
# plt.savefig('Task 8 0.06MeV.jpg', dpi=1000)
plt.show()

plt.hist(E_deposited2000,bins=50)
plt.xlabel("Energy Deposited in SI", **axesFont)
plt.ylabel("Number of Particles", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 8: Histogram of Energy Deposited by 2MeV Gamma Rays", **titleFont)
# plt.savefig('Task 8 2MeV.jpg', dpi=1000)
plt.show()


#%% TASK 9
# ______________________________________________



#%% TASK 11
# ______________________________________________
smallMeanDist = \
    np.loadtxt("11_data_v1.txt",unpack=True,delimiter=",")
    
plt.hist(smallMeanDist,bins=10)
plt.xlabel("Counts per Second / s¯¹", **axesFont)
plt.ylabel("Frequency", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 11: Distribution of Number of Counts for a Small Mean", **titleFont)

# PLOT POISSON CURVE with ERRORBARS
# plt.plot()
# plt.errorbar()

# plt.savefig('Task 11.jpg', dpi=1000)
plt.show()

# %% TASK 12
# ______________________________________________
largeMeanDist = \
    np.loadtxt("12_data.txt",unpack=True,delimiter=",")
    
plt.hist(largeMeanDist,bins=20)
plt.xlabel("Counts per Second / s¯¹", **axesFont)
plt.ylabel("Frequency", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 12: Distribution of Number of Counts for a Large Mean", **titleFont)
# plt.savefig('Task 12.jpg', dpi=1000)
plt.show()


# %% TASK 13

timeIntervalDist = \
    np.loadtxt("13_data.txt",unpack=True,delimiter=",")
    
plt.hist(timeIntervalDist,bins=20)
plt.xlabel("Counts per Second / s¯¹", **axesFont)
plt.ylabel("Frequency", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 13: Time Interval Distribution", **titleFont)
# plt.savefig('Task 13.jpg', dpi=1000)
plt.show()


# %% TASK 14
shortTimeIntervalDist = \
    np.loadtxt("14_data.txt",unpack=True,delimiter=",")
    
plt.hist(shortTimeIntervalDist,bins=20)
plt.xlabel("Counts per Second / s¯¹", **axesFont)
plt.ylabel("Frequency", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 14: Time Interval Distribution", **titleFont)
plt.savefig('Task 14.jpg', dpi=1000)
plt.show()


# %% TASK 15

Distance,Count,Time,CountRate,Val,ValError,ValErrorRelative = \
    np.loadtxt("Task 15.csv",skiprows=1,unpack=True,delimiter=",")

plt.plot(Distance,Val,'x')
plt.errorbar(x=Distance,xerr=0.001,y=Val,yerr=ValError,**errorStyle)
plt.xlabel("Distance / m", **axesFont)
plt.ylabel("$ Count Rate * Distance^{2}$ / $m^{2}s¯¹$", **axesFont)
plt.ylim(0,16)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 15: Count Rate Variation with Distance (Preliminary Readings)", **titleFont)
# plt.savefig('Task 15.jpg', dpi=1000)

pf15,cov_pf15 = np.polyfit(Distance, Val,1,cov=True)    
x15=np.linspace(0,0.8,1000)
plt.plot(x15,pf15[0]*x15+pf15[1])
plt.show()

print("m=",pf15[0],"+/-",np.sqrt(cov_pf15[0][0]),"c=",pf15[1],"+/-",np.sqrt(cov_pf15[1][1]))

plt.show()



# %% TASK 16

Distance,Count,Time,CountRate,Val,ValError,ValErrorRelative = \
    np.loadtxt("Task 16.csv",skiprows=1,unpack=True,delimiter=",")

plt.plot(Distance,Val,'x')
plt.errorbar(x=Distance,xerr=0.001,y=Val,yerr=ValError,**errorStyle)
plt.xlabel("Distance / m", **axesFont)
plt.ylabel("$ Count Rate * Distance^{2}$ / $m^{2}s¯¹$", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ylim(0,16)
plt.title("Task 16: Count Rate Variation with Distance", **titleFont)
# plt.savefig('Task 16.jpg', dpi=1000)

pf16,cov_pf16 = np.polyfit(Distance[6:], Val[6:],1,cov=True)    
x16=np.linspace(0,0.73,1000)
plt.plot(x16,pf16[0]*x16+pf16[1])
plt.show()

print("m=",pf16[0],"+/-",np.sqrt(cov_pf16[0][0]),"c=",pf16[1],"+/-",np.sqrt(cov_pf16[1][1]))


# %% TASK 19
from scipy.optimize import curve_fit

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
plt.plot(x,poly19A[0]*x + poly19A[1])

poly19B,cov_poly19B = np.polyfit(x19[9:17],y19[9:17],1,cov=True)
x= np.linspace(1,6,1000)
plt.plot(x,poly19B[0]*x + poly19B[1],color="red")

poly19C,cov_poly19C = np.polyfit(x19[18:],y19[18:],1,cov=True)
x= np.linspace(3,6,1000)
plt.plot(x,poly19C[0]*x + poly19C[1],color="green")

x_intersection = -(poly19B[1] - poly19A[1])/(poly19B[0] - poly19A[0])
print("first intersection",x_intersection,poly19A[0]*x_intersection+poly19A[1])

x_intersection = -(poly19C[1] - poly19B[1])/(poly19C[0] - poly19B[0])
print("second intersection",x_intersection,poly19B[0]*x_intersection+poly19B[1])

plt.show()

print("m_orange = ",poly19A[0],"+/-",np.sqrt(cov_poly19A[0]))
print("c_orange = ",poly19A[1],"+/-",np.sqrt(cov_poly19A[1]))
print("m_red = ",poly19B[0],"+/-",np.sqrt(cov_poly19B[0]))
print("c_red = ",poly19B[1],"+/-",np.sqrt(cov_poly19B[1]))
print("m_green = ",poly19C[0],"+/-",np.sqrt(cov_poly19C[0]))
print("c_green = ",poly19C[1],"+/-",np.sqrt(cov_poly19C[1]))

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B

params, cov_params = curve_fit(exponential_decay,x19,count19, p0=[555000,1.6,10,10])

x=np.linspace(0,6,10000)
plt.plot(x19,count19,'x')
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-')
plt.errorbar(x=x19,xerr=0.01,y=count19,yerr=countErr19, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("Count, u / s¯¹", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 19: Range of Decay Particles through Al", **titleFont)

print(params)
print(np.sqrt(cov_params[0][0]),np.sqrt(cov_params[1][1]),np.sqrt(cov_params[2][2]),np.sqrt(cov_params[3][3]))
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

print("m_orange = ",poly20A[0],"+/-",np.sqrt(cov_poly20A[0]))
print("c_orange = ",poly20A[1],"+/-",np.sqrt(cov_poly20A[1]))
print("m_red = ",poly20B[0],"+/-",np.sqrt(cov_poly20B[0]))
print("c_red = ",poly20B[1],"+/-",np.sqrt(cov_poly20B[1]))
print("m_green = ",poly20C[0],"+/-",np.sqrt(cov_poly20C[0]))
print("c_green = ",poly20C[1],"+/-",np.sqrt(cov_poly20C[1]))

area,unc_area = 1.97e-4, 1.4e-5
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B

params, cov_params = curve_fit(exponential_decay,x20,count20)

x=np.linspace(0,2,10000)
plt.plot(x20,count20,'x')
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-')
plt.errorbar(x=x20,xerr=0.01,y=count20,yerr=count20Err, **errorStyle)
plt.xlabel("Thickness / mm", **axesFont)
plt.ylabel("Count, u / s¯¹", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Task 20: Range of Decay Particles through Cu", **titleFont)

print(params)
print(np.sqrt(cov_params[0][0]),np.sqrt(cov_params[1][1]),np.sqrt(cov_params[2][2]),np.sqrt(cov_params[3][3]))
# %%
