import numpy as np
import scipy.stats as stats
from math import *
import matplotlib.pylab as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

files=['Tut7N50.npy','Tut7N100.npy','Tut7N200.npy','Tut7N500.npy','Tut7N1000.npy','Tut7N1500.npy','Tut7N2000.npy','Tut7N2100.npy']

# to save files for 3d plot in matlab
# for f in files:
#     [NumberOfParticles,T,x,x_hat,y,y_hat,particles]=np.load(f,allow_pickle=True)
#     d='a'+str(NumberOfParticles)+'.txt'
#     np.savetxt(d, particles, delimiter=',')
for f in files:
    [NumberOfParticles,T,x,x_hat,y,y_hat,particles]=np.load(f,allow_pickle=True)
    plt.plot(x,linewidth=.5,label="True State")
    plt.plot(x_hat,linewidth=.5,label="Estimated State")
    plt.title('True State vs Estimated State for N='+str(NumberOfParticles))
    plt.legend()
    plt.savefig('Plots/StateN'+str(NumberOfParticles))
    plt.show()

    plt.plot(y,linewidth=.5,label="True Observation ")
    plt.plot(y_hat,linewidth=.5,label="Estimate Observation")
    plt.title('True Observation vs Estimated Observation for N='+str(NumberOfParticles))
    plt.legend()
    plt.savefig('Plots/ObservationN'+str(NumberOfParticles))
    plt.show()

