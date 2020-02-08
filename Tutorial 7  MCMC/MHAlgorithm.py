import numpy as np
import scipy.stats as stats
from math import *
import matplotlib.pylab as plt

def targetDistribution(z):
    mu1=0
    sigma1=1
    mu2=3
    sigma2=0.5
    a1=0.5
    a2=0.5
    return a1*stats.norm.pdf(z,mu1,sigma1)+a2*stats.norm.pdf(z,mu2,sigma2)

sigma=np.array([0.1,1,10,100])
for sigmaP in sigma:
    n = 10000
    alpha = 1
    x = 0.
    vec = []
    vec.append(x)
    muP=np.mean(vec)
    proposal = stats.norm(muP,sigmaP).rvs(n) 
    for i in range(1,n):
        current_vec=vec[i-1]
        can = x + proposal[i]
        aprob = np.amin([1.,targetDistribution(can)/targetDistribution(x)]) 
        u = uniform(0,1)
        if u < aprob:
            x = can
            vec.append(x)
        else:
            vec.append(x)
    x = arange(-5,5,.1)
    y = targetDistribution(x)
    plt.subplot(2, 1, 1)
    plt.title('Metropolis-Hastings ,$\sigma_p=:$'+str(sigmaP))
    # plt.plot(vec)
    plt.plot(vec,'.',markersize=1)

    plt.subplot(2, 1, 2)
    plt.hist(vec,50,density=True,alpha=0.4)
    plt.plot(x,y)
    plt.ylabel('Frequency')
    plt.xlabel('x')
    plt.legend(('PDF','Samples'))
    plt.savefig('Plots/'+str(np.where(sigma == sigmaP)[0][0]))
    plt.show()
    

