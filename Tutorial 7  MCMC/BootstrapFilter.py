import numpy as np
import scipy.stats as stats
from math import *
import matplotlib.pylab as plt
import seaborn as sns

def getGaussianNoise(mean,sigma2,k):
    sigma=np.sqrt(sigma2)
    pdf=stats.norm(mean,sigma)
    return pdf.rvs(k)

def nextState(x,t,vt):
    return 0.5*x +25*x/(1+np.power(x,2)) + 8*np.cos(1.2*t)+vt

def getObservation(x,t,wt):
    return np.power(x,2)/20+wt

def priorX1(k):
    return getGaussianNoise(0,10,k)

def likelihood(t,y,x):
    a=abs(y-getObservation(t,x,0))
    return getGaussianNoise(0,a,1)


NumberOfParticles=100
for NumberOfParticles in  [50,100,200,500,1000,1500,2000]:
    T=200 # Number of Time Steps
    x=np.zeros(T)
    y=np.zeros(T)
    w=np.zeros(T)
    v=np.zeros(T)

    v[0]=0
    w[0]=getGaussianNoise(0,1,1)
    x[0]=0
    y[0]=getObservation(x[0],0,w[0])

    for t in range(1,T):
        v[t]=getGaussianNoise(0,10,1)
        w[t]=getGaussianNoise(0,1,1)
        x[t]=nextState(x[t-1],t,v[t])
        y[t]=getObservation(x[t],t,w[t])

    x_hat=np.zeros(T)
    y_hat=np.zeros(T)


    Weights=np.zeros([T,NumberOfParticles])
    particles=np.zeros([T,NumberOfParticles])
    xk=np.zeros(NumberOfParticles)
    wk=np.zeros(NumberOfParticles)
    for t in range(0,T):
        print('Iteration ='+str(t))
        if t==0:
            particles[t]=priorX1(NumberOfParticles)
            Weights[t]=1/NumberOfParticles
        else:
            particlesPrev=particles[t-1]
            WeightsPrev=Weights[t-1]

            for i in range(NumberOfParticles):
                
                xk[i]=nextState(particlesPrev[i],t,getGaussianNoise(0,10,1))
                pl=stats.norm(0,1).pdf(y[t]-getObservation(xk[i],t,0))
                wk[i]=pl

            wk=wk/np.sum(wk)
            if(1/np.sum(np.power(wk,2)<0.50*NumberOfParticles)):
            
                xk=np.random.choice(xk,NumberOfParticles,p=wk)
                # wk[i]=getObservation(xk[i],t,getGaussianNoise(0,0,1))
            Weights[t]=1/NumberOfParticles
            x_hat[t]=np.sum(wk*xk)
            # xk=xk/np.sum(xk)
            # #plot from lib
            particles[t]=xk
            # if(t%10==0):
            #     lb='Timestep='+str(t)+'and N='+str(NumberOfParticles)
            #     sns.kdeplot(particles[t],kernel='gau')
            #     plt.title(lb)
            #     # plt.xlim(-25,25)
            #     # plt.legend()
            # plt.show()
        
            for i in range(T):
                y_hat[i]=getObservation(x_hat[i],i,getGaussianNoise(0,1,1))

    save=[NumberOfParticles,T,x,x_hat,y,y_hat,particles]
    filename='Tut7N'+str(NumberOfParticles)
    np.save(filename,save)





