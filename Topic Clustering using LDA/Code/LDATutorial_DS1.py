

from sklearn.feature_extraction.text import CountVectorizer
import pickle as pickle
import numpy as np
import scipy.special as special
import scipy.optimize 
import time
import matplotlib.pyplot as plt

vectorizer = pickle.load(open("Data/vectorizerNews.p", "rb"))
trainDocs = pickle.load(open("Data/trainDocsNews.p", "rb"))
testDocs = pickle.load(open("Data/testDocsNews.p", "rb"))

#also load the original docs that aren't pre-processed for viewing later
origTrainDocs = pickle.load(open("Data/trainDocsNewsOrig.p", "rb"))
origTestDocs = pickle.load(open("Data/testDocsNewsOrig.p", "rb"))

print("Found ", len(trainDocs), "training and ", len(testDocs), " test documents, with a vocabulary of ", len(vectorizer.get_feature_names()), " words.")

diGamma = special.digamma


def maxVIParam(phi, gamma, B, alpha, M, k, Wd, eta):
    for d in range(M):
        N = len(Wd[d])
        #Initialization of vars, as shown in E-step. 
        phi[d] = np.ones((N,k))*1.0/k
        gamma[d] = np.ones(k)*(N/k) + alpha
        gammaOld=gamma[d]
        converged = False
        j = 0
        #YOUR CODE FOR THE E-STEP HERE
        while(not(converged)):
            j=j+1
            phi[d]=B[:,Wd[d]].T*np.exp(diGamma(gamma[d]))
            phi[d]=phi[d]/np.sum(phi[d],axis=1).reshape(-1,1)
            gamma[d]=alpha+np.sum(phi[d],axis=0)
            if(np.amax(np.abs(gamma[d]-gammaOld))<eta):
                converged=True
            gammaOld=gamma[d]
  
    return gamma, phi

def getWordVector(wd,V):
        wordVector=np.zeros([V])
        unique_elements, counts_elements = np.unique(wd, return_counts=True)
        wordVector[unique_elements]=counts_elements
        return wordVector

def MaxB(B, phi, k, V, M, Wd):
    #YOUR CODE FOR THE M-STEP HERE
    B=np.zeros([k,V])
    for d in range(M):
        for n in range(len(Wd[d])):
            B[:,Wd[d][n]]+=phi[d][n]
    return B

def L_alpha_val(a):
    val = 0
    M = len(gamma)
    k = len(a)
    for d in range(M):
        val += (np.log(scipy.special.gamma(np.sum(a))) - np.sum([np.log(scipy.special.gamma(a[i])) for i in range(k)]) + np.sum([((a[i] -1)*(diGamma(gamma[d][i]) - diGamma(np.sum(gamma[d])))) for i in range(k)]))

    return -val

def L_alpha_der(a):
    M = len(gamma)
    k = len(a)
    der = np.array(
    [(M*(diGamma(np.sum(a)) - diGamma(a[i])) + np.sum([diGamma(gamma[d][i]) - diGamma(np.sum(gamma[d])) for d in range(M)])) for i in range(k)]
    )
    return -der

def L_alpha_hess(a):
    hess = np.zeros((len(a),len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            k_delta = 1 if i == j else 0
            hess[i,j] = k_delta*M*scipy.special.polygamma(1,a[i]) - scipy.special.polygamma(1,np.sum(a))
    return -hess

def MaxA(a):
    res = scipy.optimize.minimize(L_alpha_val, a, method='Newton-CG',
        jac=L_alpha_der, hess=L_alpha_hess,
        options={'xtol': 1e-8, 'disp': False})
    # print(res.x)
    return res.x



eta = 10e-5 #threshold for convergence
np.random.seed(111)
#hyperparamater init.
V = len(vectorizer.get_feature_names()) #vocab. cardinality
M = int(len(trainDocs)) #number of documents
k = 4 #amount of emotions

nIter=100 
B = np.random.rand(k,V)

for i in range(k):
    B[i,:] = B[i]/np.sum(B[i])
    
alpha = np.ones(k)

phi = [None]*M
gamma = [None]*M

Wd = [None]*M
Bplot=[]
for d in range(M):
    Wmat = vectorizer.transform([trainDocs[d]]).toarray()[0] #get vocabulary matrix for document
    WVidxs = np.where(Wmat!=0)[0]
    WVcounts = Wmat[WVidxs]
    N = np.sum(WVcounts)
    W = np.zeros((N)).astype(int)

    i = 0
    for WVidx, WV in enumerate(WVidxs):
        for wordCount in range(WVcounts[WVidx]):
            W[i] = WV
            i+=1
    Wd[d] = W #We save the list of words for the document for analysis later


#start of parameter estimation loop
for j in range(nIter):
    if(j%10==0):
        print("Iteration: "+str(j)+" of "+str(nIter))
    #Variational EM for gamma and phi (E-step from VI section)
    gamma, phi = maxVIParam(phi, gamma, B, alpha, M, k, Wd, eta)
    Bold = np.copy(B)
    B = MaxB(B,phi,k,V,M,Wd) #first half of M-step from VI section 
    #renormalize B
    for i in range(k):
        B[i,:] = B[i]/np.sum(B[i])
    Bplot.append(np.amax(abs(B-Bold)))
    alpha = MaxA(alpha) #second half of M-step from VI section 

plt.title('Largest Absolute difference between successive $\\beta_{i,j}$ updates')
plt.xlabel('Iterations')
plt.plot(Bplot)
plt.savefig('Plots/newsgroup_convergence')
plt.show()


# pickle.dump(alpha, open("Data/myAlphaNews100.p", "wb"))
# pickle.dump(B, open("Data/myBetaNews100.p", "wb"))

#alpha = pickle.load(open("myAlphaNews.p", "rb"))
#B = pickle.load(open("myBetaNews.p", "rb"))


nTop = 20
import pandas as pd
data=[]
for i in range(k):
    topVocabs = np.argsort(B[i])[-nTop:][::-1]
    topWords = np.array(vectorizer.get_feature_names())[topVocabs]
    print("Topic:",i)
    print(topWords)
    data.append(topWords.T)
dataframe = pd.DataFrame.from_records(data)
# dataframe.to_csv('Plots/leanedTopic.csv', encoding='utf-8', index=True)



data=[]
alphaTest = pickle.load(open("Data/CompareAlphaNews200.p", "rb"))
BTest = pickle.load(open("Data/CompareBetaNews200.p", "rb"))
vecTest = pickle.load(open("Data/vectorizerNews.p", "rb"), encoding='latin1')
nTop = 20
for i in range(k):
    topVocabs = np.argsort(BTest[i])[-nTop:][::-1]
    topWords = np.array(vecTest.get_feature_names())[topVocabs]
    print("Topic:",i)
    print(topWords)
    data.append(topWords.T)
dataframe = pd.DataFrame.from_records(data)
# dataframe.to_csv('Plots/Precomputed.csv', encoding='utf-8', index=True)


# we are not re-initializing beta and alpha, we calculated them using the training docs.

V = len(vectorizer.get_feature_names()) #vocab. cardinality
M = int(len(testDocs)) #number of documents
k = 4 #amount of emotions

# #variational params (one for each doc)
phi = [None]*M
gamma = [None]*M
WdTest = [None]*M

# '''Same magic from before to get the word matrix correct, replace this if you redid this earlier.'''

for d in range(M):
    Wmat = vectorizer.transform([testDocs[d]]).toarray()[0] #get vocabulary matrix for document
    WVidxs = np.where(Wmat!=0)[0]
    WVcounts = Wmat[WVidxs]
    N = np.sum(WVcounts)
    W = np.zeros((N)).astype(int)
    i = 0
    for WVidx, WV in enumerate(WVidxs):
        for wordCount in range(WVcounts[WVidx]):
            W[i] = WV
            i+=1
    WdTest[d] = W #We save the list of words for the document for analysis later

# '''Now that you have your variables initialized for the test documents, you should be able to use your previous code for 
# maximizing the VI parameters with those variables instead. Remember, we're just calculating the variational parameters
# gamma and phi for each test document so there is no iteration between maximizing Beta and maximizing gamma and phi.'''

gamma, phi = maxVIParam(phi, gamma, B, alpha, M, k, WdTest, eta)


# #take a look at some example test documents (14-24 has a nice mix of topics, with a couple difficult ones)
dStart = 15
dEnd = 16
np.random.seed(654)
for d in np.random.randint(low=0, high=49, size=5):
# for d in range(dStart,dEnd):
    print("Estimated mixture for document ", d," is: ")
    print("_______________________")
    for i in range(len(gamma[d])):
        print("topic ", i,": ", gamma[d][i]/np.sum(gamma[d]))
    print("_______________________")
    print("Which has the following text:")
    print(" ")
    print(origTestDocs[d])
    print("__________________________________________")
    print("__________________________________________")


# #14-24 gives a good mix, but try whatever you like
dStart = 15 
dEnd = 16 
def getWordsFromMatrix(WdTest):
    originalWords  = np.array(vectorizer.get_feature_names())[WdTest] 
    return originalWords

# for dk in range(dStart,dEnd):
for dk in np.random.randint(low=0, high=49, size=5):    
    origWords = getWordsFromMatrix(WdTest[dk]) 
    wordMixtures = [origWords[n] + "\t: " + str(phi[dk][n]) for n in range(len(phi[dk]))]
    for wm in set(wordMixtures):
        print(wm)
    print("________________________________")




# #loading the AP docs dataset instead:
# #(everything else should work like before)
# vectorizer = pickle.load(open("Data/vectorizerAP.p", "rb"), encoding='latin1')
# trainDocs = pickle.load(open("Data/trainDocsAP.p", "rb"), encoding='latin1')
# testDocs = pickle.load(open("Data/testDocsAP.p", "rb"), encoding='latin1')


# #loading the moodyLyrics dataset instead:
# vectorizer = pickle.load(open("Data/vectorizerMoodyLyrics.p", "rb"), encoding='latin1')
# trainLyricsFile = pickle.load(open("Data/trainDocsMoodyLyrics.p", "rb"), encoding='latin1')
# testLyricsFile = pickle.load(open("Data/testDocsMoodyLyrics.p", "rb"), encoding='latin1')

# trainDocs = trainLyricsFile['lyrics']
# testDocs = testLyricsFile['lyrics']
# #original moods can be seen with: trainGT = trainLyricsFile['groundTruth'] but the labeling is not perfect. 




