import csv
import numpy as np
import math
import itertools
import scipy as sp
import statistics
from statistics import mode
from collections import Counter
import itertools
import operator
import random
from statistics import mean,median
from scipy.stats import multivariate_normal
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import scipy.stats as stats
import time
import prettytable
from prettytable import PrettyTable
#random.seed(2000)
from decimal import Decimal

#Extract the latent distributions (mean, variance) of the image dataset.
def Exrtract_info(Datasetsize, latentspacesize, elements, numgenerative):
    Mu = []
    LogVar = []
    samples = []
    feature_variants = []
    genfacs = []

    for i in range(Datasetsize):
        Mu.append([])
        LogVar.append([])
        samples.append([])
    for i in range(numgenerative):
        genfacs.append([])
        feature_variants.append([])

    for i in range(Datasetsize):
        t = elements[i][0]
        Mu[i] = t.split(",")
        l = Mu[i][0]
        ll = l.split("[")
        Mu[i][0] = ll[1]
        l2 = Mu[i][latentspacesize - 1]
        ll2 = l2.split("]")
        Mu[i][latentspacesize - 1] = ll2[0]
        g = elements[i][1]
        LogVar[i] = g.split(",")
        r = LogVar[i][0]
        rr = r.split("[")
        LogVar[i][0] = rr[1]
        r2 = LogVar[i][latentspacesize - 1]
        rr2 = r2.split("]")
        LogVar[i][latentspacesize - 1] = rr2[0]
        #print(elements[799])
        for j in range(numgenerative):
            feature_variants[j].append(elements[i][j + 2])

    return Mu, LogVar, feature_variants

#generating samples for a given mean and variance.
def getsample(sampling_value,m, v):
    z_sampling = []
    for p in range(sampling_value):
        ep = sp.random.standard_normal(size=1)
        z = m + v * ep
        z_sampling.append(z)
    return z_sampling

#Extract mean and variance from the dataset and sample it.
def generatesamples(sampling_value,latentspacesize,Datasetsize, Logvar,Mu):
    samples = []
    for j in range(latentspacesize):
            samples.append([])

    for j in range(latentspacesize):
        for i in range(Datasetsize):
            curLogVar = Logvar[i][j]
            curLogVar = float(curLogVar)
            curvar = math.exp(curLogVar)
            curstd = math.sqrt(curvar)
            curMu = Mu[i][j]
            curMu = float(curMu)
            sample = getsample(sampling_value,curMu, curstd)
            samples[j].append(sample)   #sampling takes place here

    return samples

def caldensity(curstd,cursamp,curMu,curvar):
        normalization = - 0.5 * (math.log(2 * math.pi) + curvar)
        inv_var = math.exp(-curvar)
        log_density = normalization - 0.5 * ((cursamp - curMu)**2 * inv_var)
        #density=(1 / (curstd * math.sqrt(2 * (math.pi)))) * math.exp(-((cursamp - curMu) ** 2) * (1 /(2 * curvar)))
        return log_density

def one_z_entropy(sampling_value,Datasetsize, Mu, Logvar,cursamples,lateid,latentspacesize):
        sample_density=[]
        Entropy = 0

        for i in range(Datasetsize):
                sample_density.append([])

        for i in range(Datasetsize):
            for k in range(sampling_value):
                sample_density[i].append([])

        for l in range(Datasetsize):
            curLogVar3 = Logvar[l][lateid]
            curLogVar3 = float(curLogVar3)
            curvar3 = math.exp(curLogVar3)
            curstd3 = math.sqrt(curvar3)
            curMu3 = Mu[l][lateid]
            curMu3 = float(curMu3)
            #print(cursamples[l])
            k=0
            for cursample in cursamples[l]:
                    c_q_zj_xi2=caldensity(curstd3, cursample, curMu3, curvar3)#Extracting mean,variance,standard-deviation and samples
                    sample_density[l][k]= c_q_zj_xi2
                    k+=1

        for k in range(sampling_value):
            sum2=[]
            for i in range(Datasetsize):
                sum2.append(sample_density[i][k])
            x=np.log(np.sum(np.exp(sum2)))/Datasetsize#changed this part. This was from the original work. torch.logsumexp()
            Entropy+=(x-math.log(Datasetsize))
            #print(Entropy)
        Entropy=-Entropy/sampling_value

        return Entropy

def est_lat_entropy(sampling_value,Datasetsize, latentspacesize, Mu, Logvar,samples):

    H_z=[]
    for j in range(latentspacesize):
        H_z.append([])

    for j in range(latentspacesize):
        cursamples=samples[j]
        H_z[j] = one_z_entropy(sampling_value,Datasetsize, Mu, Logvar, cursamples,j,latentspacesize)

    return  H_z

#Extract the unique numbers in the genfactor vectors of the dataset.
def calunique(genfactors,numgenerative):
    uniqvals =[]
    for i in range (numgenerative):
        uniqvals.append([])

    for j in range(numgenerative):
        uniqvals[j]=np.unique(genfactors[j])
        #print (uniqvals[j])
    return uniqvals

def filter_input(genfactors, Mu, Logv,samples,Datasetsize,latentspacesize,genid, uval):

    curMu=[]
    curlogvar=[]
    cursamples=[]

    for i in range(Datasetsize):
        if genfactors[genid][i]==uval:
            curMu.append(Mu[i])
            curlogvar.append(Logv[i])

    for i in range(Datasetsize):
        if genfactors[genid][i] == uval:
            cursamples.append(samples[i])
    return curMu,curlogvar,cursamples

def est_cond_ent(sampling_value,feature_variants, Mu, Logv,numgenerative, latentspacesize,uniques, Datasetsize,samples):
    #pvk, pjoint, Logvar, Mu, latentspacesize, Datasetsize,
    Cond_z_v_ent=[]

    for k in range(numgenerative):
        Cond_z_v_ent.append([])
    for k in range(numgenerative):
        for  j in range(latentspacesize):
          Cond_z_v_ent[k].append([])

    for k in range(numgenerative):
        curuniques = uniques[k]
        Entropy=0

        for j in range(latentspacesize):
            sum1=0
            for v in curuniques:

                fMu, flogvar, fsamples = filter_input(feature_variants, Mu, Logv, samples[j], Datasetsize, latentspacesize, k, v)
                curdatasize = len(fMu)
                sum1=sum1+(curdatasize/Datasetsize) * one_z_entropy(sampling_value,curdatasize, fMu, flogvar, fsamples, j,latentspacesize)
            Cond_z_v_ent[k][j] = sum1

    return Cond_z_v_ent

def calentgen(numgenerative, uniques,Datasetsize,feature_variants):
    H_v_k=[]
    for h in range(numgenerative):
        H_v_k.append([])

    for h in range(numgenerative):
        Entropy=0
        for j in uniques[h]:
            sum = 0
            for f in range(Datasetsize):
                if feature_variants[h][f]==j:
                    sum=sum+1
            sum=sum/Datasetsize
            Entropy=Entropy+sum*math.log(sum)
        H_v_k[h]=-Entropy

    return  H_v_k

def MIG_compute(Fadress, Datasetsize, latentspacesize, numgenerative,sampling_value):

    elements = []
    for i in range(Datasetsize):
        elements.append([])
    for i in range(Datasetsize):
        for j in range(4):
            elements[i].append([])

    with open(Fadress, 'rt')as f:
        data = list(csv.reader(f))
        i = 0
        for row in data:
            j = 0
            for column in row:
                elements[i][j] = column
                j = j + 1
            i = i + 1

    #Extract the mean,variance and genrative factors from the dataset
    Mu, Logvar, genfactors = Exrtract_info(Datasetsize, latentspacesize, elements, numgenerative)
    #Use the mean, variance information to generate random samples
    samples = generatesamples(sampling_value,latentspacesize,Datasetsize,Mu, Logvar)

    return Mu,Logvar,genfactors,samples

#Calculating entropy
def Calculate_Entropy(sampling_value,Datasetsize, latentspacesize, Mu, Logvar,samples, feature_variants,numgenerative):
    #Use the samples to calculate the entropy of a latent variable H(z).
    H_z = est_lat_entropy(sampling_value,Datasetsize, latentspacesize, Mu, Logvar,samples)

    uniques=calunique(feature_variants,numgenerative)#check if the image dataset has feature variations.

    H_Z_V=est_cond_ent(sampling_value,feature_variants, Mu, Logvar,numgenerative, latentspacesize,uniques, Datasetsize,samples)

    H_v_k = calentgen(numgenerative,uniques,Datasetsize,feature_variants)


    I_z_v=[]
    MIG_Tuples=[]#holds MIG information

    for k in range(numgenerative):
        I_z_v.append([])

    for k in range(numgenerative):
        for j in range(latentspacesize):
            I_z_v[k].append([])


    I_zj_vk = np.zeros((numgenerative, latentspacesize))# emperical mutual information between a feature and latent units.
    for m in range(numgenerative):
        for l in range(latentspacesize):
            I_zj_vk[m][l] = H_z[l] - H_Z_V[m][l]

    tempdict = {}
    tempmig = []
    MutualInfo = []

    for k in range(numgenerative):
        MutualInfo.append([])

    for k in range(numgenerative):# emperical mutual information search across all the latent  units.
        for j in range(latentspacesize):
            tempdict[j] = I_zj_vk[k][j]
        listofTuples = sorted(tempdict.items(), reverse=True, key=lambda x: x[1])
        MIG_Tuples.append(listofTuples)
        #print(MIG_Tuples)


       #Final MIG Formula (eqn 4 of the paper)
        if H_v_k[k] != 0:
            tempmig.append(((1 / H_v_k[k]) * (listofTuples[0][1] - listofTuples[1][1]))) # calculated MIG
        else:
            tempmig.append((listofTuples[0][1] - listofTuples[1][1]))

    #Get a ranking list of latent units based on the computed mutual information.
    Latent_Ranking = []
    for k in range(numgenerative):
        Mutual_info = []
        for i in range(latentspacesize):
            val = MIG_Tuples[k][i][0]
            Mutual_info.append(val)
        Latent_Ranking.append(Mutual_info)

    TotalMIG = 0
    for k in range(numgenerative):
        TotalMIG = TotalMIG + tempmig[k]
    TotalMIG = TotalMIG / numgenerative

    return TotalMIG,MIG_Tuples,Latent_Ranking

#To get a stable MIG score we repeat the MIG computeation Experiment several times.
#Within each Experiment, we do several iterations of MIG computations and compute an average across the iterations.
# We perform an experiment for all the shortlisted n,beta hyperparameters from random search (step1 od Deasigning a beta-VAE monitor in paper)

def takeFirst(elem):
    return elem[1]

def takeSecond(elem):
    return elem[0]
