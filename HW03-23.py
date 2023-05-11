#CSE 575 - HOMEWORK 3 - Questions 2 and 3
#Farshad Nahangi
#ASU ID:  1218881612

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:37:37 2023

@author: Fnaha
"""
#------------------------------------------Homework 3, Question 2 --------------------------------------------------------

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#import matplotlib.colors as mcolors
from webcolors import name_to_rgb

df = pd.read_csv('CSE575-HW03-Data.csv', header=None)

#replace the missing entry with the mean of that feature
feature_means = df.mean(axis=0)
df.fillna(feature_means, inplace=True)


num_rows, num_cols = df.shape
arr = df.values


col_min = np.amin(arr, axis=0)
col_max = np.amax(arr, axis=0)
J = [None] * 10
centers = [None] * 10
clusters = [None] * 10
rk = [None] * 10


def Kmeans(num_features):
    for K in range(2,10):
        clusters[K] = [None]*K
        rk[K] = [None]*K
        # generate a random array of initial centers 
        mu = np.zeros((K, num_cols))
    
        for k in range(K):
            mu[k] = arr[random.randint(0,num_rows-1)]
        count = 0
        while True:
            count += 1
            # calculating the squared norms 
            norm_squared = np.zeros((num_rows, K))
            
            r = np.zeros((num_rows,K))
            
            for x in range(num_rows):
                for k in range(K):
                    v = arr[x,0:num_features] - mu[k,0:num_features]
                    norm_squared[x,k] = np.dot(v, v)
            
            min_norms = np.amin(norm_squared, axis=1)
            
            for x in range(num_rows):
                k = np.where(norm_squared[x] == min_norms[x])[0][0]
                r[x,k] = 1
                
                
                
            mu2 = mu.copy()
            
            for k in range(K):
                rk[K][k] = np.reshape(r[:,k],(num_rows,1)).copy()
                temp = rk[K][k] * arr
                if (np.sum(rk[K][k], axis=0) !=0):
                    mu2[k]= np.sum(temp, axis=0) / np.sum(rk[K][k], axis=0)
    
        
            if (mu2 == mu).all():
                break
            else:
                mu = mu2    
        print('..', end="")    
        J[K] = sum(sum(r * norm_squared))
        centers[K] = mu
        for k in range(K):
            condition = rk[K][k] * arr != 0
            sub_arr = arr[condition]
            clusters[K][k] = sub_arr.reshape(int(sub_arr.shape[0]/num_cols), num_cols)
    return centers, clusters

# running on all 13 features:    
centers13, clusters13 = Kmeans(13)
#writing centers with 13 features in a file
with open('results_13_features.txt', 'w') as file:
    for K in range(2,10):
        file.write('K=' + str(K) + '\n')
        for k in range(K):
            file.write('Center_'+ str(k) +': ')
            for i in range(13):
                file.write(str(round(centers[K][k][i],3))+', ')
            file.write('\n')
        file.write('--------------------------------------------------------\n')
        
        
X = [2,3,4,5,6,7,8,9]
Y = J[2:10]

J2 = J.copy()
#ploting the objective function vs K
plt.figure()
plt.figure().set_dpi(300)
plt.xlabel('K')
plt.ylabel('J(K)')
plt.title('K-means Method: Objective Function J(K)')
plt.plot(X, Y)

# running on only the first 2 features:
centers2, clusters2 = Kmeans(2)

#writing centers with 2 features in a file
with open('results_2_features.txt', 'w') as file:
    for K in range(2,10):
        file.write('K=' + str(K) + '\n')
        for k in range(K):
            file.write('Center_'+ str(k) +': ')
            for i in range(2):
                file.write(str(round(centers[K][k][i],3))+', ')
            file.write('\n')
        file.write('----------------------------\n')

        
# ploting the classification for all values of K = 2, ..., 9 for 2 features:
p = [None] * 9
px = [None] * 9
py = [None] * 9
c = [None] * 9
colors = ["red", "blue", "green", "Magenta", "purple", "orange", "brown", "pink", "teal"]


for K in range(2,10):
    plt.figure()
    plt.figure().set_dpi(300)
    plt.title(str(K) + '-means Method')
    for k in range(K):
        p[k] = clusters[K][k]
        px[k] = p[k][:,0]
        py[k] = p[k][:,1]
        c[k] = centers[K][k][0:2]
        plt.scatter(px[k],py[k], color = colors[k])
        plt.scatter(c[k][0], c[k][1],  color = 'black', marker='x')

#plt.show()

# If you wish to skip running the code for question 3, you can comment out all the followin codes and uncomment the "plt.show()" code above. 
#This will allow you to display the plot associated with question 2, while bypassing the execution of the code for question 3.
#------------------------------------------Homework 3, Question 3 --------------------------------------------------------
numClusters = 5 # per question is set to 2 by default.
dimension = 2  # can be set to any integer number from 2 to 13.
threshold = 1e-5 


k_means, kmeans_clusters = Kmeans(dimension)


# Step 1. Initialize the means, covariances, and mixing coefficients ---------------
initMeans = k_means[numClusters][:,0:dimension]
initCovariances = [None] * numClusters
initPriors = np.zeros(( numClusters, 1));
data = kmeans_clusters[numClusters]

for k in range(numClusters):
    initPriors[k] = data[k][:, 0:dimension].shape[0] / arr.shape[0]
    if (data[k][:, 0:dimension].shape[0] == 0 or data[k][:, 0:dimension].shape[1] == 0):
        initCovariances[k] = np.cov(arr[:,0: dimension].T)
    else:
       initCovariances[k] = np.cov(data[k][:, 0:dimension].T)

# Step 2. E step -------------------------------------------------------------------

def gamma(n,k, Priors, Means, Covariances):
    e = Priors[k] * multivariate_normal.pdf(arr[n,0: dimension], Means[k], Covariances[k])
    s = 0
    for j in range(numClusters):
        s = s + Priors[j] * multivariate_normal.pdf(arr[n,0: dimension], Means[j], Covariances[j])
    return e / s


priors = initPriors
means = initMeans
covariances = initCovariances

log_likelihood = 0
for n in range(num_rows):
    s= 0
    for k in range(numClusters):
        s = s + initPriors[k] * multivariate_normal.pdf(arr[n,0: dimension], initMeans[k], initCovariances[k])
    log_likelihood = log_likelihood + np.log(s)


counter = 0
print('\nThis may takes few minutes, pleas wait for the process to be done.')
while True:
    Gamma = np.zeros((num_rows, numClusters))
    for n in range(num_rows):
        for k in range(numClusters):
            Gamma[n,k]=gamma(n,k, priors, means, covariances)
            
    N = np.zeros((numClusters,1))
    for k in range(numClusters):
        N[k] = sum(Gamma[:,k])
    
    means_new = np.zeros((numClusters, dimension))
    for k in range(numClusters):
        means_new[k] = sum(Gamma[:,k].reshape(num_rows,1)*arr[:,0: dimension])/ N[k]
    
    covariances_new = [None] * numClusters

    for k in range(numClusters):
        covariances_new[k] = 0
        for n in range(num_rows):
            covariances_new[k] = covariances_new[k] + Gamma[n,k] * np.dot(arr[n,0: dimension].reshape(dimension,1)-means_new[k].reshape(dimension,1), (arr[:,0: dimension][n] - means_new[k]).reshape(dimension,1).T)
        covariances_new[k] = covariances_new[k] / N[k]
        
    priors_new = N / num_rows
    
    log_likelihood_new = 0
    for n in range(num_rows):
        s= 0
        for k in range(numClusters):
            s = s + priors_new[k] * multivariate_normal.pdf(arr[n,0: dimension], means_new[k], covariances_new[k])
        log_likelihood_new = log_likelihood_new + np.log(s)



    if abs(log_likelihood - log_likelihood_new ) < threshold:
        break
    else:
        counter += 1
        log_likelihood = log_likelihood_new
        print('..', end="")
        priors = priors_new
        means = means_new
        covariances = covariances_new

gama = np.array([[0.0] * numClusters] * num_rows )

#for K in range(numClusters):
cluster_MoG = np.array([0]* num_rows)
color_MoG = [None]*num_rows
for n in range(num_rows):
    for k in range(numClusters):
        gama[n, k] = gamma(n, k, priors_new, means_new, covariances_new)
    cluster_MoG[n] = np.where(gama[n,:] == max(gama[n,:]))[0][0]
    color_MoG[n] = max(gama[n,:])

cluster_GMM = [None] * numClusters
color_GMM = [None] * numClusters

# for k in range(numClusters):
#     cluster_GMM[k] = arr[:,0: dimension][cluster_MoG==k,:]
#     color_GMM[k] = np.array(color_MoG)[cluster_MoG==k]

for k in range(numClusters):
    print('center ' + str(k) + ': ' + str(means_new[k]))
#------------------------------------------------------------------------------
# plot the points and contour levels for K = 2

if dimension == 2:

    K = numClusters
    rgb_color = [None] * numClusters

# #colors = ["red", "blue", "green", "Magenta", "purple", "orange", "brown", "pink", "teal"]
#     rgba_colors = [mcolors.to_rgba(color) for color in colors]
#     mixed_r = mixed_g = mixed_b = mixed_a = 0.0
    
#     mixed_color = [None] * num_rows
#     for i in range(num_rows):    
#         for k in range(K):
#             mixed_r += gama[i][k] * rgba_colors[k][0]
#             mixed_g += gama[i][k] * rgba_colors[k][1]
#             mixed_b += gama[i][k] * rgba_colors[k][2]
#             mixed_a += gama[i][k] * rgba_colors[k][3]

    for k in range(K):        
        # Get RGB color code from color name
        rgb_color[k] = list(name_to_rgb(colors[k]))

    mixed_color = [None] * num_rows
    for i in range(num_rows):  
        sss = [0, 0, 0]
        for k in range(K):
            temp = [gama[i][k] * x for x in rgb_color[k]]
            sss = [sss[t] + temp[t] for t in range(3)]
        mixed_color[i] = np.round(sss).astype(int)
        mixed_color[i] = [x/255 for x in mixed_color[i]]
    
    
    
#        mixed_color[i] = (mixed_r, mixed_g, mixed_b, mixed_a)
        
    for k in range(numClusters):
        cluster_GMM[k] = arr[:,0: dimension][cluster_MoG==k,:]
        color_GMM[k] = np.array(mixed_color)[cluster_MoG==k]

    # for k in range(K):
        
    # rgb_color[0]= [(x, 0, 1-x) for x in color_GMM[0]]
    # rgb_color[1]= [(1- x, 0, x) for x in color_GMM[1]]



    plt.figure()
    plt.figure().set_dpi(300)
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.title('Gaussian Mixture Model')
    for k in range(K):
        p[k] = cluster_GMM[k]
        px[k] = p[k][:, 0]
        py[k] = p[k][:, 1]
        c[k] = means_new[k]
        plt.scatter(px[k],py[k], c = colors[k])
        plt.scatter(c[k][0], c[k][1],  color = 'black', marker='x')
    
    fig, ax = plt.subplots()
    for k in range(K):
        p[k] = cluster_GMM[k]
        px[k] = p[k][:, 0]
        py[k] = p[k][:, 1]
        c[k] = means_new[k]
        ax.scatter(px[k],py[k], c = color_GMM[k])
        ax.scatter(c[k][0], c[k][1],  color = 'black', marker='x')
    
 
    
    x = np.linspace(55, 100, 100)
    y = np.linspace(-17.5, 12.5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    Z=[None]*K
    for k in range(K):
        Z[k] = multivariate_normal.pdf(pos, means_new[k], covariances_new[k])
    
    
    color_1 = '#87CEFA' # light blue
    
    mmax = [None]*K
    
    for k in range(K):
        mmax[k] = Z[k].max()
        
    
    contour_levels = [None]*K
    for k in range(K):
        contour_levels[k] = [i * 0.2 * mmax[k] for i in range(1, 10)]
        
        
    contour_plot = [None]*K
    
    for k in range(K):
        contour_plot[k] = ax.contour(X, Y, Z[k], levels=contour_levels[k], colors=colors[k], linewidths=0.5)
        
    ax.set_xlabel('1st feature')
    ax.set_ylabel('2nd feature')
    ax.set_title('Gaussian Mixture Model')
    ax.legend()
    fig.set_dpi(300)
    plt.show()