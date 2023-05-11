# K-Means-Clustering-and-Gaussian-Mixture-Model
In this project, we will implement both K-Means clustering and Gaussian Micture Model
## K-Means Clustering [30 pts]
In this problem, you will implement the K-means algorithm for clustering. You should implement from
scratch, using either MATLAB or Python. Download the data set from Canvas. The data set contains 128
data points, each has 13 features. If you encounter missing values in the dataset, replace the missing
entry with the mean of that feature. You should run your implementation with 𝑲 = 𝟐, 𝟑, . . . , 𝟗. For
each run, initialize the cluster centers randomly among the given data, and terminate the iteration if
the cluster assignment of all data points remains unchanged (in other words, each data point will be
2
assigned to the same cluster if running more iterations). You could use slide 14 of Lecture17 as the
reference for implementation.
• Plot the objective function as a function of 𝐾.
• For 𝐾 = 2, plot the points using its first two features. Use two different colors or symbols to
distinguish the two clusters.
• Write out your observations from the obtained results.
Write the observations and explanations of the result from the figure into your report (PDF) file, and
include your implementation in the zip file.
# Gaussian Mixture Model (GMM) [40 pts]
In this problem, you will implement the Gaussian Mixture Model (or Mixture of Gaussians). You should
implement from scratch, using either MATLAB or Python. Use the same dataset as in the previous
problem. You should run your implementation with 𝑲 = 𝟐, and run until convergence. You can choose
to terminate the iteration when the change of the log-likelihood, computed by equation (9.28) on page
439 of the textbook, is smaller than a small threshold (such as 1e-5). Please use the EM algorithm on
page 438 and page 439 of the textbook as the reference for implementation, and you could use the
code in “init_gmm.m” on Canvas for initializing parameters of GMM.
• For 𝐾 = 2, plot the points using its first two features. Use two different colors or symbols to
distinguish the two clusters. The cluster assignment is determined by the posterior 𝛾(𝑧𝑛𝑘)
computed by equation (9.23) of the textbook. A data point 𝑥𝑛 is assigned to cluster 1 if 𝛾(𝑧𝑛1) >
𝛾(𝑧𝑛2), and it is assigned to cluster 2 otherwise.
• Write out your observations from the obtained results.
Write the observations and explanations of the result from the figure into your report (PDF) file, and
include your implementation in the zip file.
