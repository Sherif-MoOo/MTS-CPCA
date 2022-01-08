# PCA-FOR-MTS
I couldn't find suitable framework that deals with MTS-CPCA so I'll build the algorithm from scratch

What's MTS?

MTS (Multivariate time series) is an important type of data that
is indispensable in a variety of domains as  medicine 
domain which is the evolution of a group of synchronous
variables over a duration of time as shown in the figure and
there is a lot of effort given due to the expensive of gathering 
these labeled data to be able to offer a method gives a reliable 
accuracy by only using a limited amount of these data.

![alt text](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/mts.jpg)

What's PCA?

This is an algorithm to find orthogonal transformation axes that diagonalize the covariance 
matrix.
Suppose there was a dataset X having N multivariate time series
Σ_i=cov⁡(x_i)  , x_i ∈ R^(n_i * m) where  n_i is the length of MTS sample and m is the number of the variables
Σ=1/N ∑▒Σ_i 