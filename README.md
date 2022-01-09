# PCA-FOR-MTS
I couldn't find suitable framework that deals with MTS-CPCA so I'll build the algorithm from scratch

What's MTS?

Time series data can be seen everywhere including stock exchange, financial market, medicine and engineering,
which is one kind of the important data needed to be mined for the valuable information and knowledge. It has two
categories according to various number of the variables, they are univariate time series (UTS) and multivariate time
series(MTS).

MTS (Multivariate time series) is an important type of data that
is indispensable in a variety of domains as  medicine 
domain which is the evolution of a group of synchronous
variables over a duration of time as shown in the figure and
there is a lot of effort given due to the expensive of gathering 
these labeled data to be able to offer a method gives a reliable 
accuracy by only using a limited amount of these data.

![alt text](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/mts.jpg)

What's PCA?

This is a linear unsupervised algorithm to find orthogonal transformation axes that diagonalize the covariance 
matrix the goal is to eliminate low variance and high correlated features.

Why PCA?

Due to the high
dimensionality of MTS, the dimensionality reduction is proposed to validly integrate into the clustering,classification and regression process and a good MTS accuracy can be obtained in lower dimensions

Suppose there was a dataset X having N multivariate time series
Σ_i=cov⁡(x_i)  , x_i ∈ R^(n_i * m) where  n_i is the length of MTS sample and m is the number of the variables
Σ = 1/N ∑Σ_i 