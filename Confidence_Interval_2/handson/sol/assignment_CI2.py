# import libraries

import scipy
from scipy import stats


# Q3

#94% Z-value = 1.8807
stats.norm.ppf(0.97,0, 1) #97%

stats.norm.ppf(0.03,0, 1) #3%


#98% Z-value = 2.3263
stats.norm.ppf(0.99,0, 1) #99%

stats.norm.ppf(0.01,0, 1) #1%


#96% Z-value = 2.0537
stats.norm.ppf(0.98,0, 1) #98%

stats.norm.ppf(0.02,0, 1) #2%




# Q5
#95% Z-value = 1.9599
stats.norm.ppf(0.975,0, 1) #97.5%

stats.norm.ppf(0.025,0, 1) #2.5%



# z-distribution
#95% Z-value = 1.96

stats.norm.ppf(0.975,0, 1) #97.5%

stats.norm.ppf(0.025,0, 1) #2.5%


# t-distribution
#95% t-value = 2.0595
stats.t.ppf(0.975, 1) # Given probability, find the t value

stats.t.ppf(0.025, 1)

stats.t.cdf(2.0595, 1)
#pvalue - 0.9749