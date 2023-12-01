# import libraries
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import pylab
from statsmodels.distributions.empirical_distribution import ECDF

#import the data
cars = pd.read_csv(r"D:/360digi/DS/Sharath/Confidence_Interval_1/handson/Datasets-Confidence Interval/Cars.csv") #, skip_blank_lines=True).dropna() 
cars.info()
cars.describe()

cars.MPG.mean()
cars.MPG.std()

plt.hist(cars.MPG)

#Q1
# Data is not normal so we can apply empirical distribution
#a P(MPG>38)
ecdf = ECDF(list(cars.MPG))
1-ecdf(38)
# 0.4074 is the probability

#b P(MPG<40)
ecdf(40)
# 0.7530  is the probability

#c P(20<MPG<50)
ecdf(50) - ecdf(20)
# 0.8518 is the probability between 20% and 50% of the data.

# Assuming the data is normal
# z-distribution
# cdf => cumulative distributive function

stats.cdf()
#a.	P(MPG>38)
stats.norm.cdf(38, 34.4220, 9.1314)  # Given a value, find the probability
#1 - 0.6524
1 - 0.6524 
# 0.3476 is the probability

# b. P(MPG<40)
stats.norm.cdf(40, 34.4220, 9.1314)
#0.7293

#c. P(20<MPG<50)
stats.norm.cdf(50, 34.4220, 9.1314)
# 0.9559

stats.norm.cdf(20, 34.4220, 9.1314)
# 0.0571

0.9559 - 0.0571
# 0.8987 or 89% is the probability between 20% and 50% of the data.

# Q2

wc_at = pd.read_csv(r"D:/360digi/DS/Sharath/Confidence_Interval_1/handson/Datasets-Confidence Interval/wc-at.csv")
wc_at.describe()

# Normal Quantile-Quantile Plot
# Checking whether data is normally distributed
plt.hist(cars.MPG)
stats.probplot(cars.MPG, dist = "norm", plot = pylab)

plt.hist(wc_at.Waist)
stats.probplot(wc_at.Waist, dist = "norm", plot = pylab)

plt.hist(wc_at.AT)
stats.probplot(wc_at.AT, dist = "norm", plot = pylab)


#Q3
# z-distribution

#90% Z-value = 1.6448
stats.norm.ppf(0.95,0, 1) #95%

stats.norm.ppf(0.05,0, 1) #5%

#94% Z-value = 1.8807
stats.norm.ppf(0.97,0, 1) #97

stats.norm.ppf(0.03, 0, 1) #3%


#60% Z-value = 0.8416
stats.norm.ppf(0.80,0, 1) #80%

stats.norm.ppf(0.20,0, 1) #20%

#Q4
# n = 25

# t-distribution
#95% t-value = 2.0595
stats.t.ppf(0.975, 25) # Given probability, find the t value

stats.t.ppf(0.025, 25)

stats.t.cdf(2.0595, 25)
#pvalue - 0.9749


#96% t-value = 2.1665
stats.t.ppf(0.98, 25) # Given probability, find the t value

stats.t.ppf(0.02, 25)

stats.t.cdf(-2.1665, 25)
#pvalue - 0.9799


#99% t-value = 2.7874
stats.t.ppf(0.995, 25) # Given probability, find the t value

stats.t.ppf(0.005, 25)

stats.t.cdf(2.7874, 25)
#pvalue - 0.9949


#Q5
# mu = 270 days, n = 18, x(bar) = 260 days, sigma = 90days

stats.t.cdf(-0.4714, 18)
# 0.3215 or 32% is the probability that the randomly selected bulbs last less than 260 days.


#Q6
1 - stats.norm.cdf(50, 45, 8)

# 0.266


#Q7
# P(X > 44)
1 - stats.norm.cdf(44, 38, 6)
# 0.1586

# P(38 < X < 44)
stats.norm.cdf(38, 38, 6) - stats.norm.cdf(44, 38, 6)
# 0.3413

# P(X < 30)
stats.norm.cdf(30, 38, 6)
0.0912


#Q9 
stats.norm.ppf(0.995, 100, 400)
# 1130.3317
stats.norm.ppf(0.005, 100, 400)
# -930.3317



#Q10
# Profit 1 = 45(5, 9) = 225, 405
# Profit 2 = 45(7, 16) = 315, 720
#95% probability

stats.norm.ppf(0.975, 540, 1125)  
stats.norm.ppf(0.025, 540, 1125)  

# [2744.9594, -1664.9594 ] - CI

stats.norm.ppf(0.05, 540, 1125) 
#-1310.4603


#*****************************************************************







