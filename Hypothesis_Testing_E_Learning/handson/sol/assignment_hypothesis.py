# Hypothesis Testing Assignments

# import libraries
import pandas as pd
import scipy
from scipy import stats
#import statsmodels.stats.descriptivestats as sd
#from statsmodels.stats import weightstats as stests

# 1. F&B manager wants to determine the difference in cutlet diameter between two units.
# Y is continuous and X(Unit A, Unit B) is continuous. alpha = 0.05 

#import the data
cutlet = pd.read_csv(r"D:/360digi/DS/Sharath/Hypothesis_Testing_E_Learning/handson/Datasets_Hypothesis Testing/Cutlets.csv", skip_blank_lines=True).dropna() 
cutlet.info()
cutlet.describe()

cutlet.columns
cutlet.columns = "Unit_A", "Unit_B"

# Calculating the normality test
# Hypothesis
# Ho = Data is Normal
# Ha = Data is not Normal

stats.shapiro(cutlet.Unit_A)
# pvalue=0.3199
stats.shapiro(cutlet.Unit_B)
# pvalue=0.5225 > 0.05, P high Ho fly
# Data is Normal

# Variance test
# Ho = Variance are equal
# Ha = Variance are not equal

scipy.stats.levene(cutlet.Unit_A, cutlet.Unit_B)
# pvalue=0.4176 > 0.05, P high Ho fly
# Variance is equal

# 2 Sample T test
# Ho : Average diameter size of the cutlets are equal
# Ha : Average diameter size of the cutlets are not equal

scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B)
# pvalue=0.4722 > 0.05, P high Ho fly

scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B, alternative = 'greater')
# pvalue=0.2361 > 0.05, P high Ho fly

#Conclusion: There is no significant difference between cutlets produced by both the units.

#**************************************************
# 2. Hospital wants to determine the difference in average turn around time(TAT) among the different laboratories.
# Y is continuous and X(Laboratory_1, Laboratory_2, Laboratory_3, Laboratory_4) is continuous. alpha = 0.05

#import the data
lab = pd.read_csv(r"D:/360digi/DS/Sharath/Hypothesis_Testing_E_Learning/handson/Datasets_Hypothesis Testing/lab_tat_updated.csv")#, skip_blank_lines=True).dropna() 
lab.info()
lab.describe()

# Calculating the normality test
# Hypothesis
# Ho = Data is Normal
# Ha = Data is not Normal

stats.shapiro(lab.Laboratory_1)
# pvalue=0.4232 > 0.05
stats.shapiro(lab.Laboratory_2)
# pvalue=0.8637 > 0.05
stats.shapiro(lab.Laboratory_3)
# pvalue=0.0654 > 0.05
stats.shapiro(lab.Laboratory_4)
# pvalue=0.6619 > 0.05, P high Ho fly
# Data is Normal

# Variance test
# Ho = Variance are equal
# Ha = Variance are not equal
scipy.stats.levene(lab.Laboratory_1, lab.Laboratory_2, lab.Laboratory_3, lab.Laboratory_4)
# pvalue=0.3810 > 0.05, P high Ho fly
# Variance is equal

# One - Way Anova
# Ho: All the 4 labs have equal mean turn around time
# Ha: All the 4 labs have unequal mean turn around time
stats.f_oneway(lab.Laboratory_1, lab.Laboratory_2, lab.Laboratory_3, lab.Laboratory_4)
# pvalue=2.143740909435053e-58 < 0.05, P low Ho go

#Conclusion: All the 4 labs have unequal mean turn around time


#***********************************************************
# 3. Sales of products in 4 different regions is considered and determine if the male-female buyer ratios are similar across regions.

#import the data
sales = pd.read_csv(r"D:/360digi/DS/Sharath/Hypothesis_Testing_E_Learning/handson/Datasets_Hypothesis Testing/BuyerRatio.csv")#, skip_blank_lines=True).dropna() 
sales.info()
sales.describe()

sales.loc[0:2, ['Observed Values']] = [0, 1]

# Ho: All regions have equal proportions of product sales 
# Ha: Not all regions have equal proportions of product sales

Chisquares_results = scipy.stats.chi2_contingency(sales)
Chi_square = [['Test Statistic', 'p-value'], Chisquares_results[0], Chisquares_results[1]]
Chi_square

# pvalue=0.7919 > 0.05, P high Ho fly
# All the regions have equal proportion of sales from male and female.

# Conclusion: All proportions are equal.

#**********************************************************
# 4. Manager wants to determine if the customer form is defective or error free in all 4 centres along with 5% significance level.

#import the data
cust = pd.read_csv(r"D:/360digi/DS/Sharath/Hypothesis_Testing_E_Learning/handson/Datasets_Hypothesis Testing/CustomerOrderform.csv", skip_blank_lines=True).dropna()
cust.info()
cust.describe()
cust.columns



num_rows = len(cust.Phillippines);
for k,col_name in enumerate(cust.columns):
    col_df = pd.DataFrame(num_rows*[[0,col_name]],columns=['Form','Country'])
    col_df.loc[cust[col_name] == 'Defective','Form'] = 1
    
    if not k:
        fin_df = col_df
    else:
        fin_df = fin_df.append(col_df)

fin_df.reindex(drop=True)


count = pd.crosstab(fin_df["Form"], fin_df["Country"])#, cust["Malta"], aggfunc= 'sum' )
count 

# Ho: All regions have equal proportions of defective customer form. 
# Ha: Not all regions have equal proportions of defective customer form.


Chisquares_results = scipy.stats.chi2_contingency(count)
Chi_square = [['Test Statistic', 'p-value'], Chisquares_results[0], Chisquares_results[1]]
Chi_square
# pvalue=0.2771 > 0.05, P high Ho fly
# All regions have equal proportions of defective customer form.

# Conclusion: All proportions are equal.

#***************************************************************

# 5. Manager wants to determine the % of male vs female customer walking into the store differ based on day of the week.


#import the data
walkin = pd.read_csv(r"D:/360digi/DS/Sharath/Hypothesis_Testing_E_Learning/handson/Datasets_Hypothesis Testing/Fantaloons.csv", skip_blank_lines=True).dropna()
walkin.info()
walkin.describe()
walkin.columns


count = pd.crosstab(walkin["Weekdays"], walkin["Weekend"])#, cust["Malta"], aggfunc= 'sum' )
count 

# Ho: All days the store has equal proportions of customer walk-ins. 
# Ha: Not all days the store has equal proportions of customer walk-ins.

Chisquares_results = scipy.stats.chi2_contingency(count)
Chi_square = [['Test Statistic', 'p-value'], Chisquares_results[0], Chisquares_results[1]]
Chi_square

# pvalue=1.0 > 0.05, P high Ho fly
# All days the store has equal proportions of customer walk-ins.

# Conclusion: % of male vs female customer walking into the store is similar on all days of the week.
































