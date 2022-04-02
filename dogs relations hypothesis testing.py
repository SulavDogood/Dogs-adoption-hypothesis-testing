# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:59:39 2022

@author: Sulav
"""

import numpy as np
import pandas as pd


# Import data
dogs = pd.read_csv('dog_data.csv')

# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

print(dogs)

# Finding if whippets are significantly more or less likely than other dogs to be a rescue.
whippet_rescue = dogs.is_rescue[dogs.breed == "whippet"]

num_whippet_rescues = np.sum(whippet_rescue == 1)
print(num_whippet_rescues)

num_whippets = len(whippet_rescue)
print(num_whippets)

# Hypothesis testing
#Null: 8% of whippets are rescues
#Alternative: more or less than 8% of whippets are rescues
from scipy.stats import binom_test
p_value = binom_test(num_whippet_rescues, num_whippets, 0.08 )

print(p_value)

# here we find the p_value to be 0.58 which is greater than the significance threshold of 0.05 so we accept our null hypothesis that around 8% of whippets are rescues.

wt_whippets = dogs.weight[dogs.breed =="whippet"]
wt_terriers = dogs.weight[dogs.breed =="terrier"]
wt_pitbulls = dogs.weight[dogs.breed =="pitbull"]

# Hypothesis testing:
#Null: whippets, terriers, and pitbulls all weigh the same amount on average
#Alternative: whippets, terriers, and pitbulls do not all weigh the same amount on average (at least one pair of breeds has differing average weights)
from scipy.stats import f_oneway
fstat, pval = f_oneway(wt_whippets, wt_terriers, wt_pitbulls)
print(pval)
# Since, pval is less than 0.05 we can say that there is at least one pair of dog breeds that have significantly different average weights.
# finding which of those breeds weigh different than average.
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_results = pairwise_tukeyhsd(dogs_wtp.weight,dogs_wtp.breed, 0.05)
print(tukey_results)
# we find the average weight of pitbull and whippet pair is significantly different.

# Finding if poodle and shihtzus come in different colors.
dogs_ps = dogs[dogs.breed.isin(["poodle","shihtzu"])]
table = pd.crosstab(dogs_ps.breed, dogs_ps.color)
print(table)
# Hypothesis testing
#Null: There is an association between breed (poodle vs. shihtzu) and color.
#Alternative: There is not an association between breed (poodle vs. shihtzu) and color.
from scipy.stats import chi2_contingency
chi2, pval, dof, expected = chi2_contingency(table)
print(pval)
#snce pval is less than 0.05 we reject the null hypothesis therefore there is not an association between pitbull's and shihtzu's colors.