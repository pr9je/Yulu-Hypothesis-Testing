# import libararies
# Core Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, shapiro, levene, kruskal, chi2_contingency

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/content/bike_sharing.csv')
df