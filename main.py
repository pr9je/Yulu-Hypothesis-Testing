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

# Data Overview
#Shape of the dataset
print('Number of rows: ', df.shape[0])
print('Number of columns: ', df.shape[1])

# Data types of all attributes
df.info()

# Data type correction & Feature Engineering

# Conver datetime column to proper datetime dtype.
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract useful time-based features for EDA
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.day_name()

# Human-readable leabels for seasons & weather (used only for plotting/ readability)
season_map = {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}
weather_map = {1: 'clear', 2: 'Mist + Cloudly', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}
df['season_label'] = df['season'].map(season_map)
df['weather_label'] = df['weather'].map(weather_map)

# Convert nominal/ordinal-coded integer columns to 'category' dtype
categorical_cols = ['season', 'holiday', 'workingday', 'weather']
for col in categorical_cols:
  df[col] = df[col].astype('category')

numerical_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

print('Categorical Columns:', categorical_cols)
print('Numerical Columns:', numerical_cols)
df.dtypes

# Missing value detection
df.isnull().sum()

# Duplicate records
print("Number of duplicates rows: ", df.duplicated().sum())
df = df.drop_duplicates()
print('Shape after dropping duplicates: ', df.shape)

# Statistical Summary
df['numerical_cols'].describe().T

df[categorical_cols].describe()

# Univariate Analysis
# Distribution of Continoue Variables
fig, axes = plt.subplots(3,3, figsize=(16,12))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
  sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
  axes[i].set_title(f'Distribution of {col}')
for j in range(len(numerical_cols), len(axes)):
  fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Outlier Check (Boxplots + IQR)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
  sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
  axes[i].set_title(f'Boxplot: {col}')
for j in range(len(numerical_cols), len(axes)):
  fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

def get_outliers_bounds(series):
  Q1 = series.quantile(0.25)
  Q3 = series.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return lower_bound, upper_bound

print(f"{'Column':<12}{'Lower':>10}{'Upper':>10}{'#Ouliters':>14}{'% of data':>12}")
for col in numerical_cols:
  lower, upper = get_outliers_bounds(df[col])
  n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
  pct = 100 * n_outliers / df.shape[0]
  print(f"{col:<12}{lower:>10.2f}{upper:>10.2f}{n_outliers:>14}{pct:>12.2f}")

# Outlier treatment: Clip only windspeed (likely-erroneous extreme readings)
# count/ casual / registered outliers are retained since they represent real demans spikes, not data errors, and removing them would bias the hypothesis tests that follow.
lower_ws, upper_ws = get_outliers_bounds(df['windspeed'])
df['windspeed'] = df['windspeed'].clip(lower=lower_ws, upper=upper_ws)
print(f"Windspeed clipped to [{lower_ws:.2f},{upper_ws:.2f}]")
