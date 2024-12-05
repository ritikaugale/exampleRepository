import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('/tmp/dataset01.csv')

# Task 1: Determine and print the number of data entries in column 'y'
n1 = data['y'].count()
print(f"Number of data entries in 'y': {n1}")

# Task 2: Determine and print the mean of column 'y'
mean_y = data['y'].mean()
print(f"Mean of 'y': {mean_y}")

# Task 3: Determine and print the standard deviation of column 'y'
stddev_y = data['y'].std()
print(f"Standard deviation of 'y': {stddev_y}")

# Task 4: Determine and print the variance of column 'y'
variance_y = data['y'].var()
print(f"Variance of 'y': {variance_y}")

# Task 5: Determine and print the min and max of column 'y'
min_y = data['y'].min()
max_y = data['y'].max()
print(f"Min of 'y': {min_y}, Max of 'y': {max_y}")

# Task 6: Perform OLS regression
X = data[['x']]  
X = sm.add_constant(X)  
y = data['y']  

model = sm.OLS(y, X).fit()  
print(model.summary())  

with open('/tmp/OLS_model', 'w') as f:
    f.write(model.summary().as_text())
