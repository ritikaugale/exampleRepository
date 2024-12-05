import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS, add_constant
import statsmodels.formula.api as smf
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic

d1 = pd.read_csv('dataset02.csv')

d1.describe()
print(d1.shape)

data_tonumeric = d1.apply(pd.to_numeric, errors='coerce') 
print(data_tonumeric.shape)


data_tonumeric.isnull().sum()
final_data = data_tonumeric.dropna()
print(final_data.shape)

# Outlier Removal using Z-score filter
z_scores = np.abs(zscore(final_data))
final_data = final_data[(z_scores < 3).all(axis=1)]


# Data Normalization
final_data = (final_data - final_data.min()) / (final_data.max() - final_data.min())

Q1 = final_data.quantile(0.25)
Q3 = final_data.quantile(0.75)
IQR = Q3 - Q1
final_data = final_data[~((final_data < (Q1 - 1.5 * IQR)) | (final_data > (Q3 + 1.5 * IQR))).any(axis=1)]

final_data = (final_data - final_data.min()) / (final_data.max() - final_data.min())

training_data = final_data.sample(frac=0.8, random_state=42)
training_data.to_csv('/tmp/dataset02_training.csv', index=False)

testing_data = final_data.drop(training_data.index)
testing_data.to_csv('/tmp/dataset02_testing.csv', index=False)

plt.scatter(training_data['x'], training_data['y'], color='blue', label='Training Data')
plt.scatter(testing_data['x'], testing_data['y'], color='orange', label='Testing Data')
plt.plot(training_data['x'], sm.OLS(training_data['y'], sm.add_constant(training_data[['x']])).fit().predict(sm.add_constant(training_data[['x']])), color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.legend()
plt.savefig('UE_04_App2_ScatterVisualizationAndOLSModel.pdf')
plt.show()



# Assuming 'x' is the feature and 'y' is the target column
X = add_constant(final_data[['x']])  # Feature(s) with intercept
y = final_data['y']  # Target

# Fit OLS model
ols_model = smf.ols('y ~ x', data=final_data).fit()

#Create a Box Plot
plt.figure(figsize=(10, 6))
final_data.boxplot()
plt.title("Box Plot of All Dimensions")
plt.savefig('UE_04_App2_BoxPlot.pdf')
plt.show()

# Initialize the LinearRegDiagnostic class
diagnostic = LinearRegDiagnostic(ols_model)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
vif_table, fig, ax = diagnostic(plot_context='seaborn-talk')
fig.savefig('UE_04_App2_DiagnosticPlots.pdf')
print(vif_table)