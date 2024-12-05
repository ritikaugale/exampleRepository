import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import pickle




#Data Cleaning
def clean_data(file_path):
    df = pd.read_csv(file_path)

    df.columns = ["x", "y"]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)  # Drop rows with missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    
    # Handle outliers using Z-score
    for column in df.select_dtypes(include=np.number).columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        df = df[(z_scores.abs() < 2)]
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

       # Normalize numeric columns
    for column in df.select_dtypes(include=np.number).columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaned and saved to 'cleaned_data.csv'")
    return df

def train_models(cleaned_data):
    X = cleaned_data.iloc[:, :-1].values  # Features
    y = cleaned_data.iloc[:, -1].values  # Target

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train OLS model
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    y_pred_ols = ols_model.predict(X_test)
    print(f"OLS Model - MSE: {mean_squared_error(y_test, y_pred_ols)}, R2: {r2_score(y_test, y_pred_ols)}")

    # Train ANN model
    input_size = X_train.shape[1]
    ann = buildNetwork(input_size, 5, 1)  # Simple feedforward network
    ds = SupervisedDataSet(input_size, 1)
    for i in range(len(X_train)):
        ds.addSample(X_train[i], y_train[i])
    trainer = BackpropTrainer(ann, ds)
    trainer.trainUntilConvergence(maxEpochs=100)

    # Save ANN model
    with open("UE_05_App3_ANN_Model.pkl", "wb") as file:
        pickle.dump(ann, file)
    print("ANN model trained and saved.")
    return ols_model, ann, X_test, y_test

def visualize_models(ols_model, ann, X_test, y_test):
    y_pred_ols = ols_model.predict(X_test)
    y_pred_ann = np.array([ann.activate(x) for x in X_test]).flatten()

    plt.figure(figsize=(10, 5))

    # Plot OLS predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_ols, alpha=0.5, label="OLS Predictions")
    plt.plot(y_test, y_test, color="red", label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.savefig('OLS.pdf')
    plt.title("OLS Model Performance")
    plt.legend()
    plt.show()
    plt.close()

    # Plot ANN predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_ann, alpha=0.5, label="ANN Predictions")
    plt.plot(y_test, y_test, color="red", label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.savefig('ANN.pdf')
    plt.title("ANN Model Performance")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

# Step 5: Quantitative comparison
def compare_models(ols_model, ann, X_test, y_test):
    y_pred_ols = ols_model.predict(X_test)
    y_pred_ann = np.array([ann.activate(x) for x in X_test]).flatten()

    ols_mse = mean_squared_error(y_test, y_pred_ols)
    ols_r2 = r2_score(y_test, y_pred_ols)

    ann_mse = mean_squared_error(y_test, y_pred_ann)
    ann_r2 = r2_score(y_test, y_pred_ann)
    print("Model Comparison:")
    print(f"OLS Model - MSE: {ols_mse}, R2: {ols_r2}")
    print(f"ANN Model - MSE: {ann_mse}, R2: {ann_r2}")

if __name__ == "__main__":
    df_cleaned = clean_data("UE_06_dataset04_joint_scraped_data.csv")
    ols_model, ann, X_test, y_test = train_models(df_cleaned)
    visualize_models(ols_model, ann, X_test, y_test)
    compare_models(ols_model, ann, X_test, y_test)