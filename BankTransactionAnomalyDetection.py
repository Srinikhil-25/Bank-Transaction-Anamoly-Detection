import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class BankTransactionAnomalyDetection:

    def __init__(self, file):
        if not os.path.exists(file):
            print("ERROR: File not found!")
            print("Place 'bank_transactions.csv' in the same folder.")
            exit()
        
        self.df = pd.read_csv(file, sep='\t')
        print("File loaded successfully!")

    def preprocess(self):
        print("\nMissing Values:\n", self.df.isnull().sum())

    def statistical_analysis(self):
        self.mean = self.df['Transaction_Amount'].mean()
        self.std = self.df['Transaction_Amount'].std()

        print("\nMean:", self.mean)
        print("Standard Deviation:", self.std)

    def detect_anomalies(self):
        self.df['Z_score'] = (self.df['Transaction_Amount'] - self.mean) / self.std

        self.df['Anomaly'] = self.df['Z_score'].apply(
            lambda x: 'Anomaly' if abs(x) > 2 else 'Normal'
        )

        print("\nDetected Transactions:")
        print(self.df[['Transaction_ID','Transaction_Amount','Z_score','Anomaly']])

    def visualize(self):

        plt.figure()
        plt.hist(self.df['Transaction_Amount'], bins=10)
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Transaction Amount")
        plt.ylabel("Frequency")
        plt.show()

        plt.figure()
        plt.boxplot(self.df['Transaction_Amount'])
        plt.title("Box Plot of Transaction Amount")
        plt.ylabel("Amount")
        plt.show()

        plt.figure()
        plt.scatter(self.df.index, self.df['Transaction_Amount'])
        plt.title("Transaction Amount Scatter Plot")
        plt.xlabel("Transaction Index")
        plt.ylabel("Transaction Amount")
        plt.show()

        plt.figure(figsize=(10,8))

        plt.subplot(2,2,1)
        plt.hist(self.df['Transaction_Amount'], bins=10)
        plt.title("Histogram")

        plt.subplot(2,2,2)
        plt.hist(self.df['Z_score'], bins=10)
        plt.title("Z-score Distribution")

        plt.subplot(2,2,3)
        plt.scatter(self.df.index, self.df['Transaction_Amount'])
        plt.title("Scatter Plot")

        plt.subplot(2,2,4)
        plt.boxplot(self.df['Transaction_Amount'])
        plt.title("Box Plot")

        plt.tight_layout()
        plt.show()


# Main
file_name = "bank_transactions.csv"

obj = BankTransactionAnomalyDetection(file_name)
obj.preprocess()
obj.statistical_analysis()
obj.detect_anomalies()
obj.visualize()