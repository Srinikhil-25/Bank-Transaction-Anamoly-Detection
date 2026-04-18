import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(file):
    print("Loading dataset...")
    df = pd.read_csv(file)

    # Clean column names (FIX for your error)
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully!\n")
    print("Columns in dataset:", df.columns.tolist(), "\n")

    return df

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess_data(df):
    print("Preprocessing data...\n")

    df = df.copy()

    # Remove missing values
    df.dropna(inplace=True)

    # Convert date column safely
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract hour from time
    df['Hour'] = df['Time'].str.split(':').str[0].astype(int)

    print("Preprocessing completed!\n")
    return df

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
def detect_anomalies(df):

    print("Applying statistical methods...\n")

    # Z-score
    df['z_score'] = (df['Transaction Amount'] - df['Transaction Amount'].mean()) / df['Transaction Amount'].std()

    # IQR
    Q1 = df['Transaction Amount'].quantile(0.25)
    Q3 = df['Transaction Amount'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Final anomaly flag
    df['Anomaly'] = (
        (df['Transaction Amount'] > upper) |
        (df['Transaction Amount'] < lower) |
        (abs(df['z_score']) > 3)
    ).astype(int)

    print("Detection completed!\n")
    return df

# -------------------------------
# SHOW RESULTS
# -------------------------------
def show_results(df):

    anomalies = df[df['Anomaly'] == 1]

    print("Detected Anomalies:\n")

    if anomalies.empty:
        print("No anomalies found.\n")
    else:
        print(anomalies[['Transaction ID','Transaction Amount','Location']])
        print("\nTotal anomalies:", len(anomalies))

# -------------------------------
# VISUALIZATION
# -------------------------------
def visualize(df):

    # Histogram
    plt.figure()
    plt.hist(df['Transaction Amount'])
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()

    # Boxplot
    plt.figure()
    plt.boxplot(df['Transaction Amount'])
    plt.title("Outlier Detection (Boxplot)")
    plt.show()

    # Time analysis
    plt.figure()
    df.groupby('Hour')['Transaction Amount'].count().plot()
    plt.title("Transactions by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.show()

# -------------------------------
# MAIN
# -------------------------------
def main():

    df = load_data("transactions.csv")

    df = preprocess_data(df)

    df = detect_anomalies(df)

    show_results(df)

    visualize(df)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()