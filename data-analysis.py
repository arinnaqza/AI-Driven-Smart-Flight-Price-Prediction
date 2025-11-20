import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Set global display options for better visibility
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# --- 2.1 Initial Data Load & Inspection ---

def load_and_inspect_data(file_path):
    """Loads the dataset and performs initial structural inspection."""
    print("--- 1. LOADING DATA ---")
    try:
        # NOTE: Using the exact file path for the original, robust dataset
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    except FileNotFoundError:
        # Fallback for standard file naming conventions if the uploaded path is complex
        try:
            df = pd.read_csv(file_path.split('/')[-1])
            print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        except:
            print(f"ERROR: File not found. Please ensure the actual path to your CSV is correct.")
            return None

    print("\n--- 2. HEAD OF DATA (First 5 Rows) ---")
    print(df.head())

    print("\n--- 3. DATA INFORMATION (Types, Null Counts) ---")
    df.info()

    print("\n--- 4. DESCRIPTIVE STATISTICS ---")
    # For numerical columns like price
    print(df.describe().T)
    
    print("\n--- 5. DATASET COLUMNS (Actual Names) ---")
    print(df.columns.tolist())

    return df

# --- 2.2 Data Cleaning and Preprocessing ---

# Constant for conversion (used 2019 average)
INR_TO_KRW_RATE = 17 

def clean_data(df):
    """Handles missing values, cleans inconsistent data, and performs currency conversion."""
    if df is None:
        return None

    print("\n--- 6. DATA CLEANING & PREPROCESSING ---")

    # 0. Rename columns for clarity and future steps
    df.rename(columns={
        'stops': 'Total_Stops',
        'duration': 'Duration_Minutes', 
        'days_left': 'Days_Left' 
    }, inplace=True)
    print("Standardized key column names.")

    # 1. Handle Missing Values: Focus on 'Total_Stops' and 'price'
    if 'Total_Stops' in df.columns and 'price' in df.columns:
        df.dropna(subset=['Total_Stops', 'price'], inplace=True)
        print(f"Dropped rows with missing 'Total_Stops' or 'price'. Remaining rows: {len(df)}")
    
    # 2. Clean 'Total_Stops' Column (Convert text to numerical)
    if 'Total_Stops' in df.columns:
        stop_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4}
        df['Total_Stops'] = df['Total_Stops'].astype(str).str.lower().map(stop_mapping).fillna(df['Total_Stops'])
        df['Total_Stops'] = pd.to_numeric(df['Total_Stops'], errors='coerce').astype('Int64')
        df.dropna(subset=['Total_Stops'], inplace=True)
        print("Converted 'Total_Stops' from text ('zero') to numerical values (0, 1, 2, ...).")

    # 3. Ensure 'price' is numerical and drop any remaining nulls
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    df.dropna(inplace=True)
    
    # 4. Currency Conversion (INR to KRW)
    df['price'] = df['price'] * INR_TO_KRW_RATE
    print(f"**CONVERTED PRICE: Prices scaled by {INR_TO_KRW_RATE} (INR to KRW).**")
    
    print("\n--- 7. FINAL CLEANED DATA INFO ---")
    df.info()
    return df

# --- 3.1 Feature Engineering: Extract Temporal and Ordinal Features ---

def engineer_features(df):
    """Creates new features based on existing data for better model performance."""
    if df is None:
        return None
        
    print("\n--- 9. FEATURE ENGINEERING (Task 3.1) ---")
    
    # 1. Ordinal Encoding for 'class' (Economy vs. Business)
    class_mapping = {'Economy': 0, 'Business': 1}
    df['Class_Encoded'] = df['class'].map(class_mapping)
    print("Created 'Class_Encoded' (Ordinal: Economy=0, Business=1).")

    # 2. Create a combined Route Feature (Source + Destination)
    df['Route'] = df['source_city'] + ' to ' + df['destination_city']
    print("Created 'Route' feature.")

    # Drop original columns that are now redundant or less useful after cleaning/encoding
    df.drop(columns=['flight', 'index'], errors='ignore', inplace=True)
    
    print("Dropped redundant source columns ('flight', 'index').")
    print(f"New shape after feature engineering: {df.shape}")
    return df

# --- 3.2 Encoding Categorical Variables ---

def encode_categorical_features(df):
    """Applies One-Hot Encoding to nominal categorical variables."""
    if df is None:
        return None

    print("\n--- 10. ENCODING CATEGORICAL VARIABLES (Task 3.2) ---")

    # Define the columns to be One-Hot Encoded
    categorical_features = ['airline', 'departure_time', 'arrival_time', 'Route', 'source_city', 'destination_city', 'class']
    
    features_to_encode = [col for col in categorical_features if col in df.columns]

    if not features_to_encode:
        print("No categorical features found for encoding. Skipping OHE.")
        return df

    # Separate target variable (price) and features (X)
    X = df.drop('price', axis=1)
    y = df['price']

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_to_encode)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Fit and transform the data
    X_encoded = preprocessor.fit_transform(X)
    
    # Convert back to a DataFrame with meaningful column names
    feature_names = preprocessor.get_feature_names_out()
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)

    # Re-attach the target variable
    final_df = pd.concat([X_encoded_df, y], axis=1)
    
    print(f"Successfully applied One-Hot Encoding. New encoded features: {X_encoded_df.shape[1]}.")
    
    return final_df

# --- 3.3 Feature Scaling & Train-Test Split ---

def scale_and_split(df):
    """Scales numerical features and splits data into training and testing sets."""
    if df is None:
        return None, None, None, None

    print("\n--- 11. FEATURE SCALING & TRAIN-TEST SPLIT (Task 3.3) ---")

    # 1. Define the features (X) and the target (y)
    X = df.drop('price', axis=1)
    y = df['price']
    
    # 2. Identify Numerical Features for Scaling
    numerical_features = ['Duration_Minutes', 'Days_Left', 'Total_Stops'] 
    scaling_features = [col for col in numerical_features if col in X.columns]

    if scaling_features:
        scaler = StandardScaler()
        X[scaling_features] = scaler.fit_transform(X[scaling_features])
        print(f"Scaled numerical features: {scaling_features}")
    else:
        print("No numerical features found for scaling.")

    # 3. Perform Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining Set Size (80%): {X_train.shape[0]} samples")
    print(f"Testing Set Size (20%): {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# --- 2.3 Univariate/Bivariate Analysis (EDA) ---

def perform_eda(df):
    """Performs key visualizations to understand data distribution and relationships."""
    if df is None:
        return

    print("\n--- 8. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS) ---")
    
    # 1. Distribution of the Target Variable (Price)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=50)
    plt.title('Distribution of Flight Prices (KRW)') 
    plt.xlabel('Price (KRW)')
    plt.ylabel('Frequency')
    plt.savefig('eda_price_distribution_krw.png')
    plt.show()

    # 2. Key Bivariate Analysis: Price vs. Days Left (The Core Goal)
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='Days_Left', y='price', data=df, estimator='median', errorbar=None)
    plt.title('Median Flight Price (KRW) vs. Days Left to Departure (Optimal Booking Window)')
    plt.xlabel('Days Left to Departure')
    plt.ylabel('Median Price (KRW)')
    plt.gca().invert_xaxis()
    plt.savefig('eda_price_vs_days_left_krw.png')
    plt.show()
    
    # 3. Price vs. Airline
    plt.figure(figsize=(14, 6))
    sns.boxplot(y='price', x='airline', data=df.sort_values('price', ascending=False), palette="viridis")
    plt.title('Price Distribution by Airline (KRW)')
    plt.xlabel('Airline')
    plt.ylabel('Price (KRW)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_price_vs_airline_krw.png')
    plt.show()

    # 4. Price vs. Total Stops
    plt.figure(figsize=(8, 6))
    sns.boxplot(y='price', x='Total_Stops', data=df.sort_values('Total_Stops'))
    plt.title('Price Distribution by Total Stops (KRW)')
    plt.xlabel('Total Stops (0 = non-stop)')
    plt.ylabel('Price (KRW)')
    plt.savefig('eda_price_vs_stops_krw.png')
    plt.show()
    
    print("\nVisualizations complete. Check your project folder for the saved PNG files.")


# --- MAIN EXECUTION ---

# Define the uploaded file path
FILE_NAME = 'airlines_flights_data.csv'

# 1. Load data (Task 2.1)
data_raw = load_and_inspect_data(FILE_NAME)

# 2. Clean data and convert currency (Task 2.2)
data_cleaned = clean_data(data_raw)

if data_cleaned is not None:
    # 3. Feature Engineering (Task 3.1) 
    data_processed = engineer_features(data_cleaned)
    
    # 4. Perform EDA (Task 2.3) - Runs BEFORE encoding
    perform_eda(data_processed)
    
    # 5. Perform Encoding (Task 3.2)
    data_encoded = encode_categorical_features(data_processed)
    
    # 6. Perform Scaling and Splitting (Task 3.3)
    X_train, X_test, y_train, y_test = scale_and_split(data_encoded)
    
    # 7. Save the split datasets (KRW) for Phase 4
    if X_train is not None:
        X_train.to_csv('X_train_krw.csv', index=False)
        X_test.to_csv('X_test_krw.csv', index=False)
        y_train.to_csv('y_train_krw.csv', index=False)
        y_test.to_csv('y_test_krw.csv', index=False)
        print("\nTraining and Testing sets saved to separate KRW CSV files.")
    
    print("\nData preparation and preprocessing successfully completed!")