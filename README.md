# AI-Driven Smart Flight Price Prediction

## Project Description ✈️ 

Flight tickets are one of the biggest financial concerns for international students who frequently travel between their home country and their study destination. Because airfare prices fluctuate due to multiple factors such as season, airline, and and departure schedule, students often struggle to find the best deals. This project aims to apply machine learning techniques to predict the fluctuations of flight ticket prices.

Using publicly available datasets on Kaggle, [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction), the project will analyse how features like airline, number of stops, duration, departure and arrival time influence ticket costs. 

Unlike existing platforms such as Skyscanner that displays real-time ticket prices, this project focuses on understanding the underlying price patterns and forecasting future costs trends based on historical data. The results are expected to help international students identify the most cost-effective times to book flights and make more informed travel plans.

## Team Members

| Name | Department | University | Email |
| :----: | :------: | :----: | :----: |
| Arinna Qaisara | Information Systems | Hanyang University | arinnaqza@gmail.com |
| Nur Sabrina | Information Systems | Hanyang University | sabrinarramly@gmail.com |
| Winnie Eslyn | Information Systems | Hanyang University | winnieeslyn@gmail.com |

## I. Introduction

The motivation behind this project is to analyse real flight data and understand how ticket prices change as the departure date approaches. Instead of predicting exact prices, our goal is to uncover patterns between the number of days left before departure and the cost of the flight. By studying this relationship, we aim to identify which time windows typically offer cheaper prices and which periods see price spikes.

By the end of the project, we want to:

- Determine the optimal time frame to book flights based on historical patterns
- Visualize how prices behave as the departure date gets closer
- Provide insights that help international students make smarter booking decisions
- Build a machine learning model that learns from these patterns and predicts expected price trends

This project ultimately aims to offer practical, data-driven guidance that helps students save money and plan their travels more effectively.

## II. Datasets

A. Data Source and Context

The dataset we used for this project was derived from the [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) collection available on Kaggle. This data captures essential factors influencing airline pricing, such as route, date, duration, and number of stops. A critical preliminary step in our data preparation was the conversion of all original ticket prices from Indian Rupees (INR) to our required currency, Korean Won (KRW). This sets the financial scale for all our model evaluations.

B. Original Features and Target

The raw dataset contained key categorical and temporal information:

| Feature Name | Description | Data Type (Original) |  |
| :----: | :------: | :----: | :----: |
| Price | The ticket fare | Numerical (Target) |
| Airline | Name of the airline carrier. | Categorical | 
| Date_of_Journey | Date of flight departure | Date/Time |
| Source / Destination | Departure and arrival cities | Categorical |
| Duration | Total flight time (e.g., "2h 50m") | String/Duration | 
| Total_Stops | Number of layovers | Categorical/String |

C. Final Modeling Data Structure

The CSV files used to train our models (`X_train_krw.csv`, etc.) represent the data after extensive preprocessing and feature engineering. Our preparation steps included:

1. Temporal Feature Extraction: We engineered the dates and times into crucial numerical features, such as Days Left before departure, and separate features for the Departure Hour and Minute.

2. Duration Conversion: We converted the raw duration string into a single, continuous numerical feature: Total Duration in Minutes.

3. Categorical Encoding: We converted all non-numerical features (like Airline, Source, and Destination) into a machine-readable format using One-Hot Encoding.

The resulting X data matrices contained these cleaned, engineered features, while the y data held the corresponding ticket prices in KRW.

## III. Methodology
This project evaluates different machine learning models to predict airline ticket prices in KRW. The methodology follows a structured workflow that begins with baseline modeling, progresses into advanced tree boosting models, and ends with model comparison and selection.

1. Data Loading and Preparation
All features and labels were preprocessed beforehand and exported as:

- `X_train_krw.csv` 
- `X_test_krw.csv`
- `y_train_krw.csv`
- `y_test_krw.csv`

The `load_data()`  function loads these datasets and ensures that the data structure matches scikit-learn’s expectations. Both training and testing labels were flattened using `.squeeze()` for compatibility with regressors.

2. Model Evaluation Strategy
We evaluate the performance consistently across all models using the test dataset (`X_test`, `y_test`) and two primary regression metrics:

Mean Absolute Error (MAE): The key business metric, representing the average error magnitude, measured in Korean Won (KRW). It is less sensitive to outliers than Mean Squared Error (MSE).

R-squared $$(R^2)$$ Score: A statistical measure representing the proportion of the variance in the target variable that is predictable from the features.

3. Modeling Workflow: Three Stages
Our analysis proceeds through three distinct modeling stages to establish a benchmark and find the optimal predictor.

A. Stage 1: Baseline Model (Linear Regression)

We first implement a Linear Regression model using `task_4_1_baseline_model`.

Goal: The primary goal is to establish a simple, interpretable performance floor, a baseline MAE against which all advanced models must compete.

Implementation: We train the standard `sklearn.linear_model.LinearRegression` on the training set.

B. Stage 2: Initial Advanced Model 

The core of our advanced modeling involves using a powerful Gradient Boosting Machine (GBM). We explored two versions of this stage:

Version 1: LightGBM (LGBM)

Model: `lightgbm.LGBMRegressor`.

- Goal: We train a fast, efficient GBM model using solid initial parameters.

- Initial Parameters: We configure the model to optimize directly for MAE (`objective='regression_l1'`, `metric='mae'`). We also incorporate ensemble techniques like feature subsampling (`feature_fraction=0.8`) and sample subsampling (`bagging_fraction=0.8`) to manage overfitting.

Version 2: XGBoost

Model: `xgboost.XGBRegressor`.

- Initial Parameters: We configure this model for a standard squared error objective (`objective='reg:squarederror'`) and use similar subsampling controls (`colsample_bytree=0.8`, `subsample=0.8`).

C. Stage 3: Final Tuned Champion Model

In this stage, we apply an optimized set of hyperparameters to the final champion model to maximize its predictive power.

Tuning Methods

LightGBM Tuning: We load the best hyperparameters (e.g., increased `n_estimators`, `optimized learning_rate`, specific `num_leaves`) determined from a previous Randomized Search Cross-Validation (`RandomizedSearchCV`) process.

XGBoost Tuning: We implement the optimized settings by translating the successful LightGBM parameters into corresponding XGBoost parameters (e.g., deeper `max_depth=10`, faster `learning_rate=0.1`). This acts as an educated guess for achieving near-optimal performance in the XGBoost architecture.



## IV. Evaluation & Analysis

## V. Related Work
