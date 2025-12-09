# AI-Driven Smart Flight Price Prediction

## Project Description ✈️ 

### [Video Presentation](https://youtu.be/oU3fG_jbkgE)

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

The dataset we used for this project was derived from the [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) collection available on Kaggle. 

<img width="1021" height="651" alt="Screenshot (158)" src="https://github.com/user-attachments/assets/f25cd4c8-e109-4ac2-a813-c883816bd938" />

This data captures essential factors influencing airline pricing, such as route, date, duration, and number of stops. A critical preliminary step in our data preparation was the conversion of all original ticket prices from Indian Rupees (INR) to our required currency, Korean Won (KRW). This sets the financial scale for all our model evaluations.

B. Original Features and Target

The raw dataset contained key categorical and temporal information:

| Feature Name | Description | Data Type (Original) |
| :---: | :---: | :---: |
| Price | The ticket fare | Numerical (Target) |
| Airline | Name of the airline carrier | Categorical |
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

## III. Exploratory Data Analysis

Before training our machine learning models, we performed Exploratory Data Analysis (EDA) to better understand the structure of the dataset and uncover key patterns related to flight pricing. This step is crucial for identifying influential features, visualizing trends, and validating the assumptions behind our modelling approach.

1. Distribution of Flight Prices (KRW)
   
   <img width="1000" height="600" alt="eda_price_distribution_krw" src="https://github.com/user-attachments/assets/d83d50dd-ed0f-4a53-a269-e833c95b8e1a" />

   The distribution of ticket prices (converted from INR to KRW) is highly right-skewed:

    - Many flights fall in the lower price range (under ~200,000 KRW)
    - A long tail extends toward expensive flights (up to 2,000,000 KRW)
    - Significant outliers exist, indicating large price variability
    - Multiple peaks suggest mixed flight categories (low-cost vs premium carriers)

    This skewed distribution justifies the need for non-linear models such as XGBoost or LightGBM, which handle irregular patterns better than simple linear models.
   
2. Median Flight Price vs. Days Left to Departure
   
   <img width="1200" height="700" alt="eda_price_vs_days_left_krw" src="https://github.com/user-attachments/assets/d6201a74-42a7-4a88-859c-1d19335510da" />

    This plot directly addresses the core research question: **"When is the best time to book a flight?"
**
   Key observations:

   - Prices are stable and lowest between 20-40 days before departure
   - Booking too early (>40-50 days out) shows irregular fluctuations
   - Prices rise dramatically in the final 10 days
   - Last-minute booking (0-5 days) shows the highest, most unstable prices

    The optimal booking window appears to be 20-40 days before departure, offering the most consistent and afforadable prices.

3. Price Distribution by Airline
   
   <img width="1400" height="600" alt="eda_price_vs_airline_krw" src="https://github.com/user-attachments/assets/e1f4722a-1885-461d-800e-f22274382e62" />

    The boxplot reveals strong differences in airline pricing:

   - Vistara and Air India show significantly higher and more variable ticket prices
   - SpiceJet, Go First, IndiGo, and AirAsia remain mostly within lower price ranges
   - Premium airlines have wider boxes and longer whiskers, indicating higher volatility
   - Budget airlines show tighter distributions, reflecting more predictable pricing
     
4. Price Distribution by Total Stops

   <img width="800" height="600" alt="eda_price_vs_stops_krw" src="https://github.com/user-attachments/assets/64e498d2-c58e-4313-a9e3-f83f6328142a" />
    
    Comparing 0-stop vs 1-stop flights:

   - 0-stop flights have lower median prices and smaller variation
   - 1-stop flights have dramatically wider price spread, sometimes exceeding 2,000,000 KRW
   - Some multi-stop routes are unexpectedly expensive, likely due to route length, layover duration, or airline combinations.

## IV. Methodology
This project evaluates different machine learning models to predict airline ticket prices in KRW. The methodology follows a structured workflow that begins with baseline modeling, progresses into advanced tree boosting models, and ends with model comparison and selection.

**1. Data Loading and Preparation**

All features and labels were preprocessed beforehand and exported as:

- `X_train_krw.csv` 
- `X_test_krw.csv`
- `y_train_krw.csv`
- `y_test_krw.csv`

```
def load_data():
    X_train = pd.read_csv('X_train_krw.csv')
    X_test = pd.read_csv('X_test_krw.csv')
    y_train = pd.read_csv('y_train_krw.csv').squeeze()
    y_test = pd.read_csv('y_test_krw.csv').squeeze()
    return X_train, X_test, y_train, y_test
```

The `load_data()`  function loads these datasets and ensures that the data structure matches scikit-learn’s expectations. Both training and testing labels were flattened using `.squeeze()` for compatibility with regressors.

**2. Model Evaluation Strategy**

We evaluate the performance consistently across all models using the test dataset (`X_test`, `y_test`) and two primary regression metrics:

Mean Absolute Error (MAE): The key business metric, representing the average error magnitude, measured in Korean Won (KRW). It is less sensitive to outliers than Mean Squared Error (MSE).

R-squared $$(R^2)$$ Score: A statistical measure representing the proportion of the variance in the target variable that is predictable from the features.

**3. Modeling Workflow: Three Stages**

Our analysis proceeds through three distinct modeling stages to establish a benchmark and find the optimal predictor.

**A. Stage 1: Baseline Model (Linear Regression)**

We establish the performance floor using the simplest predictive model: **Linear Regression**. 

This model assumes a straight-line relationship between features and price. It serves as our benchmark; if our advanced models cannot beat this simple approach, we know our feature engineering strategy needs revision.

```
def task_4_1_baseline_model(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model, mae, r2
```

**B. Stage 2: Initial Advanced Model**

We then introduce a powerful Gradient Boosting Machine (GBM) to capture non-linearities in the data. We use specific hyperparameters to control the model's complexity:

`n_estimators=1000`: We set the number of trees to 1000 to ensure the model has enough iterations to learn complex patterns.

`learning_rate=0.05`: We use a lower learning rate to slow down the training, which acts as a regularization technique to prevent the model from overfitting to the training data.

**Version 1: LightGBM (LGBM)**

Model: `lightgbm.LGBMRegressor`.

For LightGBM, we explicitly set the objective to regression_l1. This forces the model to optimize for Mean Absolute Error (MAE) directly, rather than the standard Mean Squared Error. Since MAE is our primary business metric, this alignment often yields better practical results..

- Initial Parameters: We configure the model to optimize directly for MAE (`objective='regression_l1'`, `metric='mae'`). We also incorporate ensemble techniques like feature subsampling (`feature_fraction=0.8`) and sample subsampling (`bagging_fraction=0.8`) to manage overfitting.

```
lgbm_params = {
    'objective': 'regression_l1',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train, y_train)
```

**Version 2: XGBoost**

Model: `xgboost.XGBRegressor`.

For XGBoost, we use the standard reg:squarederror objective. We introduce randomness using colsample_bytree (using only 80% of features per tree) and subsample (using only 80% of data rows per tree). This randomness makes the model more robust and less likely to memorize noise in the dataset.

- Initial Parameters: We configure this model for a standard squared error objective (`objective='reg:squarederror'`) and use similar subsampling controls (`colsample_bytree=0.8`, `subsample=0.8`).

```
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)

```

**C. Stage 3: Final Tuned Champion Model**

In this stage, we apply an optimized set of hyperparameters to the final champion model to maximize its predictive power.

Tuning Methods

LightGBM Tuning: We load the best hyperparameters (e.g., increased `n_estimators`, `optimized learning_rate`, specific `num_leaves`) determined from a previous Randomized Search Cross-Validation (`RandomizedSearchCV`) process.

XGBoost Tuning: We implement the optimized settings by translating the successful LightGBM parameters into corresponding XGBoost parameters (e.g., deeper `max_depth=10`, faster `learning_rate=0.1`). This acts as an educated guess for achieving near-optimal performance in the XGBoost architecture.

## V. Model Evaluation Strategy and Analysis

**A. Evaluation Strategy**

The `evaluate_model` function calculates our key metrics. In the print statement, we use the f-string formatting `:,.2f` for the MAE. This adds comma separators (e.g., 10,000) and fixes the decimal precision to two places, ensuring the currency output is readable and professional.

```
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:,.2f} KRW")
    return mae, r2
```

- Mean Absolute Error (MAE): Our primary metric, measured in KRW. We focus on minimizing this value.

- R-squared ($$R^2$$) Score: Used to ensure the model explains a high proportion of the variability in price. We aim to maximize this score toward 1.0.

**B. Performance Analysis**

1. Baseline Assessment: The MAE from the Linear Regression model sets the initial benchmark. Any significant price prediction model must achieve an MAE substantially lower than this baseline.

2. Advanced Model Impact: When introducing the initial GBM, we anticipate a sharp drop in MAE and a steep rise in $$R^2$$, validating our choice of a non-linear GBM architecture.

3. Tuning Effectiveness: The Final Tuned Champion Model is expected to deliver the lowest MAE of all three stages. The reduction in MAE from the initial GBM to the final tuned GBM quantifies the direct value of our optimization efforts.

**C. Final Model Selection**

Our final model selection is based on the single most important metric: the model that produces the lowest Mean Absolute Error (MAE) on the unseen test set is declared our champion, as it consistently provides the most accurate and reliable flight price predictions in KRW.

## VI. Streamlit Web Application

To make our model accessible and easy to use, we deployed a simple interactive web application using Streamlit. The app allows users to input key flight details, such as airline, number of stops, class, flight duration, and days left before departure, and instantly receive a price prediction in KRW.

This web interface demonstrates how the machine learning model can be used in a real-world scenario, helping users estimate whether their intended booking date is likely to be affordable or expensive based on historical pricing patterns.

You can try the live app here:
🔗 https://ai-driven-smart-flight-price-prediction-dhun4sch2aytpntbyx9kfe.streamlit.app/

## VII. Limitations

**1. Dataset Contains Only Indian Domestic Flights**
   
Our model is trained on Indian domestic flight data because there is no publicly available dataset for Korea–Malaysia or other international routes that contain the exact data we needed for our analysis.
This means the pricing patterns may not perfectly match Korean or Southeast Asian airline behavior, even though general trends are similar.

**2. Limited Features Compared to Real Airline Pricing Systems**
   
Actual airline pricing considers many additional factors such as:

- real-time demand
- fuel cost changes
- weather/seasons
- dynamic pricing algorithms
- promotions or seasonal events

Because our dataset doesn’t include these variables, the predictions can only capture broad patterns, not exact real-world price behavior.

**3. Historical Data, Not Real-Time Market Data**

Ideally, we wanted to collect our own real-time flight prices using web scraping. However, airline ticket prices change daily, and collecting meaningful time-series data would require at least 30 days of continuous scraping. Because this was not feasible within our project timeline, we relied on a publicly available Kaggle dataset instead. As a result, our model captures historical patterns but cannot reflect current market fluctuations or sudden price changes.

## VIII. Related Work 

A relevant study to our project is ["Large-Scale Airline Ticket Price Prediction Using Ensemble Machine Learning Algorithm" by Muzaffer Ertürk, Murat Emeç, Ayşe Atılgan Sarıdoğan, and Nabi Küçükgergerli (2025)](https://dergipark.org.tr/tr/download/article-file/4957193). The authors used a large dataset of more than 300,000 flights and compared several machine learning models, including Linear Regression, Random Forest, LightGBM, CatBoost, and XGBoost.

Their result shows that ensemble models, especially XGBoost, achieve the highest prediction accuracy. This research is highly relevant to our project because we also use similar features and apply a boosting model (XGBoost). While the paper focuses on large-scale price prediction, our project adapts the same techniques to help international students understand the best time to book affordable flight tickets based on historial pricing trends.

