# Stock Market Prediction Project

## Overview
This project focuses on predicting stock market performance by forecasting daily closing prices and recommending an optimal strategy—either Buy, Hold, or Sell—for a specific stock. This is a valuable service for high-net-worth individuals (HNIs) at Dane Street, looking to make significant, long-term investments. The stock market's volatility and sensitivity to global events make these predictions challenging, but accurate predictions can yield substantial profits.

Our approach involves two main tasks:
1. **Predicting the daily close price of the stock** using time-series models.
2. **Recommending a strategy (Buy, Hold, or Sell)** based on the predicted price, leveraging classification techniques.

## Data and Evaluation

The project requires submitting:
- **Close Price Predictions**: Evaluated using Symmetric Mean Absolute Percentage Error (SMAPE).
- **Strategy Recommendation**: Evaluated based on classification accuracy.

### SMAPE Calculation

The Symmetric Mean Absolute Percentage Error (SMAPE) is calculated using:

```python
smape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
```

### Strategy Accuracy Calculation

The strategy accuracy is calculated using:

```python
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred) * 100
```

The evaluation metric uses (1 - Accuracy) as the loss metric. The final score is a weighted sum of the two metrics; thus, a lower score is better.

## Objectives

1. Analyze the provided data, visualize trends, and draw insights.
2. Model future stock price trends using statistical time-series models.
3. Model the best strategy for stock prices using classification techniques.
4. Provide a comparative analysis of all models based on performance metrics.

> **Note**: Neural Networks were avoided to maintain model interpretability and comply with course guidelines. Statistical time-series models were preferred for time-series predictions.

## Methods

### 1. Close Price Prediction

We applied the following time-series forecasting models:
- **Holt-Winters Method**
- **ARIMA**
- **ARIMAX**
- **SARIMA**
- **SARIMAX**

These models were selected for their interpretability and effectiveness in time-series forecasting.

### 2. Strategy Recommendation

After engineering features like `Open`, `Close`, `Volume`, `MA_Open_10`, `MA_Volume_10`, `Open_Pct_Change`, `Volume_Pct_Change`, `OBV`, and `Strategy`, we tested multiple classification algorithms. Here is the list of models we used:

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
```

#### Model Selection and Cross-Validation

Using cross-validation with `StratifiedKFold`, we optimized model parameters through `GridSearchCV`. Each model’s performance was assessed based on accuracy.

#### Best Model
After testing several models, **XGBoost** yielded the highest accuracy of **0.6934**.

## Conclusion

This project successfully demonstrates a methodology for predicting stock prices and suggesting optimal trading strategies. While SMAPE evaluated close price predictions, classification accuracy assessed strategy recommendations. The final score—a weighted combination of SMAPE and strategy loss—provides an overall effectiveness measure.

## Future Work
Further improvements could include:
- Experimenting with hybrid models to combine time-series and classification techniques.
- Implementing advanced ensemble techniques to improve strategy classification accuracy.
- Exploring additional market factors and global events to improve predictive power.

