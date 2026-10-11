# Week 5: Machine Learning Architecture Design

## 1. Project Objective

The objective of Week 5 is to design and implement initial machine learning frameworks for the Advanced Chooser Option Pricing Model project.

Two approaches are investigated:

- **Approach A:** Predict short-term volatility using machine learning and use the prediction as an input to the Black–Scholes–Merton (BSM) Chooser Option Pricing Model.
- **Approach B:** Develop an end-to-end supervised learning framework for financial price prediction.

The current experiments use JPMorgan Chase & Co. (JPM) daily financial data. JPM stock price prediction is treated as a separate experiment and is not considered a direct prediction of Chooser option market prices.

## 2. Data and Feature Engineering

The input dataset is `data/processed/jpm_features.csv`.

It contains 1,701 observations and 23 columns, covering daily observations from March 2018 through December 2024 after the existing feature engineering process.

The engineered variables include:

- Stock price and trading volume
- Treasury yield and VIX
- Daily returns
- Rolling volatility over 5, 20, and 60 trading days
- Price momentum and volume changes
- VIX and interest-rate momentum
- A VIX-derived sentiment proxy

The sentiment proxy is not based on actual news or textual sentiment data. It is a derived indicator and should be interpreted accordingly.

The supervised learning target for Approach A is the absolute return of the next trading observation:

`Target_Next_Day_Abs_Return`

This target is a proxy for short-term price movement magnitude. It is not equivalent to realized volatility calculated over a multi-day horizon.

For Approach B, the target is the next trading day's JPM stock return. The predicted return is converted into an implied next-day stock price for evaluation.

## 3. Data Splitting Strategy

The data is sorted chronologically and divided into three subsets:

| Dataset | Observations | Date Range | Approximate Share |
|---|---:|---|---:|
| Training | 1,190 | 2018-03-29 to 2022-12-16 | 70% |
| Validation | 255 | 2022-12-19 to 2023-12-22 | 15% |
| Testing | 255 | 2023-12-26 to 2024-12-30 | 15% |

The partitions are created in chronological order rather than randomly. This helps reduce the risk of using future observations to predict the past.

The current implementation constructs next-observation targets within each partition and removes the final observation in each partition when its target is unavailable. Therefore, the number of usable labeled observations may be slightly smaller than the original partition sizes.

## 4. Approach A: Machine Learning Volatility Proxy Prediction

### 4.1 Architecture

The workflow is:

1. Load the engineered JPM dataset.
2. Construct the next-day absolute-return target.
3. Split observations chronologically into training, validation, and testing subsets.
4. Train Random Forest regression models using the training subset.
5. Compare candidate hyperparameter configurations using validation MAE.
6. Evaluate the selected configuration on the held-out test subset.
7. Save predictions, evaluation metrics, feature importance, and the trained model.

### 4.2 Model

The initial model is Random Forest Regression.

The selected configuration is:

- Number of trees: 100
- Maximum tree depth: 5
- Minimum samples per leaf: 5
- Random state: 42

### 4.3 Evaluation

The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

The initial test results are:

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Training-mean baseline | 0.008481 | 0.011892 |
| Random Forest | 0.006623 | 0.011283 |

The Random Forest model performs better than the training-mean baseline on both metrics in this experiment.

The target remains a next-day absolute-return proxy rather than a direct observation of future market volatility. Further validation is needed before interpreting its predictions as annualized volatility inputs to the BSM model.

### 4.4 Outputs

The generated outputs are stored under `data/processed/week5/`:

- `random_forest_results.csv`
- `random_forest_test_predictions.csv`
- `random_forest_volatility_model.joblib`
- `random_forest_feature_importance.csv`

## 5. Approach B: End-to-End Supervised Learning Framework

### 5.1 Architecture

The workflow is:

1. Construct the next-day JPM stock return target.
2. Use engineered financial and macroeconomic variables as predictors.
3. Train a zero-return baseline, Linear Regression, and Random Forest Regression.
4. Select the best model based on validation MAE.
5. Evaluate all three models on the held-out test subset.
6. Convert predicted returns into implied next-day stock prices.
7. Save evaluation metrics and test predictions.

The zero-return baseline predicts a return of zero, equivalent to predicting that the next closing price equals the current closing price.

### 5.2 Initial Test Results

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Zero-return baseline | 1.992481 | 3.132873 |
| Linear Regression | 2.276945 | 3.334940 |
| Random Forest | 2.131365 | 3.271747 |

The zero-return baseline outperforms both machine learning models on the current test subset.

These results indicate that the initial machine learning models have not demonstrated an improvement over the simple baseline for next-day JPM stock price prediction.

### 5.3 Outputs

The generated outputs are stored under `data/processed/week5/`:

- `end_to_end_results.csv`
- `end_to_end_test_predictions.csv`

## 6. Limitations

1. The current dataset does not contain verified transaction prices for actual Chooser options traded on CME.
2. JPM stock prices are not valid substitutes for Chooser option prices.
3. BSM-generated Chooser prices are model outputs, not observed market labels.
4. The sentiment proxy is derived from VIX and does not represent actual news sentiment.
5. Next-day absolute returns are only a proxy for short-term volatility.
6. The initial end-to-end stock prediction models do not outperform the zero-return baseline on the current test subset.
7. The reported results are preliminary and should not be interpreted as evidence of profitability or real-market Chooser pricing accuracy.

## 7. Conclusion and Next Steps

Week 5 establishes initial machine learning frameworks for both volatility proxy prediction and end-to-end stock return prediction.

Approach A outperforms its training-mean baseline in the current experiment. Approach B does not outperform its zero-return baseline.

The next steps are to document the experimental results, verify the reproducibility of all generated outputs, and investigate whether improved volatility targets or additional predictive features can improve generalization.

Evaluation of actual Chooser option pricing accuracy remains pending until reliable market price labels become available.