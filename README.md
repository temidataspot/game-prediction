# Horse Racing Probabilistic Modeling 

**Tools**: Python (Jupyter Notebook in VS Code) | LightGBM | XGBoost | Scikit-learn | Pandas | NumPy  

---

## Project Files

- [train.csv](https://github.com/temidataspot/game-prediction/blob/main/train.csv) – Training dataset used to build the model  
- [test.csv](https://github.com/temidataspot/game-prediction/blob/main/test.csv) – Test dataset used for final predictions  
- [horse_racing_predictions.csv](https://github.com/temidataspot/game-prediction/blob/main/Predicted_Probabilities.csv) – Final predictions with normalized win probabilities 
- [horse_racing_modeling.ipynb](https://github.com/temidataspot/game-prediction/blob/main/PredictionModel.ipynb) – Full Jupyter Notebook with code for preprocessing, modeling, calibration, and evaluation

---

## Project Goal

The goal of this project was to develop a **probabilistic model** that predicts the **likelihood of each horse winning** its race. A key requirement was to ensure that for every race, the **sum of predicted win probabilities across all horses is exactly 1**—forming a proper probability distribution.

---

## Feature Selection & Modeling

- Target: `Position == 1` (1 = win, 0 = not win)
- Dropped features: `betfairSP`, `timeSecs`, `pdsBeaten`, `NMFP` (to prevent data leakage)
- Models:
  - **XGBoost** as baseline
  - **LightGBM** for final modeling due to faster performance and native handling of categorical features
- Calibration:
  - Applied **Isotonic Regression** using `CalibratedClassifierCV` for improving the reliability of probability estimates

---

## Feature Engineering

Key focus was on **race-relative features** that describe how each horse compares to others in the same race:

- **Speed vs Field Average** – Captures if a horse is faster/slower than average
- Considered (but excluded):
  - **Speed Rank in Race**
  - **Market Odds vs Field Average**

---

## Probability Normalisation

Although `predict_proba()` from LightGBM and calibration via isotonic regression produced probabilities, they didn’t naturally sum to 1 per race. To fix this:
- Grouped predictions by `Race_ID`
- Normalized each horse’s probability within its race so the sum equals 1
- Ensured final predictions represent **valid probability distributions**

---

## Evaluation Metrics

| Metric      | Value |
|-------------|-------|
| **Log Loss**| 0.32  |
| **Brier Score** | 0.09 |

- Log Loss penalizes overconfident incorrect predictions
- Brier Score reflects the average squared difference between predicted and actual outcomes

---

## Assumptions & Limitations

- Assumed static, up-to-date features (no real-time updates like live odds)
- No external features (e.g., weather, jockey changes)
- Class imbalance handled using `class_weight='balanced'`
- LightGBM required column name cleaning (regex fix for special characters)
- Manual normalization applied post-prediction to satisfy probability constraints
- Submission races with missing IDs were recovered by merging predictions back to original `test.csv`

---

## Summary

This project demonstrates how **structured modeling and calibration techniques** can be used to derive meaningful win probabilities in a high-variance, competitive domain like horse racing. Special focus was placed on **model trustworthiness**, **fair comparisons across races**, and **interpretable outputs**.

---

*All visuals contained in the [jupyter notebook](https://github.com/temidataspot/game-prediction/blob/main/PredictionModel.ipynb)*
