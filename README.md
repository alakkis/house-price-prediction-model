# House Price Prediction – Advanced Regression Project

This project uses real housing data to build a machine learning system that predicts house prices. The goal is to apply industry-standard preprocessing and modeling techniques, evaluate results, and understand the value of each method used.

---

## 🧠 Intuitive Overview

- I was given a dataset of homes with features like square footage, year built, bathrooms, etc.
- My goal: predict the house sale price as accurately as possible.
- I cleaned the data, engineered meaningful features, tested several machine learning models, and built ensemble systems to combine their strengths.

---

## 🧮 Algorithms & Underlying Math (Simple View)

Each model tries to learn a function `f(X)` that maps input features `X` to predicted prices `y` by minimizing prediction error.

- **Linear Regression**:  
  Predicts `y = Xβ + ε` by minimizing squared error:  
  `Loss = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - Xᵢβ)²`

- **Ridge Regression**:  
  Adds a penalty to prevent overfitting:  
  `Loss = Σ(yᵢ - Xᵢβ)² + λΣβⱼ²`

- **Random Forest**:  
  Builds many trees on random data slices and averages predictions:  
  `ŷ = (1/N) * Σ Treeᵢ(X)`

- **Gradient Boosting / XGBoost / LightGBM / CatBoost**:  
  Add models sequentially to reduce errors:  
  `ŷᵗ = ŷᵗ⁻¹ + η * Tree(residuals)`

- **Ensemble Models (Voting, Stacking)**:  
  Combine predictions of many models to improve stability.

All models aim to minimize the **Root Mean Squared Error (RMSE)**, which tells us how far off our predictions are from real prices, on average.

---

## 🔁 Step-by-Step Process

1. **Data Cleaning**
   - Removed outliers based on visual inspection
   - Handled missing values using domain knowledge

2. **Feature Engineering**
   - Created new features: `house age`, `total SF`, `total baths`, etc.
   - Log-transformed the target `SalePrice` to reduce skew

3. **Encoding and Scaling**
   - Applied `Ordinal Encoding`, `One-Hot Encoding`, and `Standard Scaling`
   - Built a custom `Pipeline` to apply all preprocessing steps consistently

4. **Model Training & Tuning**
   - Trained and cross-validated:
     - Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, CatBoost, LightGBM
   - Used `GridSearchCV` for hyperparameter tuning

5. **Ensembling**
   - Built a **Voting Regressor** to average predictions from top models
   - Built a **Stacking Regressor** using all models, with Voting as the meta-model

6. **Final Prediction**
   - Transformed the test set using the pipeline
   - Generated final predictions in a Kaggle-style format (`submission.csv`)

---

## 📊 Results

| Model              | RMSE       |
|--------------------|------------|
| Linear Regression  | 6.66e+18   |
| Random Forest      | 0.1340     |
| XGBoost            | 0.1193     |
| Ridge Regression   | **0.1091** |
| Gradient Boosting  | 0.1136     |
| LightGBM           | 0.1273     |
| CatBoost           | 0.1177     |
| Voting Regressor   | 0.1223     |
| Stacking Regressor | 0.1246     |

- **Best model**: Ridge Regression with RMSE = **0.1091**
- **Ensemble models** were also strong, with slightly higher RMSE but better stability

*Note: RMSE scores shown above are based on a hold-out validation set (20% split from training data) and evaluated using `cross_val_score` or similar techniques.*

---

## 🧾 Notes

- Final predictions were generated in a Kaggle-style format, but the actual competition deadline was missed, so no leaderboard score is available.
- This was my first full ML system build and primarily a learning project. While the Kaggle submission was missed, the focus was on mastering preprocessing, model tuning, and performance evaluation.

---


