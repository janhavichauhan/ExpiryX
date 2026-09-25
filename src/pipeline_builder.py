import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def build_and_train_pipeline(df, random_state=42):
    """
    Constructs a leak-free Scikit-Learn pipeline and trains a HistGradientBoostingRegressor
    on the preprocessed Kaggle dataset.
    """
    X = df.drop(columns=['units_sold'])
    y = df['units_sold']

    num_cols = ['original_price', 'days_to_expiry', 'discount_pct', 'inventory_depth', 'hour_of_day']
    cat_cols = ['category', 'store_tier']

    # Holdout Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    # 1. Numerical Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Master Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 4. End-to-End Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=random_state))
    ])

    # Fit pipeline on training split
    model_pipeline.fit(X_train, y_train)

    # Evaluate on holdout test set
    y_pred = model_pipeline.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    return model_pipeline, metrics