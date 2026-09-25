import pandas as pd
from src.data_loader import load_and_preprocess_kaggle_data
from src.pipeline_builder import build_and_train_pipeline
from src.optimizer import optimize_discount_rate

def main():
    print("==================================================================")
    print("           ExpiryX: Dynamic Pricing Optimizer")
    print("         Perishable Inventory Discount Optimization")
    print("==================================================================")

    # 1. Load Data
    print("\n[Step 1] Loading and cleaning Kaggle Store Sales dataset...")
    df = load_and_preprocess_kaggle_data(train_path='data/train.csv', store_path='data/store.csv')
    print(f"--> Successfully loaded {len(df)} store records.")
    print("--> Features matrix ready:")
    print(df.head(3))

    # 2. Build Pipeline & Train
    print("\n[Step 2] Building Scikit-Learn Pipeline & Training Model...")
    model_pipeline, metrics = build_and_train_pipeline(df)
    print("--> Holdout Test Evaluation Metrics:")
    print(f"    - Mean Absolute Error (MAE)  : {metrics['MAE']:.4f}")
    print(f"    - Root Mean Sq. Error (RMSE) : {metrics['RMSE']:.4f}")
    print(f"    - R-Squared Score (R2)       : {metrics['R2']:.4f}")

    # 3. Optimize Live Sample
    print("\n[Step 3] Running SciPy Profit Margin Optimizer on Sample Item...")
    sample_item = pd.DataFrame([{
        'category': 'a',
        'store_tier': 'a',
        'original_price': 120.0,
        'days_to_expiry': 2.0,     # Near expiry urgency!
        'discount_pct': 0.0,       # Initial state
        'inventory_depth': 500,
        'hour_of_day': 5
    }])

    optimal_discount, max_profit = optimize_discount_rate(sample_item, model_pipeline, cost_price=65.0)

    print("\n=================== LIVE OPTIMIZATION OUTPUT ===================")
    print(f" Product Category        : Grocery Store (Near Expiry)")
    print(f" Original Retail Price   : $120.00")
    print(f" Optimal Discount Rate   : {optimal_discount * 100:.2f}%")
    print(f" Recommended Clear Price : ${120.0 * (1 - optimal_discount):.2f}")
    print(f" Max Expected Profit     : ${max_profit:.2f}")
    print("================================================================")

if __name__ == '__main__':
    main()