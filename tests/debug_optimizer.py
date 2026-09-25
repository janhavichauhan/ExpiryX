import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import load_and_preprocess_kaggle_data
from src.pipeline_builder import build_and_train_pipeline
from src.optimizer import optimize_discount_rate

def main():
    print("=" * 80)
    print("DEBUG: Analyzing Profit Curve with NEW Elasticity Model")
    print("=" * 80)

    # Load and train model
    df = load_and_preprocess_kaggle_data(train_path='data/train.csv', store_path='data/store.csv')
    model_pipeline, metrics = build_and_train_pipeline(df)

    # Test scenario: Budget Item, Expiring Soon
    item_df = pd.DataFrame([{
        'category': 'c',
        'store_tier': 'c',
        'original_price': 50.0,
        'days_to_expiry': 2.0,
        'discount_pct': 0.0,
        'inventory_depth': 800,
        'hour_of_day': 5
    }])

    orig_price = 50.0
    cost_price = 25.0
    inv_depth = 800
    days_to_expiry = 2.0

    print("\nScenario: Budget Item, Expiring Soon")
    print(f"  Original Price: ${orig_price}, Cost: ${cost_price}, Inventory: {inv_depth}, Days: {days_to_expiry}")
    print("\nProfit curve across discount rates (WITH demand uncertainty & elasticity):")
    print("-" * 100)
    print(f"{'Discount':<12} {'Final Price':<15} {'Base Units':<15} {'Elasticity':<15} {'Final Units':<15} {'Profit':<15}")
    print("-" * 100)

    profits = []
    discounts_tested = np.linspace(0, 0.50, 11)

    for discount in discounts_tested:
        temp_df = item_df.copy()
        temp_df['discount_pct'] = discount

        base_pred = model_pipeline.predict(temp_df)[0]

        # NEW: Cap predictions at 70% of inventory (demand uncertainty)
        base_pred_capped = min(base_pred, inv_depth * 0.70)

        # Apply elasticity: lower prices -> higher demand (UPDATED)
        urgency_multiplier = min(4.0, 8.0 / (days_to_expiry + 0.3))
        base_elasticity = 0.50
        elasticity_factor = (discount / 0.5) * base_elasticity * urgency_multiplier

        pred_units = base_pred_capped * (1 + elasticity_factor)
        pred_units = np.clip(pred_units, 0, inv_depth)

        final_price = orig_price * (1 - discount)
        revenue = pred_units * final_price
        cogs = pred_units * cost_price
        unsold_units = inv_depth - pred_units

        urgency_factor = 0.5 + (1.5 / (days_to_expiry + 0.2))
        spoilage_penalty = unsold_units * cost_price * min(urgency_factor, 1.0)

        profit = revenue - cogs - spoilage_penalty
        profits.append(profit)

        elasticity_pct = elasticity_factor * 100
        print(f"{discount*100:>6.1f}%      ${final_price:>12.2f}   {base_pred_capped:>12.0f}    +{elasticity_pct:>10.1f}%   {pred_units:>12.0f}    ${profit:>12.2f}")

    max_profit_idx = np.argmax(profits)
    optimal_discount = discounts_tested[max_profit_idx]
    max_profit = profits[max_profit_idx]

    print("-" * 100)
    print(f"\nFromProfit Curve: {optimal_discount*100:.1f}% discount -> ${profits[max_profit_idx]:.2f} profit")

    # Now run the actual optimizer
    print("\n" + "=" * 80)
    print("Running official optimizer:")
    actual_discount, actual_profit = optimize_discount_rate(item_df, model_pipeline, cost_price)
    print(f"Result: {actual_discount*100:.1f}% discount -> ${actual_profit:.2f} profit")
    print("\nNOTE: Model NOW considers:")
    print("  - Demand Uncertainty: Not all inventory will sell (capped at 70%)")
    print("  - Price Elasticity: Lower prices increase volume")
    print("  - Urgency Multiplier: Expiring items get bigger elasticity boost")
    print("  - Spoilage Penalty: Higher for urgent items")

if __name__ == '__main__':
    main()
