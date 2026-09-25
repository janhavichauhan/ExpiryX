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
    print("TESTING MULTIPLE SCENARIOS FOR PRICING OPTIMIZATION")
    print("=" * 80)

    # Load and train model once
    print("\n[Setup] Loading data and training model...")
    df = load_and_preprocess_kaggle_data(train_path='data/train.csv', store_path='data/store.csv')
    model_pipeline, metrics = build_and_train_pipeline(df)
    print(f"Model trained with R² = {metrics['R2']:.4f}")

    # Define test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Premium Item, Long Shelf Life',
            'category': 'a',
            'store_tier': 'a',
            'original_price': 200.0,
            'days_to_expiry': 14.0,
            'inventory_depth': 300,
            'cost_price': 100.0
        },
        {
            'name': 'Scenario 2: Budget Item, Expiring Soon',
            'category': 'c',
            'store_tier': 'c',
            'original_price': 50.0,
            'days_to_expiry': 2.0,
            'inventory_depth': 800,
            'cost_price': 25.0
        },
        {
            'name': 'Scenario 3: Mid-Range Item, CRITICAL Expiry',
            'category': 'b',
            'store_tier': 'b',
            'original_price': 80.0,
            'days_to_expiry': 1.0,
            'inventory_depth': 500,
            'cost_price': 40.0
        },
        {
            'name': 'Scenario 4: High-Margin Item, Moderate Inventory',
            'category': 'a',
            'store_tier': 'a',
            'original_price': 150.0,
            'days_to_expiry': 7.0,
            'inventory_depth': 200,
            'cost_price': 60.0
        },
        {
            'name': 'Scenario 5: Low-Price Item, Deep Stock',
            'category': 'c',
            'store_tier': 'b',
            'original_price': 30.0,
            'days_to_expiry': 5.0,
            'inventory_depth': 1500,
            'cost_price': 12.0
        },
        {
            'name': 'Scenario 6: High-Margin Premium, Expiring ASAP',
            'category': 'a',
            'store_tier': 'a',
            'original_price': 250.0,
            'days_to_expiry': 0.5,
            'inventory_depth': 100,
            'cost_price': 120.0
        }
    ]

    # Run scenarios
    results = []
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 80)

        item_df = pd.DataFrame([{
            'category': scenario['category'],
            'store_tier': scenario['store_tier'],
            'original_price': scenario['original_price'],
            'days_to_expiry': scenario['days_to_expiry'],
            'discount_pct': 0.0,
            'inventory_depth': scenario['inventory_depth'],
            'hour_of_day': 5
        }])

        optimal_discount, max_profit = optimize_discount_rate(
            item_df,
            model_pipeline,
            cost_price=scenario['cost_price']
        )

        discounted_price = scenario['original_price'] * (1 - optimal_discount)
        margin_per_unit = discounted_price - scenario['cost_price']
        margin_pct = (margin_per_unit / discounted_price) * 100 if discounted_price > 0 else 0

        print(f"  Original Price      : ${scenario['original_price']:.2f}")
        print(f"  Cost Price          : ${scenario['cost_price']:.2f}")
        print(f"  Stock Level         : {scenario['inventory_depth']} units")
        print(f"  Days to Expiry      : {scenario['days_to_expiry']} days")
        print(f"  >> OPTIMAL DISCOUNT : {optimal_discount*100:.2f}%")
        print(f"  >> FINAL PRICE      : ${discounted_price:.2f}")
        print(f"  >> MARGIN PER UNIT  : ${margin_per_unit:.2f} ({margin_pct:.1f}%)")
        print(f"  >> MAX PROFIT       : ${max_profit:.2f}")

        results.append({
            'Scenario': scenario['name'].split(': ')[1],
            'Original Price': f"${scenario['original_price']:.0f}",
            'Days to Expiry': scenario['days_to_expiry'],
            'Inventory': scenario['inventory_depth'],
            'Optimal Discount': f"{optimal_discount*100:.1f}%",
            'Final Price': f"${discounted_price:.2f}",
            'Max Profit': f"${max_profit:.0f}"
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("- Premium items with long shelf life: Keep price high (lower discounts)")
    print("- Budget items expiring soon: Deeper discounts to move stock")
    print("- Critical expiry (1-2 days): Higher discounts despite margin loss")
    print("- High inventory + low expiry: More aggressive discounting to clear")
    print("- Low inventory + long shelf life: Full price, no urgency to discount")

if __name__ == '__main__':
    main()
