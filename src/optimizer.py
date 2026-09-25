import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def optimize_discount_rate(item_df, trained_pipeline, cost_price):
    """
    Uses SciPy to find the optimal discount rate (0% to 50%) that maximizes net margin
    considering inventory holding costs and spoilage penalties.
    Includes price elasticity modeling and expiry-based urgency.
    """
    orig_price = item_df['original_price'].values[0]
    inv_depth = item_df['inventory_depth'].values[0]
    days_to_expiry = item_df['days_to_expiry'].values[0]

    def objective_function(discount):
        temp_df = item_df.copy()
        temp_df['discount_pct'] = discount

        # Get base prediction from model
        base_pred = trained_pipeline.predict(temp_df)[0]

        # Cap predictions at 70% of inventory (demand uncertainty: not everything sells)
        # This creates realistic elasticity where discounts matter
        base_pred = min(base_pred, inv_depth * 0.70)

        # Price elasticity: demand increases as price decreases
        # Stronger elasticity for urgent items (low days_to_expiry)
        urgency_multiplier = min(4.0, 8.0 / (days_to_expiry + 0.3))
        base_elasticity = 0.50  # 50% volume lift at full 50% discount

        elasticity_factor = (discount / 0.5) * base_elasticity * urgency_multiplier
        pred_units = base_pred * (1 + elasticity_factor)
        pred_units = np.clip(pred_units, 0, inv_depth)

        selling_price = orig_price * (1 - discount)
        revenue = pred_units * selling_price

        # Spoilage penalty: higher when items expiring soon
        unsold_units = inv_depth - pred_units
        urgency_factor = 0.5 + (1.5 / (days_to_expiry + 0.2))  # 0.5 to 2.0 based on urgency
        spoilage_penalty = unsold_units * cost_price * min(urgency_factor, 1.0)

        profit = revenue - (pred_units * cost_price) - spoilage_penalty
        return -profit  # Minimize negative profit = Maximize profit

    result = minimize_scalar(objective_function, bounds=(0.0, 0.50), method='bounded')
    return result.x, -result.fun
