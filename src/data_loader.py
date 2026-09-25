import pandas as pd
import numpy as np

def load_and_preprocess_kaggle_data(train_path='data/train.csv', store_path='data/store.csv'):
    """
    Loads real Rossmann Store Sales Kaggle dataset, merges store metadata,
    and maps columns to our dynamic pricing domain schema.
    """
    # 1. Read CSV files
    train_df = pd.read_csv(train_path, low_memory=False)
    
    # Optional merge if store.csv exists in data/
    try:
        store_df = pd.read_csv(store_path)
        df = pd.merge(train_df, store_df, on='Store', how='inner')
    except FileNotFoundError:
        df = train_df.copy()

    # 2. Filter out closed store days and zero-sales records
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

    # 3. Take a representative sample (~10,000 records) for fast execution
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)

    # 4. Map existing columns to our pricing engine schema
    column_mapping = {
        'Sales': 'units_sold',
        'Customers': 'inventory_depth',
        'StoreType': 'category',
        'Assortment': 'store_tier',
        'DayOfWeek': 'hour_of_day'  # Using DayOfWeek as temporal proxy
    }
    df = df.rename(columns=column_mapping)

    # 5. Transform binary 'Promo' indicator to continuous discount percentage
    # Promo = 1 indicates an active 20% promotional markdown
    df['discount_pct'] = df['Promo'].apply(lambda x: 0.20 if x == 1 else 0.0)

    # 6. Derive synthetic/mapped 'original_price' and 'days_to_expiry'
    # Base price independent of units_sold (avoids data leakage)
    # Derived from store tier and category with random variation
    np.random.seed(42)
    base_prices = {'a': 100, 'c': 120, 'b': 80}
    df['original_price'] = df['store_tier'].map(base_prices).fillna(100)
    df['original_price'] += np.random.normal(0, 15, size=len(df))
    df['original_price'] = df['original_price'].clip(lower=20.0, upper=300.0).round(2)

    # Days to expiry derived from store competition distance (proxy for inventory turnover)
    if 'CompetitionDistance' in df.columns:
        df['days_to_expiry'] = (df['CompetitionDistance'] % 14 + 1).astype(float)
    else:
        np.random.seed(42)
        df['days_to_expiry'] = np.random.randint(1, 15, size=len(df))

    # Fill default values for category/store_tier if store.csv was not merged
    if 'category' not in df.columns:
        df['category'] = 'Grocery'
    if 'store_tier' not in df.columns:
        df['store_tier'] = 'Standard'

    # Select core feature set
    selected_cols = [
        'category', 'store_tier', 'original_price', 
        'days_to_expiry', 'discount_pct', 'inventory_depth', 
        'hour_of_day', 'units_sold'
    ]
    df = df[selected_cols].copy()

    # Inject real-world missing values to test pipeline robustness
    np.random.seed(42)
    df.loc[df.sample(frac=0.04, random_state=42).index, 'days_to_expiry'] = np.nan
    df.loc[df.sample(frac=0.02, random_state=42).index, 'category'] = np.nan

    return df