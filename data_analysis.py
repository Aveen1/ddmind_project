import pandas as pd
import numpy as np

def calculate_growth(df):
    """Calculate year-over-year percentage growth."""
    return df.pct_change(axis=1) * 100

def calculate_concentration(df):
    """Calculate concentration (percentage of total) for each cell."""
    return df.div(df.sum(axis=0), axis=1) * 100

def create_top_n_concentration(df, n_list=[10, 20, 50, 100]):
    """Calculate concentration for top N customers."""
    results = {}
    df_last_period = df[df.columns[-1]].sort_values(ascending=False)
    total_sum = df_last_period.sum()
    
    for n in n_list:
        if len(df_last_period) >= n:
            top_n = df_last_period.head(n)
            concentration = (top_n.sum() / total_sum) * 100
            results[f'Top {n}'] = {
                'count': n,
                'sum': top_n.sum(),
                'concentration': concentration,
                'customers': top_n.index.tolist()
            }
    
    return results

def create_top_n_table(results):
    """Create a formatted table for top N concentration results."""
    data = {
        'Customer Count': [],
        'Total Value': [],
        'Concentration (%)': []
    }
    
    for key, value in results.items():
        data['Customer Count'].append(value['count'])
        data['Total Value'].append(value['sum'])
        data['Concentration (%)'].append(round(value['concentration'], 2))
    
    return pd.DataFrame(data, index=[k for k in results.keys()])