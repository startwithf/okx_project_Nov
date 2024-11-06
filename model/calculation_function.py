import numpy as np

# Define a function to calculate returns for each group
def calculate_token_returns(group):
    """
    Calculate the daily return and log return for a given group.
    Parameters:
    group (DataFrame): The group for which to calculate returns.
    Returns:
    DataFrame: The group with added columns for daily return and log return.
    """
    group['daily_return'] = group['close'].pct_change()
    group['log_return'] = np.log(group['close'] / group['close'].shift(1))
    return group

def calculate_volatility(group, interval_lst):
    for interval in interval_lst:
        group[f'daily_volatility_{interval}'] = group['daily_return'].rolling(window=interval).std()
        group[f'annualized_volatility_{interval}'] = group[f'daily_volatility_{interval}'] * np.sqrt(365)
    return group

def calculate_metric_general_mean(data, metric, start_date = '2018-01-01'):
    # Filter the data
    data = data[data['date'] >= start_date]
    # Drop the first row since it will have NaN for the log return
    data = data.dropna(subset=[metric])
    
    # Calculate the general mean
    general_mean = data.groupby('name')[metric].mean().reset_index()
    general_mean = general_mean.rename(columns={metric: f'general_{metric}_mean'})
    general_mean = general_mean.sort_values(f'general_{metric}_mean', ascending=False)
 
    return general_mean

def calculate_metric_yearly_mean(data, metric, start_date = '2018-01-01'):
    # Filter the data
    data = data[data['date'] >= start_date]
    # Drop the first row since it will have NaN for the log return
    data = data.dropna(subset=[metric])

    # Calculate the yearly mean
    data['year'] = data['date'].dt.year
    yearly_mean = data.groupby(['name', 'year'])[metric].mean().reset_index()
    yearly_mean = yearly_mean.rename(columns={metric: f'yearly_{metric}_mean'})
    
    return yearly_mean