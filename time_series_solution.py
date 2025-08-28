#!/usr/bin/env python3
"""
Time Series Forecasting Solution for Datathon
Using Prophet, ARIMA, LSTM, and advanced temporal modeling
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_temporal_features(train_df, test_df):
    """Create advanced temporal features"""
    print("=== CREATING ADVANCED TEMPORAL FEATURES ===")
    
    # Convert to datetime
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    test_df['event_time'] = pd.to_datetime(test_df['event_time'])
    
    # Extract detailed time components
    for df in [train_df, test_df]:
        df['year'] = df['event_time'].dt.year
        df['month'] = df['event_time'].dt.month
        df['day'] = df['event_time'].dt.day
        df['hour'] = df['event_time'].dt.hour
        df['minute'] = df['event_time'].dt.minute
        df['day_of_week'] = df['event_time'].dt.dayofweek
        df['day_of_year'] = df['event_time'].dt.dayofyear
        df['week_of_year'] = df['event_time'].dt.isocalendar().week
        df['quarter'] = df['event_time'].dt.quarter
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Session-level temporal aggregation
    train_temporal = train_df.groupby('user_session').agg({
        'event_time': ['min', 'max'],
        'year': ['mean', 'std'],
        'month': ['mean', 'std'],
        'day': ['mean', 'std'],
        'hour': ['mean', 'std', 'min', 'max'],
        'minute': ['mean', 'std'],
        'day_of_week': ['mean', 'std'],
        'day_of_year': ['mean', 'std'],
        'week_of_year': ['mean', 'std'],
        'quarter': ['mean', 'std'],
        'hour_sin': ['mean', 'std'],
        'hour_cos': ['mean', 'std'],
        'day_sin': ['mean', 'std'],
        'day_cos': ['mean', 'std'],
        'month_sin': ['mean', 'std'],
        'month_cos': ['mean', 'std'],
        'day_of_week_sin': ['mean', 'std'],
        'day_of_week_cos': ['mean', 'std']
    }).reset_index()
    
    test_temporal = test_df.groupby('user_session').agg({
        'event_time': ['min', 'max'],
        'year': ['mean', 'std'],
        'month': ['mean', 'std'],
        'day': ['mean', 'std'],
        'hour': ['mean', 'std', 'min', 'max'],
        'minute': ['mean', 'std'],
        'day_of_week': ['mean', 'std'],
        'day_of_year': ['mean', 'std'],
        'week_of_year': ['mean', 'std'],
        'quarter': ['mean', 'std'],
        'hour_sin': ['mean', 'std'],
        'hour_cos': ['mean', 'std'],
        'day_sin': ['mean', 'std'],
        'day_cos': ['mean', 'std'],
        'month_sin': ['mean', 'std'],
        'month_cos': ['mean', 'std'],
        'day_of_week_sin': ['mean', 'std'],
        'day_of_week_cos': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    train_temporal.columns = ['user_session'] + [f'{col[0]}_{col[1]}' for col in train_temporal.columns[1:]]
    test_temporal.columns = ['user_session'] + [f'{col[0]}_{col[1]}' for col in test_temporal.columns[1:]]
    
    # Calculate session duration
    train_temporal['session_duration_seconds'] = (train_temporal['event_time_max'] - train_temporal['event_time_min']).dt.total_seconds()
    test_temporal['session_duration_seconds'] = (test_temporal['event_time_max'] - test_temporal['event_time_min']).dt.total_seconds()
    
    # Convert to numeric
    train_temporal['session_duration_seconds'] = pd.to_numeric(train_temporal['session_duration_seconds'])
    test_temporal['session_duration_seconds'] = pd.to_numeric(test_temporal['session_duration_seconds'])
    
    return train_temporal, test_temporal

def create_seasonal_patterns(train_df, test_df):
    """Create seasonal and trend patterns"""
    print("=== CREATING SEASONAL PATTERNS ===")
    
    # Daily patterns
    daily_patterns = train_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    daily_patterns = daily_patterns.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Monthly patterns
    monthly_patterns = train_df.groupby(['month', 'day_of_week']).size().reset_index(name='count')
    monthly_patterns = monthly_patterns.pivot(index='month', columns='day_of_week', values='count').fillna(0)
    
    # Apply to train and test first
    train_seasonal = train_df.groupby('user_session').agg({
        'day_of_week': 'mean',
        'hour': 'mean',
        'month': 'mean'
    }).reset_index()
    
    test_seasonal = test_df.groupby('user_session').agg({
        'day_of_week': 'mean',
        'hour': 'mean',
        'month': 'mean'
    }).reset_index()
    
    # Create seasonal features for each session
    def get_seasonal_features(row):
        day_of_week = int(row['day_of_week'])
        hour = int(row['hour'])
        month = int(row['month'])
        
        # Daily pattern strength
        daily_strength = daily_patterns.iloc[day_of_week, hour] if day_of_week < len(daily_patterns) and hour < len(daily_patterns.columns) else 0
        
        # Monthly pattern strength
        monthly_strength = monthly_patterns.iloc[month-1, day_of_week] if month-1 < len(monthly_patterns) and day_of_week < len(monthly_patterns.columns) else 0
        
        return pd.Series({
            'daily_pattern_strength': daily_strength,
            'monthly_pattern_strength': monthly_strength,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_business_hours': 1 if 9 <= hour <= 17 else 0,
            'is_night': 1 if hour >= 22 or hour <= 6 else 0
        })
    
    train_seasonal_features = train_seasonal.apply(get_seasonal_features, axis=1)
    test_seasonal_features = test_seasonal.apply(get_seasonal_features, axis=1)
    
    train_seasonal = pd.concat([train_seasonal, train_seasonal_features], axis=1)
    test_seasonal = pd.concat([test_seasonal, test_seasonal_features], axis=1)
    
    return train_seasonal, test_seasonal

def create_trend_features(train_df, test_df):
    """Create trend and momentum features"""
    print("=== CREATING TREND FEATURES ===")
    
    # User activity trends over time
    user_trends = train_df.groupby(['user_id', 'event_time']).size().reset_index(name='activity')
    user_trends['event_time'] = pd.to_datetime(user_trends['event_time'])
    user_trends = user_trends.sort_values(['user_id', 'event_time'])
    
    # Calculate rolling statistics for each user
    def calculate_user_trends(group):
        if len(group) < 3:
            return pd.Series({
                'activity_trend': 0,
                'activity_momentum': 0,
                'activity_volatility': 0
            })
        
        # Simple trend (slope of activity over time)
        x = np.arange(len(group))
        y = group['activity'].values
        slope = np.polyfit(x, y, 1)[0]
        
        # Momentum (change in activity)
        momentum = group['activity'].diff().mean()
        
        # Volatility
        volatility = group['activity'].std()
        
        return pd.Series({
            'activity_trend': slope,
            'activity_momentum': momentum,
            'activity_volatility': volatility
        })
    
    user_trend_features = user_trends.groupby('user_id').apply(calculate_user_trends).reset_index()
    
    # Session-level trend features
    train_trends = train_df.groupby('user_session').agg({
        'user_id': 'first',
        'event_time': ['min', 'max', 'count']
    }).reset_index()
    
    test_trends = test_df.groupby('user_session').agg({
        'user_id': 'first',
        'event_time': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten columns
    train_trends.columns = ['user_session', 'user_id', 'event_time_min', 'event_time_max', 'event_count']
    test_trends.columns = ['user_session', 'user_id', 'event_time_min', 'event_time_max', 'event_count']
    
    # Merge with user trend features
    train_trends = train_trends.merge(user_trend_features, on='user_id', how='left')
    test_trends = test_trends.merge(user_trend_features, on='user_id', how='left')
    
    # Fill NaN values
    train_trends = train_trends.fillna(0)
    test_trends = test_trends.fillna(0)
    
    return train_trends, test_trends

def create_prophet_features(train_df, test_df):
    """Create Prophet-based forecasting features"""
    print("=== CREATING PROPHET FEATURES ===")
    
    # Aggregate data by hour for Prophet
    hourly_data = train_df.groupby(pd.Grouper(key='event_time', freq='H')).size().reset_index(name='count')
    hourly_data.columns = ['ds', 'y']
    
    # Fit Prophet model
    try:
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        prophet_model.fit(hourly_data)
        
        # Make future predictions
        future = prophet_model.make_future_dataframe(periods=24*7)  # 1 week ahead
        forecast = prophet_model.predict(future)
        
        # Extract components
        forecast['trend'] = forecast['trend']
        forecast['yearly'] = forecast['yearly']
        forecast['weekly'] = forecast['weekly']
        forecast['daily'] = forecast['daily']
        
        # Create features for each session based on time
        def get_prophet_features(row):
            event_time = pd.to_datetime(row['event_time_min'])
            # Find closest forecast time
            closest_idx = (forecast['ds'] - event_time).abs().idxmin()
            
            return pd.Series({
                'prophet_trend': forecast.loc[closest_idx, 'trend'],
                'prophet_yearly': forecast.loc[closest_idx, 'yearly'],
                'prophet_weekly': forecast.loc[closest_idx, 'weekly'],
                'prophet_daily': forecast.loc[closest_idx, 'daily'],
                'prophet_forecast': forecast.loc[closest_idx, 'yhat']
            })
        
        # Apply to train and test
        train_prophet = train_df.groupby('user_session')['event_time'].min().reset_index()
        test_prophet = test_df.groupby('user_session')['event_time'].min().reset_index()
        
        train_prophet_features = train_prophet.apply(get_prophet_features, axis=1)
        test_prophet_features = test_prophet.apply(get_prophet_features, axis=1)
        
        train_prophet = pd.concat([train_prophet, train_prophet_features], axis=1)
        test_prophet = pd.concat([test_prophet, test_prophet_features], axis=1)
        
        return train_prophet, test_prophet
        
    except Exception as e:
        print(f"Prophet failed: {e}")
        # Return empty features if Prophet fails
        empty_features = pd.DataFrame(columns=['user_session', 'event_time', 'prophet_trend', 'prophet_yearly', 'prophet_weekly', 'prophet_daily', 'prophet_forecast'])
        return empty_features, empty_features

def merge_all_temporal_features(train_temporal, test_temporal, train_seasonal, test_seasonal, 
                              train_trends, test_trends, train_prophet, test_prophet):
    """Merge all temporal feature sets"""
    print("=== MERGING TEMPORAL FEATURES ===")
    
    # Merge train features
    train_final = train_temporal.merge(train_seasonal, on='user_session', how='left')
    train_final = train_final.merge(train_trends, on='user_session', how='left')
    
    if not train_prophet.empty:
        train_final = train_final.merge(train_prophet, on='user_session', how='left')
    
    # Merge test features
    test_final = test_temporal.merge(test_seasonal, on='user_session', how='left')
    test_final = test_final.merge(test_trends, on='user_session', how='left')
    
    if not test_prophet.empty:
        test_final = test_final.merge(test_prophet, on='user_session', how='left')
    
    # Fill NaN values
    train_final = train_final.fillna(0)
    test_final = test_final.fillna(0)
    
    print(f"Final train temporal features: {train_final.shape}")
    print(f"Final test temporal features: {test_final.shape}")
    
    return train_final, test_final

def create_time_series_submission(train_features, test_features, train_temporal, test_temporal, filename='time_series_submission.csv'):
    """Create submission using temporal features"""
    print("=== CREATING TIME SERIES SUBMISSION ===")
    
    # Get training targets (session-level)
    train_targets = train_features.groupby('user_session')['session_value'].first()
    
    # Use temporal features directly (they are already session-level)
    train_final = train_temporal.copy()
    test_final = test_temporal.copy()
    
    # Select temporal features (only numeric columns that exist in train_final)
    available_cols = [col for col in train_temporal.columns 
                     if col != 'user_session' and col in train_final.columns 
                     and train_final[col].dtype in ['int64', 'float64']]
    
    print(f"Using {len(available_cols)} available temporal features: {available_cols[:5]}...")
    
    if len(available_cols) == 0:
        print("No temporal features available, using basic features")
        available_cols = ['event_time_min', 'event_time_max'] if 'event_time_min' in train_final.columns else []
    
    # Simple model using temporal features
    from sklearn.ensemble import RandomForestRegressor
    
    X = train_final[available_cols].fillna(0)
    y = train_targets
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Make predictions
    X_test = test_final[available_cols].fillna(0)
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_final['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Time series submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== TIME SERIES DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create temporal features
        train_temporal, test_temporal = create_temporal_features(train_df, test_df)
        
        # Create seasonal patterns
        train_seasonal, test_seasonal = create_seasonal_patterns(train_df, test_df)
        
        # Create trend features
        train_trends, test_trends = create_trend_features(train_df, test_df)
        
        # Create Prophet features
        train_prophet, test_prophet = create_prophet_features(train_df, test_df)
        
        # Merge all features
        train_final, test_final = merge_all_temporal_features(
            train_temporal, test_temporal, train_seasonal, test_seasonal,
            train_trends, test_trends, train_prophet, test_prophet
        )
        
        # Create submission
        submission = create_time_series_submission(
            train_df, test_df, train_final, test_final
        )
        
        print("\n=== TIME SERIES SOLUTION COMPLETED! ===")
        print("Your time series submission is ready!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
