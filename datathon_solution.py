#!/usr/bin/env python3
"""
Datathon Competition Solution
Predicts session_value for each user_session in test data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test data"""
    print("Loading data...")
    
    # Load training data
    train_df = pd.read_csv('train.csv')
    print(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv('test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    # Load sample submission to understand format
    sample_submission = pd.read_csv('sample_submission.csv')
    print(f"Sample submission shape: {sample_submission.shape}")
    
    return train_df, test_df, sample_submission

def explore_data(train_df, test_df):
    """Explore and understand the data structure"""
    print("\n=== Data Exploration ===")
    
    print("\nTraining data columns:")
    print(train_df.columns.tolist())
    
    print("\nTraining data info:")
    print(train_df.info())
    
    print("\nTraining data sample:")
    print(train_df.head())
    
    print("\nEvent types in training data:")
    print(train_df['event_type'].value_counts())
    
    print("\nUnique sessions in training data:")
    print(f"Total unique sessions: {train_df['user_session'].nunique()}")
    
    print("\nSession values statistics:")
    print(train_df['session_value'].describe())

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    print("\n=== Feature Engineering ===")
    
    # For training data, group by session and create features
    train_features = train_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': 'nunique',
        'category_id': 'nunique',
        'user_id': 'first',
        'session_value': 'first'  # Target variable
    }).reset_index()
    
    # Flatten column names
    train_features.columns = ['user_session', 'event_count', 'event_type_count', 
                            'product_count', 'category_count', 'user_id', 'session_value']
    
    # For test data, group by session and create features
    test_features = test_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': 'nunique',
        'category_id': 'nunique',
        'user_id': 'first'
    }).reset_index()
    
    # Flatten column names
    test_features.columns = ['user_session', 'event_count', 'event_type_count', 
                           'product_count', 'category_count', 'user_id']
    
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    return train_features, test_features

def train_model(train_features):
    """Train a machine learning model"""
    print("\n=== Training Model ===")
    
    # Prepare features and target
    feature_cols = ['event_count', 'event_type_count', 'product_count', 'category_count']
    X = train_features[feature_cols]
    y = train_features['session_value']
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {np.sqrt(mse):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance)
    
    return model, feature_cols

def make_predictions(model, test_features, feature_cols):
    """Make predictions on test data"""
    print("\n=== Making Predictions ===")
    
    # Prepare test features
    X_test = test_features[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    
    return submission

def save_submission(submission, filename='submission.csv'):
    """Save submission file"""
    print(f"\n=== Saving Submission ===")
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    
    # Show sample of submission
    print("\nSample submission:")
    print(submission.head(10))

def main():
    """Main function to run the complete pipeline"""
    print("=== DATATHON COMPETITION SOLUTION ===\n")
    
    try:
        # Load data
        train_df, test_df, sample_submission = load_data()
        
        # Explore data
        explore_data(train_df, test_df)
        
        # Prepare features
        train_features, test_features = prepare_features(train_df, test_df)
        
        # Train model
        model, feature_cols = train_model(train_features)
        
        # Make predictions
        submission = make_predictions(model, test_features, feature_cols)
        
        # Save submission
        save_submission(submission)
        
        print("\n=== COMPLETED SUCCESSFULLY! ===")
        print("Your submission file 'submission.csv' is ready for upload!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
