#!/usr/bin/env python3
"""
Advanced Datathon Solution with Creative Features
Multiple models and innovative approaches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load data and perform deep analysis"""
    print("=== ADVANCED DATA ANALYSIS ===")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Convert event_time to datetime
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    test_df['event_time'] = pd.to_datetime(test_df['event_time'])
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    return train_df, test_df

def create_creative_features(train_df, test_df):
    """Create innovative features that others might not think of"""
    print("\n=== CREATIVE FEATURE ENGINEERING ===")
    
    # 1. TIME-BASED FEATURES (Çok önemli!)
    print("Creating time-based features...")
    
    # Session duration (first to last event)
    train_session_times = train_df.groupby('user_session')['event_time'].agg(['min', 'max'])
    train_session_times['session_duration_minutes'] = (train_session_times['max'] - train_session_times['min']).dt.total_seconds() / 60
    
    test_session_times = test_df.groupby('user_session')['event_time'].agg(['min', 'max'])
    test_session_times['session_duration_minutes'] = (test_session_times['max'] - test_session_times['min']).dt.total_seconds() / 60
    
    # Time of day features
    train_df['hour'] = train_df['event_time'].dt.hour
    test_df['hour'] = test_df['event_time'].dt.hour
    
    # Day of week
    train_df['day_of_week'] = train_df['event_time'].dt.dayofweek
    test_df['day_of_week'] = test_df['event_time'].dt.dayofweek
    
    # 2. BEHAVIORAL PATTERN FEATURES
    print("Creating behavioral pattern features...")
    
    # Event sequence patterns
    train_events = train_df.groupby('user_session')['event_type'].apply(list)
    test_events = test_df.groupby('user_session')['event_type'].apply(list)
    
    # Count specific event sequences
    def count_event_patterns(events):
        patterns = {
            'view_to_cart': 0,  # VIEW -> ADD_CART
            'cart_to_buy': 0,   # ADD_CART -> BUY
            'cart_remove': 0,   # ADD_CART -> REMOVE_CART
            'bounce': 0         # Only VIEW events
        }
        
        if len(events) == 1 and events[0] == 'VIEW':
            patterns['bounce'] = 1
        else:
            for i in range(len(events) - 1):
                if events[i] == 'VIEW' and events[i+1] == 'ADD_CART':
                    patterns['view_to_cart'] += 1
                elif events[i] == 'ADD_CART' and events[i+1] == 'BUY':
                    patterns['cart_to_buy'] += 1
                elif events[i] == 'ADD_CART' and events[i+1] == 'REMOVE_CART':
                    patterns['cart_remove'] += 1
        
        return patterns
    
    train_patterns = train_events.apply(count_event_patterns)
    test_patterns = test_events.apply(count_event_patterns)
    
    # 3. PRODUCT ENGAGEMENT FEATURES
    print("Creating product engagement features...")
    
    # Product interaction depth
    train_product_interactions = train_df.groupby('user_session').agg({
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique'
    }).reset_index()
    
    test_product_interactions = test_df.groupby('user_session').agg({
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique'
    }).reset_index()
    
    # Flatten columns
    train_product_interactions.columns = ['user_session', 'unique_products', 'total_product_views', 'unique_categories']
    test_product_interactions.columns = ['user_session', 'unique_products', 'total_product_views', 'unique_categories']
    
    # 4. USER BEHAVIOR FEATURES
    print("Creating user behavior features...")
    
    # User session frequency
    train_user_sessions = train_df.groupby('user_id')['user_session'].nunique().reset_index()
    train_user_sessions.columns = ['user_id', 'user_total_sessions']
    
    test_user_sessions = test_df.groupby('user_id')['user_session'].nunique().reset_index()
    test_user_sessions.columns = ['user_id', 'user_total_sessions']
    
    # 5. ADVANCED AGGREGATION FEATURES
    print("Creating advanced aggregation features...")
    
    # Event type ratios
    train_event_ratios = train_df.groupby('user_session')['event_type'].value_counts().unstack(fill_value=0)
    test_event_ratios = test_df.groupby('user_session')['event_type'].value_counts().unstack(fill_value=0)
    
    # Fill missing event types with 0
    for event_type in ['VIEW', 'ADD_CART', 'REMOVE_CART', 'BUY']:
        if event_type not in train_event_ratios.columns:
            train_event_ratios[event_type] = 0
        if event_type not in test_event_ratios.columns:
            test_event_ratios[event_type] = 0
    
    # Calculate ratios
    train_event_ratios['total_events'] = train_event_ratios.sum(axis=1)
    test_event_ratios['total_events'] = test_event_ratios.sum(axis=1)
    
    for event_type in ['VIEW', 'ADD_CART', 'REMOVE_CART', 'BUY']:
        train_event_ratios[f'{event_type}_ratio'] = train_event_ratios[event_type] / train_event_ratios['total_events']
        test_event_ratios[f'{event_type}_ratio'] = test_event_ratios[event_type] / test_event_ratios['total_events']
    
    # 6. MERGE ALL FEATURES
    print("Merging all features...")
    
    # Training features
    train_features = train_df.groupby('user_session').agg({
        'user_id': 'first',
        'session_value': 'first'
    }).reset_index()
    
    # Merge all feature sets
    train_features = train_features.merge(train_session_times, left_on='user_session', right_index=True)
    train_features = train_features.merge(train_product_interactions, on='user_session')
    train_features = train_features.merge(train_user_sessions, on='user_id')
    train_features = train_features.merge(train_event_ratios, left_on='user_session', right_index=True)
    
    # Test features
    test_features = test_df.groupby('user_session').agg({
        'user_id': 'first'
    }).reset_index()
    
    test_features = test_features.merge(test_session_times, left_on='user_session', right_index=True)
    test_features = test_features.merge(test_product_interactions, on='user_session')
    test_features = test_features.merge(test_user_sessions, on='user_id')
    test_features = test_features.merge(test_event_ratios, left_on='user_session', right_index=True)
    
    # 7. ADD PATTERN FEATURES
    train_patterns_df = pd.DataFrame(train_patterns.tolist(), index=train_patterns.index).reset_index()
    test_patterns_df = pd.DataFrame(test_patterns.tolist(), index=test_patterns.index).reset_index()
    
    train_features = train_features.merge(train_patterns_df, on='user_session')
    test_features = test_features.merge(test_patterns_df, on='user_session')
    
    # 8. CREATE INTERACTION FEATURES
    print("Creating interaction features...")
    
    # Product engagement score
    train_features['product_engagement_score'] = (train_features['unique_products'] * train_features['total_product_views']) / (train_features['total_events'] + 1)
    test_features['product_engagement_score'] = (test_features['unique_products'] * test_features['total_product_views']) / (test_features['total_events'] + 1)
    
    # Conversion probability
    train_features['conversion_probability'] = train_features['cart_to_buy'] / (train_features['view_to_cart'] + 1)
    test_features['conversion_probability'] = test_features['cart_to_buy'] / (test_features['view_to_cart'] + 1)
    
    # Bounce rate indicator
    train_features['bounce_indicator'] = train_features['bounce'].astype(int)
    test_features['bounce_indicator'] = test_features['bounce'].astype(int)
    
    print(f"Final training features: {train_features.shape}")
    print(f"Final test features: {test_features.shape}")
    
    return train_features, test_features

def train_multiple_models(train_features):
    """Train multiple different models"""
    print("\n=== TRAINING MULTIPLE MODELS ===")
    
    # Select features (exclude non-numeric and target)
    exclude_cols = ['user_session', 'user_id', 'session_value', 'min', 'max']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features:")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    X = train_features[feature_cols].fillna(0)
    y = train_features['session_value']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)
            
            # Train on full data
            model.fit(X_scaled, y)
            
            results[name] = {
                'model': model,
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std()
            }
            
            print(f"{name} - CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return results, feature_cols, scaler

def ensemble_predictions(results, test_features, feature_cols, scaler):
    """Make ensemble predictions"""
    print("\n=== ENSEMBLE PREDICTIONS ===")
    
    X_test = test_features[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions from all models
    predictions = {}
    for name, result in results.items():
        try:
            pred = result['model'].predict(X_test_scaled)
            predictions[name] = pred
            print(f"{name} predictions: {pred.min():.2f} to {pred.max():.2f}")
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Create ensemble (weighted average based on CV performance)
    ensemble_pred = np.zeros(len(X_test))
    total_weight = 0
    
    for name, result in results.items():
        if name in predictions:
            # Weight inversely proportional to RMSE
            weight = 1 / (result['cv_rmse_mean'] + 1e-6)
            ensemble_pred += weight * predictions[name]
            total_weight += weight
    
    ensemble_pred /= total_weight
    
    print(f"\nEnsemble predictions: {ensemble_pred.min():.2f} to {ensemble_pred.max():.2f}")
    
    return ensemble_pred

def create_final_submission(test_features, predictions, filename='advanced_submission.csv'):
    """Create final submission file"""
    print(f"\n=== CREATING FINAL SUBMISSION ===")
    
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Advanced submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    print(f"Mean value: {submission['session_value'].mean():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== ADVANCED DATATHON SOLUTION ===\n")
    
    try:
        # Load and analyze data
        train_df, test_df = load_and_analyze_data()
        
        # Create creative features
        train_features, test_features = create_creative_features(train_df, test_df)
        
        # Train multiple models
        results, feature_cols, scaler = train_multiple_models(train_features)
        
        # Make ensemble predictions
        predictions = ensemble_predictions(results, test_features, feature_cols, scaler)
        
        # Create final submission
        submission = create_final_submission(test_features, predictions)
        
        print("\n=== ADVANCED SOLUTION COMPLETED! ===")
        print("Your advanced submission is ready!")
        
        # Show model rankings
        print("\nModel Rankings (by CV RMSE):")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_rmse_mean'])
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"{i}. {name}: {result['cv_rmse_mean']:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
