#!/usr/bin/env python3
"""
Multi-Modal Learning Solution for Datathon
Combining different data representations and learning approaches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_text_features(train_df, test_df):
    """Create text-based features from categorical data"""
    print("=== CREATING TEXT FEATURES ===")
    
    # Combine all categorical data into text
    def create_session_text(group):
        events = ' '.join(group['event_type'].tolist())
        products = ' '.join(group['product_id'].tolist())
        categories = ' '.join(group['category_id'].tolist())
        return f"{events} {products} {categories}"
    
    # Create text for each session
    train_texts = train_df.groupby('user_session').apply(create_session_text).reset_index()
    train_texts.columns = ['user_session', 'session_text']
    
    test_texts = test_df.groupby('user_session').apply(create_session_text).reset_index()
    test_texts.columns = ['user_session', 'session_text']
    
    # Simple text features
    train_texts['text_length'] = train_texts['session_text'].str.len()
    train_texts['word_count'] = train_texts['session_text'].str.split().str.len()
    train_texts['unique_words'] = train_texts['session_text'].str.split().apply(lambda x: len(set(x)))
    
    test_texts['text_length'] = test_texts['session_text'].str.len()
    test_texts['word_count'] = test_texts['session_text'].str.split().str.len()
    test_texts['unique_words'] = test_texts['session_text'].str.split().apply(lambda x: len(set(x)))
    
    # Event type frequency in text
    event_types = ['VIEW', 'ADD_CART', 'REMOVE_CART', 'BUY']
    for event in event_types:
        train_texts[f'{event}_freq'] = train_texts['session_text'].str.count(event)
        test_texts[f'{event}_freq'] = test_texts['session_text'].str.count(event)
    
    # Text complexity features
    train_texts['text_diversity'] = train_texts['unique_words'] / (train_texts['word_count'] + 1)
    test_texts['text_diversity'] = test_texts['unique_words'] / (test_texts['word_count'] + 1)
    
    return train_texts, test_texts

def create_numerical_features(train_df, test_df):
    """Create numerical features"""
    print("=== CREATING NUMERICAL FEATURES ===")
    
    # Basic numerical features
    train_numerical = train_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique',
        'user_id': 'first'
    }).reset_index()
    
    test_numerical = test_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique',
        'user_id': 'first'
    }).reset_index()
    
    # Flatten columns
    train_numerical.columns = ['user_session', 'event_count', 'event_type_count', 'product_count', 'total_views', 'category_count', 'user_id']
    test_numerical.columns = ['user_session', 'event_count', 'event_type_count', 'product_count', 'total_views', 'category_count', 'user_id']
    
    # Add derived features
    train_numerical['event_product_ratio'] = train_numerical['event_count'] / (train_numerical['product_count'] + 1)
    train_numerical['product_category_ratio'] = train_numerical['product_count'] / (train_numerical['category_count'] + 1)
    train_numerical['engagement_score'] = (train_numerical['product_count'] * train_numerical['total_views']) / (train_numerical['event_count'] + 1)
    
    test_numerical['event_product_ratio'] = test_numerical['event_count'] / (test_numerical['product_count'] + 1)
    test_numerical['product_category_ratio'] = test_numerical['product_count'] / (test_numerical['category_count'] + 1)
    test_numerical['engagement_score'] = (test_numerical['product_count'] * test_numerical['total_views']) / (test_numerical['event_count'] + 1)
    
    return train_numerical, test_numerical

def create_categorical_features(train_df, test_df):
    """Create categorical features with encoding"""
    print("=== CREATING CATEGORICAL FEATURES ===")
    
    # Create categorical features
    train_categorical = train_df.groupby('user_session').agg({
        'user_id': 'first',
        'event_type': lambda x: list(x),
        'product_id': lambda x: list(x),
        'category_id': lambda x: list(x)
    }).reset_index()
    
    test_categorical = test_df.groupby('user_session').agg({
        'user_id': 'first',
        'event_type': lambda x: list(x),
        'product_id': lambda x: list(x),
        'category_id': lambda x: list(x)
    }).reset_index()
    
    # Encode categorical variables
    label_encoders = {}
    
    for col in ['user_id']:
        le = LabelEncoder()
        all_values = pd.concat([train_categorical[col], test_categorical[col]]).unique()
        le.fit(all_values)
        
        train_categorical[f'{col}_encoded'] = le.transform(train_categorical[col])
        test_categorical[f'{col}_encoded'] = le.transform(test_categorical[col])
        label_encoders[col] = le
    
    # Create sequence-based categorical features
    def create_sequence_features(events, products, categories):
        # Event sequence pattern
        event_pattern = '_'.join(events)
        
        # Product interaction pattern
        product_pattern = '_'.join(products)
        
        # Category exploration pattern
        category_pattern = '_'.join(categories)
        
        return pd.Series({
            'event_pattern': event_pattern,
            'product_pattern': product_pattern,
            'category_pattern': category_pattern,
            'pattern_length': len(events)
        })
    
    # Apply sequence feature creation
    train_seq_features = train_categorical.apply(
        lambda x: create_sequence_features(x['event_type'], x['product_id'], x['category_id']), axis=1
    )
    test_seq_features = test_categorical.apply(
        lambda x: create_sequence_features(x['event_type'], x['product_id'], x['category_id']), axis=1
    )
    
    # Merge sequence features
    train_categorical = pd.concat([train_categorical, train_seq_features], axis=1)
    test_categorical = pd.concat([test_categorical, test_seq_features], axis=1)
    
    return train_categorical, test_categorical

def create_temporal_features(train_df, test_df):
    """Create temporal features"""
    print("=== CREATING TEMPORAL FEATURES ===")
    
    # Convert to datetime
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    test_df['event_time'] = pd.to_datetime(test_df['event_time'])
    
    # Session-level temporal features
    train_temporal = train_df.groupby('user_session')['event_time'].agg(['min', 'max']).reset_index()
    test_temporal = test_df.groupby('user_session')['event_time'].agg(['min', 'max']).reset_index()
    
    # Calculate duration
    train_temporal['duration_seconds'] = (train_temporal['max'] - train_temporal['min']).dt.total_seconds()
    test_temporal['duration_seconds'] = (test_temporal['max'] - test_temporal['min']).dt.total_seconds()
    
    # Time-based features
    train_df['hour'] = train_df['event_time'].dt.hour
    test_df['hour'] = test_df['event_time'].dt.hour
    
    train_df['day_of_week'] = train_df['event_time'].dt.dayofweek
    test_df['day_of_week'] = test_df['event_time'].dt.dayofweek
    
    # Aggregate time features
    train_time_stats = train_df.groupby('user_session').agg({
        'hour': ['mean', 'std', 'min', 'max'],
        'day_of_week': ['mean', 'std']
    }).reset_index()
    
    test_time_stats = test_df.groupby('user_session').agg({
        'hour': ['mean', 'std', 'min', 'max'],
        'day_of_week': ['mean', 'std']
    }).reset_index()
    
    # Flatten columns
    train_time_stats.columns = ['user_session', 'hour_mean', 'hour_std', 'hour_min', 'hour_max', 'day_mean', 'day_std']
    test_time_stats.columns = ['user_session', 'hour_mean', 'hour_std', 'hour_min', 'hour_max', 'day_mean', 'day_std']
    
    # Merge temporal features
    train_temporal = train_temporal.merge(train_time_stats, on='user_session')
    test_temporal = test_temporal.merge(test_time_stats, on='user_session')
    
    return train_temporal, test_temporal

def create_behavioral_features(train_df, test_df):
    """Create behavioral pattern features"""
    print("=== CREATING BEHAVIORAL FEATURES ===")
    
    # Event sequence analysis
    def analyze_behavior(events, products, categories):
        # Conversion path analysis
        if 'BUY' in events:
            buy_index = events.index('BUY')
            if 'ADD_CART' in events[:buy_index]:
                conversion_path = 1
            else:
                conversion_path = 0
        else:
            conversion_path = 0
        
        # Bounce analysis
        if len(events) == 1 and events[0] == 'VIEW':
            bounce = 1
        else:
            bounce = 0
        
        # Cart abandonment
        cart_events = [i for i, e in enumerate(events) if e == 'ADD_CART']
        remove_events = [i for i, e in enumerate(events) if e == 'REMOVE_CART']
        
        if cart_events and not remove_events:
            cart_abandonment = 0  # No removal, might be good
        elif cart_events and remove_events:
            cart_abandonment = 1  # Had removal
        else:
            cart_abandonment = 0
        
        # Product exploration
        unique_products = len(set(products))
        unique_categories = len(set(categories))
        
        return pd.Series({
            'conversion_path': conversion_path,
            'bounce': bounce,
            'cart_abandonment': cart_abandonment,
            'product_exploration': unique_products,
            'category_exploration': unique_categories,
            'exploration_ratio': unique_products / (unique_categories + 1)
        })
    
    # Apply behavioral analysis
    train_behavior = train_df.groupby('user_session').agg({
        'event_type': lambda x: list(x),
        'product_id': lambda x: list(x),
        'category_id': lambda x: list(x)
    }).reset_index()
    
    test_behavior = test_df.groupby('user_session').agg({
        'event_type': lambda x: list(x),
        'product_id': lambda x: list(x),
        'category_id': lambda x: list(x)
    }).reset_index()
    
    train_behavior_features = train_behavior.apply(
        lambda x: analyze_behavior(x['event_type'], x['product_id'], x['category_id']), axis=1
    )
    test_behavior_features = test_behavior.apply(
        lambda x: analyze_behavior(x['event_type'], x['product_id'], x['category_id']), axis=1
    )
    
    # Merge behavioral features
    train_behavior = pd.concat([train_behavior, train_behavior_features], axis=1)
    test_behavior = pd.concat([test_behavior, test_behavior_features], axis=1)
    
    return train_behavior, test_behavior

def merge_all_modalities(train_texts, test_texts, train_numerical, test_numerical,
                        train_categorical, test_categorical, train_temporal, test_temporal,
                        train_behavior, test_behavior):
    """Merge all modality features"""
    print("=== MERGING ALL MODALITIES ===")
    
    # Merge train features
    train_final = train_texts.merge(train_numerical, on='user_session', how='left')
    train_final = train_final.merge(train_categorical, on='user_session', how='left')
    train_final = train_final.merge(train_temporal, on='user_session', how='left')
    train_final = train_final.merge(train_behavior, on='user_session', how='left')
    
    # Merge test features
    test_final = test_texts.merge(test_numerical, on='user_session', how='left')
    test_final = test_final.merge(test_categorical, on='user_session', how='left')
    test_final = test_final.merge(test_temporal, on='user_session', how='left')
    test_final = test_final.merge(test_behavior, on='user_session', how='left')
    
    # Fill NaN values
    train_final = train_final.fillna(0)
    test_final = test_final.fillna(0)
    
    print(f"Final train features: {train_final.shape}")
    print(f"Final test features: {test_final.shape}")
    
    return train_final, test_final

def train_multimodal_models(train_features, test_features):
    """Train models using multi-modal features"""
    print("=== TRAINING MULTI-MODAL MODELS ===")
    
    # Get training targets
    train_targets = train_features.groupby('user_session')['session_value'].first()
    
    # Select features (exclude non-numeric and target)
    exclude_cols = ['user_session', 'session_text', 'event_pattern', 'product_pattern', 'category_pattern', 'session_value']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features:")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    X = train_features[feature_cols].fillna(0)
    y = train_targets
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models for different modalities
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
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
    
    return results, feature_cols, scaler

def make_multimodal_predictions(results, test_features, feature_cols, scaler):
    """Make predictions using multi-modal models"""
    print("=== MAKING MULTI-MODAL PREDICTIONS ===")
    
    X_test = test_features[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions from all models
    predictions = {}
    for name, result in results.items():
        pred = result['model'].predict(X_test_scaled)
        predictions[name] = pred
        print(f"{name} predictions: {pred.min():.2f} to {pred.max():.2f}")
    
    # Ensemble predictions (weighted average based on CV performance)
    ensemble_pred = np.zeros(len(X_test))
    total_weight = 0
    
    for name, result in results.items():
        weight = 1 / (result['cv_rmse_mean'] + 1e-6)
        ensemble_pred += weight * predictions[name]
        total_weight += weight
    
    ensemble_pred /= total_weight
    
    print(f"\nEnsemble predictions: {ensemble_pred.min():.2f} to {ensemble_pred.max():.2f}")
    
    return ensemble_pred

def create_multimodal_submission(test_features, predictions, filename='multimodal_submission.csv'):
    """Create submission file from multi-modal predictions"""
    print(f"Creating multi-modal submission...")
    
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Multi-modal submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== MULTI-MODAL DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create features for each modality
        train_texts, test_texts = create_text_features(train_df, test_df)
        train_numerical, test_numerical = create_numerical_features(train_df, test_df)
        train_categorical, test_categorical = create_categorical_features(train_df, test_df)
        train_temporal, test_temporal = create_temporal_features(train_df, test_df)
        train_behavior, test_behavior = create_behavioral_features(train_df, test_df)
        
        # Merge all modalities
        train_final, test_final = merge_all_modalities(
            train_texts, test_texts, train_numerical, test_numerical,
            train_categorical, test_categorical, train_temporal, test_temporal,
            train_behavior, test_behavior
        )
        
        # Train models
        results, feature_cols, scaler = train_multimodal_models(train_final, test_final)
        
        # Make predictions
        predictions = make_multimodal_predictions(results, test_final, feature_cols, scaler)
        
        # Create submission
        submission = create_multimodal_submission(test_final, predictions)
        
        print("\n=== MULTI-MODAL SOLUTION COMPLETED! ===")
        print("Your multi-modal submission is ready!")
        
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
