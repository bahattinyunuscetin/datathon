#!/usr/bin/env python3
"""
Graph-Based Solution for Datathon
Using network analysis, graph neural networks, and community detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_user_product_graph(train_df, test_df):
    """Create a graph connecting users, products, and categories"""
    print("=== CREATING USER-PRODUCT GRAPH ===")
    
    # Combine train and test data
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create graph
    G = nx.Graph()
    
    print("Adding nodes and edges...")
    
    # Add nodes
    for user_id in all_data['user_id'].unique():
        G.add_node(user_id, type='user')
    
    for product_id in all_data['product_id'].unique():
        G.add_node(product_id, type='product')
    
    for category_id in all_data['category_id'].unique():
        G.add_node(category_id, type='category')
    
    # Add edges with weights based on interaction frequency
    user_product_edges = all_data.groupby(['user_id', 'product_id']).size().reset_index(name='weight')
    user_category_edges = all_data.groupby(['user_id', 'category_id']).size().reset_index(name='weight')
    product_category_edges = all_data.groupby(['product_id', 'category_id']).size().reset_index(name='weight')
    
    # Add edges to graph
    for _, row in user_product_edges.iterrows():
        G.add_edge(row['user_id'], row['product_id'], weight=row['weight'], type='user_product')
    
    for _, row in user_category_edges.iterrows():
        G.add_edge(row['user_id'], row['category_id'], weight=row['weight'], type='user_category')
    
    for _, row in product_category_edges.iterrows():
        G.add_edge(row['product_id'], row['category_id'], weight=row['weight'], type='product_category')
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Users: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'user'])}")
    print(f"Products: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'product'])}")
    print(f"Categories: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'category'])}")
    
    return G, all_data

def extract_graph_features(G, train_df, test_df):
    """Extract graph-based features for each session"""
    print("=== EXTRACTING GRAPH FEATURES ===")
    
    # Get training targets
    train_targets = train_df.groupby('user_session')['session_value'].first()
    
    # Create session features
    train_features = train_df.groupby('user_session').agg({
        'user_id': 'first',
        'product_id': 'nunique',
        'category_id': 'nunique',
        'event_type': 'count'
    }).reset_index()
    
    test_features = test_df.groupby('user_session').agg({
        'user_id': 'first',
        'product_id': 'nunique',
        'category_id': 'nunique',
        'event_type': 'count'
    }).reset_index()
    
    # Add graph-based features
    print("Calculating centrality measures...")
    
    # Calculate centrality measures for users
    user_centrality = nx.degree_centrality(G)
    user_betweenness = nx.betweenness_centrality(G)
    user_closeness = nx.closeness_centrality(G)
    user_pagerank = nx.pagerank(G)
    
    # Calculate centrality measures for products
    product_centrality = nx.degree_centrality(G)
    product_betweenness = nx.betweenness_centrality(G)
    product_closeness = nx.closeness_centrality(G)
    product_pagerank = nx.pagerank(G)
    
    # Calculate centrality measures for categories
    category_centrality = nx.degree_centrality(G)
    category_betweenness = nx.betweenness_centrality(G)
    category_closeness = nx.closeness_centrality(G)
    category_pagerank = nx.pagerank(G)
    
    # Add centrality features to train features
    train_features['user_degree_centrality'] = train_features['user_id'].map(user_centrality)
    train_features['user_betweenness_centrality'] = train_features['user_id'].map(user_betweenness)
    train_features['user_closeness_centrality'] = train_features['user_id'].map(user_closeness)
    train_features['user_pagerank'] = train_features['user_id'].map(user_pagerank)
    
    # Add centrality features to test features
    test_features['user_degree_centrality'] = test_features['user_id'].map(user_centrality)
    test_features['user_betweenness_centrality'] = test_features['user_id'].map(user_betweenness)
    test_features['user_closeness_centrality'] = test_features['user_id'].map(user_closeness)
    test_features['user_pagerank'] = test_features['user_id'].map(user_pagerank)
    
    # Calculate community features
    print("Detecting communities...")
    communities = list(nx.community.greedy_modularity_communities(G))
    
    # Assign community labels
    community_labels = {}
    for i, community in enumerate(communities):
        for node in community:
            community_labels[node] = i
    
    train_features['user_community'] = train_features['user_id'].map(community_labels)
    test_features['user_community'] = test_features['user_id'].map(community_labels)
    
    # Calculate clustering coefficient
    print("Calculating clustering coefficients...")
    clustering_coeff = nx.clustering(G)
    
    train_features['user_clustering'] = train_features['user_id'].map(clustering_coeff)
    test_features['user_clustering'] = test_features['user_id'].map(clustering_coeff)
    
    # Calculate ego network features
    print("Calculating ego network features...")
    
    def get_ego_features(user_id):
        try:
            ego = nx.ego_graph(G, user_id, radius=1)
            return {
                'ego_nodes': ego.number_of_nodes(),
                'ego_edges': ego.number_of_edges(),
                'ego_density': nx.density(ego)
            }
        except:
            return {'ego_nodes': 0, 'ego_edges': 0, 'ego_density': 0}
    
    # Apply to train features
    ego_features_train = train_features['user_id'].apply(get_ego_features)
    ego_features_test = test_features['user_id'].apply(get_ego_features)
    
    train_features['ego_nodes'] = ego_features_train.apply(lambda x: x['ego_nodes'])
    train_features['ego_edges'] = ego_features_train.apply(lambda x: x['ego_edges'])
    train_features['ego_density'] = ego_features_train.apply(lambda x: x['ego_density'])
    
    test_features['ego_nodes'] = ego_features_test.apply(lambda x: x['ego_nodes'])
    test_features['ego_edges'] = ego_features_test.apply(lambda x: x['ego_edges'])
    test_features['ego_density'] = ego_features_test.apply(lambda x: x['ego_density'])
    
    # Add target variable to train features
    train_features = train_features.merge(train_targets, left_on='user_session', right_index=True)
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    return train_features, test_features

def create_interaction_patterns(train_df, test_df):
    """Create interaction pattern features"""
    print("=== CREATING INTERACTION PATTERNS ===")
    
    # Event type mapping
    event_mapping = {'VIEW': 1, 'ADD_CART': 2, 'REMOVE_CART': 3, 'BUY': 4}
    
    # Create session sequences
    train_sessions = train_df.groupby('user_session')['event_type'].apply(
        lambda x: [event_mapping[event] for event in x]
    ).reset_index()
    
    test_sessions = test_df.groupby('user_session')['event_type'].apply(
        lambda x: [event_mapping[event] for event in x]
    ).reset_index()
    
    # Calculate pattern features
    def extract_pattern_features(sequence):
        if len(sequence) == 0:
            return {
                'sequence_length': 0,
                'unique_events': 0,
                'event_variance': 0,
                'conversion_path': 0,
                'bounce_rate': 1
            }
        
        # Basic sequence features
        features = {
            'sequence_length': len(sequence),
            'unique_events': len(set(sequence)),
            'event_variance': np.var(sequence) if len(sequence) > 1 else 0
        }
        
        # Conversion path analysis
        if 4 in sequence:  # BUY event
            buy_index = sequence.index(4)
            if 2 in sequence[:buy_index]:  # ADD_CART before BUY
                features['conversion_path'] = 1
            else:
                features['conversion_path'] = 0
        else:
            features['conversion_path'] = 0
        
        # Bounce rate (only VIEW events)
        if len(sequence) == 1 and sequence[0] == 1:
            features['bounce_rate'] = 1
        else:
            features['bounce_rate'] = 0
        
        return features
    
    # Apply pattern extraction
    train_patterns = train_sessions['event_type'].apply(extract_pattern_features)
    test_patterns = test_sessions['event_type'].apply(extract_pattern_features)
    
    # Convert to DataFrame
    train_patterns_df = pd.DataFrame(train_patterns.tolist())
    test_patterns_df = pd.DataFrame(test_patterns.tolist())
    
    return train_patterns_df, test_patterns_df

def create_temporal_features(train_df, test_df):
    """Create temporal and behavioral features"""
    print("=== CREATING TEMPORAL FEATURES ===")
    
    # Convert to datetime
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    test_df['event_time'] = pd.to_datetime(test_df['event_time'])
    
    # Session-level temporal features
    train_temporal = train_df.groupby('user_session')['event_time'].agg(['min', 'max']).reset_index()
    test_temporal = test_df.groupby('user_session')['event_time'].agg(['min', 'max']).reset_index()
    
    # Calculate duration
    train_temporal['duration_minutes'] = (train_temporal['max'] - train_temporal['min']).dt.total_seconds() / 60
    test_temporal['duration_minutes'] = (test_temporal['max'] - test_temporal['min']).dt.total_seconds() / 60
    
    # Time of day features
    train_df['hour'] = train_df['event_time'].dt.hour
    test_df['hour'] = test_df['event_time'].dt.hour
    
    train_hour_stats = train_df.groupby('user_session')['hour'].agg(['mean', 'std', 'min', 'max']).reset_index()
    test_hour_stats = test_df.groupby('user_session')['hour'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Day of week features
    train_df['day_of_week'] = train_df['event_time'].dt.dayofweek
    test_df['day_of_week'] = test_df['event_time'].dt.dayofweek
    
    train_day_stats = train_df.groupby('user_session')['day_of_week'].agg(['mean', 'std']).reset_index()
    test_day_stats = test_df.groupby('user_session')['day_of_week'].agg(['mean', 'std']).reset_index()
    
    # Merge temporal features
    train_temporal = train_temporal.merge(train_hour_stats, on='user_session')
    train_temporal = train_temporal.merge(train_day_stats, on='user_session')
    
    test_temporal = test_temporal.merge(test_hour_stats, on='user_session')
    test_temporal = test_temporal.merge(test_day_stats, on='user_session')
    
    return train_temporal, test_temporal

def merge_all_features(train_features, test_features, train_patterns, test_patterns, 
                      train_temporal, test_temporal):
    """Merge all feature sets"""
    print("=== MERGING ALL FEATURES ===")
    
    # Merge train features
    train_final = train_features.merge(train_patterns, left_index=True, right_index=True)
    train_final = train_final.merge(train_temporal, on='user_session')
    
    # Merge test features
    test_final = test_features.merge(test_patterns, left_index=True, right_index=True)
    test_final = test_final.merge(test_temporal, on='user_session')
    
    print(f"Final train features: {train_final.shape}")
    print(f"Final test features: {test_final.shape}")
    
    return train_final, test_final

def train_graph_models(train_features, test_features):
    """Train models using graph features"""
    print("=== TRAINING GRAPH-BASED MODELS ===")
    
    # Select features (exclude non-numeric and target)
    exclude_cols = ['user_session', 'user_id', 'session_value', 'min', 'max']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features:")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    X = train_features[feature_cols].fillna(0)
    y = train_features['session_value']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
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

def make_graph_predictions(results, test_features, feature_cols, scaler):
    """Make predictions using graph-based models"""
    print("=== MAKING GRAPH-BASED PREDICTIONS ===")
    
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

def create_graph_submission(test_features, predictions, filename='graph_analysis_submission.csv'):
    """Create submission file from graph analysis"""
    print(f"Creating graph analysis submission...")
    
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Graph analysis submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def visualize_graph(G, filename='graph_visualization.png'):
    """Create a visualization of the graph"""
    print("Creating graph visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes by type
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
    category_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'category']
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='blue', 
                          node_size=50, alpha=0.7, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=product_nodes, node_color='red', 
                          node_size=30, alpha=0.7, label='Products')
    nx.draw_networkx_nodes(G, pos, nodelist=category_nodes, node_color='green', 
                          node_size=40, alpha=0.7, label='Categories')
    
    plt.title('User-Product-Category Network Graph')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph visualization saved to {filename}")

def main():
    """Main function"""
    print("=== GRAPH-BASED DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create graph
        G, all_data = create_user_product_graph(train_df, test_df)
        
        # Extract graph features
        train_features, test_features = extract_graph_features(G, train_df, test_df)
        
        # Create interaction patterns
        train_patterns, test_patterns = create_interaction_patterns(train_df, test_df)
        
        # Create temporal features
        train_temporal, test_temporal = create_temporal_features(train_df, test_df)
        
        # Merge all features
        train_final, test_final = merge_all_features(
            train_features, test_features, train_patterns, test_patterns,
            train_temporal, test_temporal
        )
        
        # Train models
        results, feature_cols, scaler = train_graph_models(train_final, test_final)
        
        # Make predictions
        predictions = make_graph_predictions(results, test_final, feature_cols, scaler)
        
        # Create submission
        submission = create_graph_submission(test_final, predictions)
        
        # Create visualization
        visualize_graph(G)
        
        print("\n=== GRAPH-BASED SOLUTION COMPLETED! ===")
        print("Your graph analysis submission is ready!")
        
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
