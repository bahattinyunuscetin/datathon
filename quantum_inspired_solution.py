#!/usr/bin/env python3
"""
Quantum-Inspired Solution for Datathon
Using quantum-like algorithms and quantum machine learning concepts
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class QuantumInspiredFeatures:
    """Quantum-inspired feature engineering"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_superposition_features(self, train_df, test_df):
        """Create superposition-like features (linear combinations)"""
        print("=== CREATING SUPERPOSITION FEATURES ===")
        
        # Create session-level features
        train_features = train_df.groupby('user_session').agg({
            'event_type': ['count', 'nunique'],
            'product_id': ['nunique', 'count'],
            'category_id': 'nunique',
            'user_id': 'first'
        }).reset_index()
        
        test_features = test_df.groupby('user_session').agg({
            'event_type': ['count', 'nunique'],
            'product_id': ['nunique', 'count'],
            'category_id': 'nunique',
            'user_id': 'first'
        }).reset_index()
        
        # Flatten columns
        train_features.columns = ['user_session', 'event_count', 'event_type_count', 'product_count', 'total_views', 'category_count', 'user_id']
        test_features.columns = ['user_session', 'event_count', 'event_type_count', 'product_count', 'total_views', 'category_count', 'user_id']
        
        # Create superposition features (quantum-like linear combinations)
        # These represent "quantum states" of sessions
        
        # Superposition 1: Event-Product entanglement
        train_features['event_product_superposition'] = (
            np.sin(train_features['event_count'] * np.pi / 100) * 
            np.cos(train_features['product_count'] * np.pi / 50)
        )
        test_features['event_product_superposition'] = (
            np.sin(test_features['event_count'] * np.pi / 100) * 
            np.cos(test_features['product_count'] * np.pi / 50)
        )
        
        # Superposition 2: Category-Event entanglement
        train_features['category_event_superposition'] = (
            np.exp(-train_features['category_count'] / 10) * 
            np.sin(train_features['event_count'] * np.pi / 50)
        )
        test_features['category_event_superposition'] = (
            np.exp(-test_features['category_count'] / 10) * 
            np.sin(test_features['event_count'] * np.pi / 50)
        )
        
        # Superposition 3: Complex quantum state
        train_features['complex_quantum_state'] = (
            train_features['event_count'] * np.exp(1j * train_features['product_count'] * np.pi / 100)
        ).real  # Take real part
        test_features['complex_quantum_state'] = (
            test_features['event_count'] * np.exp(1j * test_features['product_count'] * np.pi / 100)
        ).real  # Take real part
        
        return train_features, test_features
    
    def create_entanglement_features(self, train_df, test_df):
        """Create entanglement-like features (correlations)"""
        print("=== CREATING ENTANGLEMENT FEATURES ===")
        
        # Calculate correlations between different aspects
        def calculate_entanglement(group):
            events = group['event_type'].tolist()
            products = group['product_id'].tolist()
            categories = group['category_id'].tolist()
            
            # Event-Product entanglement (correlation)
            event_product_corr = np.corrcoef(
                [len(events)] * len(products), 
                [len(products)] * len(events)
            )[0, 1] if len(events) > 1 and len(products) > 1 else 0
            
            # Event-Category entanglement
            event_category_corr = np.corrcoef(
                [len(events)] * len(categories), 
                [len(categories)] * len(events)
            )[0, 1] if len(events) > 1 and len(categories) > 1 else 0
            
            # Product-Category entanglement
            product_category_corr = np.corrcoef(
                [len(products)] * len(categories), 
                [len(categories)] * len(products)
            )[0, 1] if len(products) > 1 and len(categories) > 1 else 0
            
            # Quantum coherence (measure of order)
            coherence = np.std([len(events), len(products), len(categories)]) / np.mean([len(events), len(products), len(categories)]) if np.mean([len(events), len(products), len(categories)]) > 0 else 0
            
            return pd.Series({
                'event_product_entanglement': event_product_corr,
                'event_category_entanglement': event_category_corr,
                'product_category_entanglement': product_category_corr,
                'quantum_coherence': coherence
            })
        
        # Apply to train and test
        train_entanglement = train_df.groupby('user_session').apply(calculate_entanglement).reset_index()
        test_entanglement = test_df.groupby('user_session').apply(calculate_entanglement).reset_index()
        
        return train_entanglement, test_entanglement
    
    def create_quantum_tunneling_features(self, train_df, test_df):
        """Create quantum tunneling-like features (jumps between states)"""
        print("=== CREATING QUANTUM TUNNELING FEATURES ===")
        
        def calculate_tunneling(group):
            events = group['event_type'].tolist()
            
            # Quantum tunneling: sudden jumps between event types
            tunneling_events = 0
            for i in range(1, len(events)):
                # Detect sudden changes (tunneling)
                if events[i] != events[i-1]:
                    tunneling_events += 1
            
            # Tunneling probability
            tunneling_prob = tunneling_events / (len(events) - 1) if len(events) > 1 else 0
            
            # Quantum barrier height (resistance to change)
            barrier_height = 1 - tunneling_prob
            
            # Energy levels (different event types)
            energy_levels = len(set(events))
            
            # Ground state (most common event)
            if events:
                ground_state = max(set(events), key=events.count)
                ground_state_energy = events.count(ground_state) / len(events)
            else:
                ground_state_energy = 0
            
            return pd.Series({
                'tunneling_events': tunneling_events,
                'tunneling_probability': tunneling_prob,
                'quantum_barrier_height': barrier_height,
                'energy_levels': energy_levels,
                'ground_state_energy': ground_state_energy
            })
        
        # Apply to train and test
        train_tunneling = train_df.groupby('user_session').apply(calculate_tunneling).reset_index()
        test_tunneling = test_df.groupby('user_session').apply(calculate_tunneling).reset_index()
        
        return train_tunneling, test_tunneling
    
    def create_quantum_interference_features(self, train_df, test_df):
        """Create quantum interference-like features (constructive/destructive)"""
        print("=== CREATING QUANTUM INTERFERENCE FEATURES ===")
        
        def calculate_interference(group):
            events = group['event_type'].tolist()
            products = group['product_id'].tolist()
            
            # Constructive interference (reinforcing patterns)
            constructive_patterns = 0
            for i in range(len(events) - 1):
                if events[i] == events[i+1]:  # Same event type
                    constructive_patterns += 1
            
            # Destructive interference (canceling patterns)
            destructive_patterns = 0
            for i in range(len(events) - 1):
                if events[i] != events[i+1]:  # Different event types
                    destructive_patterns += 1
            
            # Interference pattern strength
            total_patterns = len(events) - 1 if len(events) > 1 else 0
            interference_strength = (constructive_patterns - destructive_patterns) / total_patterns if total_patterns > 0 else 0
            
            # Quantum phase (pattern complexity)
            phase = np.angle(complex(constructive_patterns, destructive_patterns))
            
            # Wave function amplitude
            amplitude = np.sqrt(constructive_patterns**2 + destructive_patterns**2)
            
            return pd.Series({
                'constructive_interference': constructive_patterns,
                'destructive_interference': destructive_patterns,
                'interference_strength': interference_strength,
                'quantum_phase': phase,
                'wave_amplitude': amplitude
            })
        
        # Apply to train and test
        train_interference = train_df.groupby('user_session').apply(calculate_interference).reset_index()
        test_interference = test_df.groupby('user_session').apply(calculate_interference).reset_index()
        
        return train_interference, test_interference
    
    def create_quantum_measurement_features(self, train_df, test_df):
        """Create quantum measurement-like features (observable properties)"""
        print("=== CREATING QUANTUM MEASUREMENT FEATURES ===")
        
        def calculate_measurement(group):
            events = group['event_type'].tolist()
            products = group['product_id'].tolist()
            categories = group['category_id'].tolist()
            
            # Measurement uncertainty (Heisenberg principle)
            event_uncertainty = np.std([len(events), len(products), len(categories)]) if len(events) > 0 else 0
            
            # Observable properties
            event_momentum = len(events)  # "momentum" of events
            product_position = len(products)  # "position" in product space
            category_spin = len(categories)  # "spin" in category space
            
            # Commutation relations (quantum mechanics)
            # [Event, Product] = Event*Product - Product*Event
            commutation = abs(len(events) * len(products) - len(products) * len(events))
            
            # Quantum numbers
            principal_quantum_number = len(events)
            angular_momentum_quantum_number = len(products)
            magnetic_quantum_number = len(categories)
            
            return pd.Series({
                'measurement_uncertainty': event_uncertainty,
                'event_momentum': event_momentum,
                'product_position': product_position,
                'category_spin': category_spin,
                'commutation_relation': commutation,
                'principal_quantum_number': principal_quantum_number,
                'angular_momentum_quantum_number': angular_momentum_quantum_number,
                'magnetic_quantum_number': magnetic_quantum_number
            })
        
        # Apply to train and test
        train_measurement = train_df.groupby('user_session').apply(calculate_measurement).reset_index()
        test_measurement = test_df.groupby('user_session').apply(calculate_measurement).reset_index()
        
        return train_measurement, test_measurement

class QuantumInspiredModel:
    """Quantum-inspired machine learning model"""
    
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.quantum_state = None
        self.feature_names = []
    
    def create_quantum_circuit_features(self, X):
        """Create quantum circuit-inspired features"""
        print("=== CREATING QUANTUM CIRCUIT FEATURES ===")
        
        # Simulate quantum gates
        n_samples, n_features = X.shape
        
        # Hadamard gate effect (superposition)
        hadamard_features = np.zeros((n_samples, n_features))
        for i in range(n_features):
            hadamard_features[:, i] = (X[:, i] + np.random.normal(0, 0.1, n_samples)) / np.sqrt(2)
        
        # CNOT gate effect (entanglement)
        cnot_features = np.zeros((n_samples, n_features))
        for i in range(0, n_features - 1, 2):
            cnot_features[:, i] = X[:, i]
            cnot_features[:, i+1] = X[:, i+1] ^ (X[:, i] > np.median(X[:, i]))
        
        # Phase gate effect (rotation)
        phase_features = np.zeros((n_samples, n_features))
        for i in range(n_features):
            phase = np.pi * (i + 1) / n_features
            phase_features[:, i] = X[:, i] * np.cos(phase) + np.random.normal(0, 0.1, n_samples) * np.sin(phase)
        
        # Combine quantum features
        quantum_features = np.concatenate([
            hadamard_features, cnot_features, phase_features
        ], axis=1)
        
        # Update feature names
        self.feature_names = [f'hadamard_{i}' for i in range(n_features)] + \
                           [f'cnot_{i}' for i in range(n_features)] + \
                           [f'phase_{i}' for i in range(n_features)]
        
        return quantum_features
    
    def quantum_ensemble_predict(self, X, base_models):
        """Quantum-inspired ensemble prediction"""
        print("=== QUANTUM ENSEMBLE PREDICTION ===")
        
        # Get predictions from base models
        predictions = []
        for model in base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Quantum superposition of predictions
        # Create quantum weights based on model performance
        quantum_weights = np.exp(1j * np.linspace(0, 2*np.pi, len(base_models)))
        quantum_weights = quantum_weights / np.sum(np.abs(quantum_weights))
        
        # Apply quantum weights
        quantum_predictions = np.zeros(predictions.shape[1])
        for i in range(predictions.shape[1]):
            quantum_predictions[i] = np.real(
                np.sum(quantum_weights * predictions[:, i])
            )
        
        return quantum_predictions

def create_quantum_features(train_df, test_df):
    """Create all quantum-inspired features"""
    print("=== CREATING QUANTUM-INSPIRED FEATURES ===")
    
    quantum_features = QuantumInspiredFeatures()
    
    # Create different types of quantum features
    train_superposition, test_superposition = quantum_features.create_superposition_features(train_df, test_df)
    train_entanglement, test_entanglement = quantum_features.create_entanglement_features(train_df, test_df)
    train_tunneling, test_tunneling = quantum_features.create_quantum_tunneling_features(train_df, test_df)
    train_interference, test_interference = quantum_features.create_quantum_interference_features(train_df, test_df)
    train_measurement, test_measurement = quantum_features.create_quantum_measurement_features(train_df, test_df)
    
    # Merge all quantum features
    train_quantum = train_superposition.merge(train_entanglement, on='user_session', how='left')
    train_quantum = train_quantum.merge(train_tunneling, on='user_session', how='left')
    train_quantum = train_quantum.merge(train_interference, on='user_session', how='left')
    train_quantum = train_quantum.merge(train_measurement, on='user_session', how='left')
    
    test_quantum = test_superposition.merge(test_entanglement, on='user_session', how='left')
    test_quantum = test_quantum.merge(test_tunneling, on='user_session', how='left')
    test_quantum = test_quantum.merge(test_interference, on='user_session', how='left')
    test_quantum = test_quantum.merge(test_measurement, on='user_session', how='left')
    
    # Fill NaN values
    train_quantum = train_quantum.fillna(0)
    test_quantum = test_quantum.fillna(0)
    
    print(f"Quantum features created - Train: {train_quantum.shape}, Test: {test_quantum.shape}")
    
    return train_quantum, test_quantum

def train_quantum_models(train_features, test_features):
    """Train models using quantum-inspired features"""
    print("=== TRAINING QUANTUM-INSPIRED MODELS ===")
    
    # Get training targets
    train_targets = train_features.groupby('user_session')['session_value'].first()
    
    # Select quantum features
    exclude_cols = ['user_session', 'user_id', 'session_value']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} quantum features:")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    X = train_features[feature_cols].fillna(0)
    y = train_targets
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create quantum-inspired model
    quantum_model = QuantumInspiredModel(n_qubits=8)
    
    # Create quantum circuit features
    X_quantum = quantum_model.create_quantum_circuit_features(X_scaled)
    
    # Train multiple models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    models = {
        'Quantum_RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Quantum_GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Quantum_Ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_quantum, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # Train on full data
        model.fit(X_quantum, y)
        
        results[name] = {
            'model': model,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std()
        }
        
        print(f"{name} - CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    
    return results, feature_cols, scaler, quantum_model

def make_quantum_predictions(results, test_features, feature_cols, scaler, quantum_model):
    """Make predictions using quantum-inspired models"""
    print("=== MAKING QUANTUM PREDICTIONS ===")
    
    X_test = test_features[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    
    # Create quantum circuit features for test data
    X_test_quantum = quantum_model.create_quantum_circuit_features(X_test_scaled)
    
    # Get predictions from all models
    predictions = {}
    for name, result in results.items():
        pred = result['model'].predict(X_test_quantum)
        predictions[name] = pred
        print(f"{name} predictions: {pred.min():.2f} to {pred.max():.2f}")
    
    # Quantum ensemble prediction
    base_models = [result['model'] for result in results.values()]
    quantum_ensemble_pred = quantum_model.quantum_ensemble_predict(X_test_quantum, base_models)
    
    print(f"\nQuantum ensemble predictions: {quantum_ensemble_pred.min():.2f} to {quantum_ensemble_pred.max():.2f}")
    
    return quantum_ensemble_pred

def create_quantum_submission(test_features, predictions, filename='quantum_inspired_submission.csv'):
    """Create submission file from quantum-inspired predictions"""
    print(f"Creating quantum-inspired submission...")
    
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Quantum-inspired submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== QUANTUM-INSPIRED DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create quantum-inspired features
        train_quantum, test_quantum = create_quantum_features(train_df, test_df)
        
        # Merge with basic features
        train_basic = train_df.groupby('user_session').agg({
            'event_type': 'count',
            'product_id': 'nunique',
            'category_id': 'nunique',
            'user_id': 'first',
            'session_value': 'first'
        }).reset_index()
        
        test_basic = test_df.groupby('user_session').agg({
            'event_type': 'count',
            'product_id': 'nunique',
            'category_id': 'nunique',
            'user_id': 'first'
        }).reset_index()
        
        # Merge features
        train_final = train_basic.merge(train_quantum, on='user_session', how='left')
        test_final = test_basic.merge(test_quantum, on='user_session', how='left')
        
        # Train quantum models
        results, feature_cols, scaler, quantum_model = train_quantum_models(train_final, test_final)
        
        # Make predictions
        predictions = make_quantum_predictions(results, test_final, feature_cols, scaler, quantum_model)
        
        # Create submission
        submission = create_quantum_submission(test_final, predictions)
        
        print("\n=== QUANTUM-INSPIRED SOLUTION COMPLETED! ===")
        print("Your quantum-inspired submission is ready!")
        
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
