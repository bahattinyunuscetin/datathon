#!/usr/bin/env python3
"""
Deep Learning Solution for Datathon
Using transformers, sequence modeling, and advanced neural networks
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SessionDataset(Dataset):
    """Custom dataset for session data"""
    def __init__(self, sessions, targets=None, max_length=50):
        self.sessions = sessions
        self.targets = targets
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        
        # Pad or truncate to max_length
        if len(session) > self.max_length:
            session = session[:self.max_length]
        else:
            session = session + [0] * (self.max_length - len(session))
        
        session_tensor = torch.tensor(session, dtype=torch.long)
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return session_tensor, target
        else:
            return session_tensor

class TransformerModel(nn.Module):
    """Transformer-based model for session prediction"""
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Final layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x.squeeze()

class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism"""
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = attn_out.mean(dim=1)
        
        # Final layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x.squeeze()

class TabularNN(nn.Module):
    """Neural network for tabular features"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(TabularNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

def create_sequence_features(train_df, test_df):
    """Create sequence-based features"""
    print("Creating sequence features...")
    
    # Create event type mapping
    event_mapping = {'VIEW': 1, 'ADD_CART': 2, 'REMOVE_CART': 3, 'BUY': 4}
    
    # Create session sequences
    train_sessions = train_df.groupby('user_session')['event_type'].apply(
        lambda x: [event_mapping[event] for event in x]
    ).reset_index()
    
    test_sessions = test_df.groupby('user_session')['event_type'].apply(
        lambda x: [event_mapping[event] for event in x]
    ).reset_index()
    
    # Get targets for training
    train_targets = train_df.groupby('user_session')['session_value'].first().values
    
    return train_sessions, test_sessions, train_targets, len(event_mapping) + 1

def create_advanced_tabular_features(train_df, test_df):
    """Create advanced tabular features"""
    print("Creating advanced tabular features...")
    
    # Time-based features
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    test_df['event_time'] = pd.to_datetime(test_df['event_time'])
    
    # Session-level features
    train_features = train_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique',
        'user_id': 'first',
        'session_value': 'first'
    }).reset_index()
    
    test_features = test_df.groupby('user_session').agg({
        'event_type': ['count', 'nunique'],
        'product_id': ['nunique', 'count'],
        'category_id': 'nunique',
        'user_id': 'first'
    }).reset_index()
    
    # Flatten columns
    train_features.columns = ['user_session', 'event_count', 'event_type_count', 
                            'product_count', 'total_views', 'category_count', 'user_id', 'session_value']
    test_features.columns = ['user_session', 'event_count', 'event_type_count', 
                           'product_count', 'total_views', 'category_count', 'user_id']
    
    # Add time features
    train_session_times = train_df.groupby('user_session')['event_time'].agg(['min', 'max'])
    train_session_times['duration'] = (train_session_times['max'] - train_session_times['min']).dt.total_seconds() / 60
    
    test_session_times = test_df.groupby('user_session')['event_time'].agg(['min', 'max'])
    test_session_times['duration'] = (test_session_times['max'] - test_session_times['min']).dt.total_seconds() / 60
    
    train_features = train_features.merge(train_session_times, left_on='user_session', right_index=True)
    test_features = test_features.merge(test_session_times, left_on='user_session', right_index=True)
    
    # Add user behavior features
    train_user_sessions = train_df.groupby('user_id')['user_session'].nunique().reset_index()
    train_user_sessions.columns = ['user_id', 'user_total_sessions']
    
    test_user_sessions = test_df.groupby('user_id')['user_session'].nunique().reset_index()
    test_user_sessions.columns = ['user_id', 'user_total_sessions']
    
    train_features = train_features.merge(train_user_sessions, on='user_id')
    test_features = test_features.merge(test_user_sessions, on='user_id')
    
    # Add interaction features
    train_features['engagement_score'] = (train_features['product_count'] * train_features['total_views']) / (train_features['event_count'] + 1)
    test_features['engagement_score'] = (test_features['product_count'] * test_features['total_views']) / (test_features['event_count'] + 1)
    
    return train_features, test_features

def train_deep_models(train_sessions, test_sessions, train_targets, vocab_size, 
                      train_features, test_features, epochs=50):
    """Train deep learning models"""
    print("Training deep learning models...")
    
    # Prepare sequence data
    train_sequences = train_sessions['event_type'].values
    test_sequences = test_sessions['event_type'].values
    
    # Create datasets
    train_dataset = SessionDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize models
    transformer_model = TransformerModel(vocab_size).to(device)
    lstm_model = LSTMWithAttention(vocab_size).to(device)
    
    # Prepare tabular features
    tabular_cols = ['event_count', 'event_type_count', 'product_count', 'total_views', 
                    'category_count', 'duration', 'user_total_sessions', 'engagement_score']
    
    X_tabular = train_features[tabular_cols].fillna(0).values
    y_tabular = train_features['session_value'].values
    
    # Scale tabular features
    scaler = StandardScaler()
    X_tabular_scaled = scaler.fit_transform(X_tabular)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_tabular_scaled, y_tabular, test_size=0.2, random_state=42
    )
    
    # Initialize tabular NN
    tabular_model = TabularNN(len(tabular_cols)).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Training functions
    def train_transformer():
        print("Training Transformer...")
        optimizer = optim.AdamW(transformer_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        transformer_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = transformer_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Transformer Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_lstm():
        print("Training LSTM with Attention...")
        optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = lstm_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_tabular():
        print("Training Tabular Neural Network...")
        optimizer = optim.AdamW(tabular_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        tabular_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = tabular_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    val_outputs = tabular_model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    print(f"Tabular Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    # Train all models
    train_transformer()
    train_lstm()
    train_tabular()
    
    return transformer_model, lstm_model, tabular_model, scaler, tabular_cols

def make_deep_predictions(transformer_model, lstm_model, tabular_model, 
                         test_sessions, test_features, scaler, tabular_cols):
    """Make predictions using deep learning models"""
    print("Making deep learning predictions...")
    
    # Prepare test data
    test_sequences = test_sessions['event_type'].values
    test_dataset = SessionDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get sequence predictions
    transformer_model.eval()
    lstm_model.eval()
    
    transformer_preds = []
    lstm_preds = []
    
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            
            # Transformer predictions
            trans_pred = transformer_model(batch_x).cpu().numpy()
            transformer_preds.extend(trans_pred)
            
            # LSTM predictions
            lstm_pred = lstm_model(batch_x).cpu().numpy()
            lstm_preds.extend(lstm_pred)
    
    # Get tabular predictions
    X_test_tabular = test_features[tabular_cols].fillna(0).values
    X_test_scaled = scaler.transform(X_test_tabular)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    tabular_model.eval()
    with torch.no_grad():
        tabular_preds = tabular_model(X_test_tensor).cpu().numpy()
    
    # Ensemble predictions (weighted average)
    ensemble_preds = (0.4 * np.array(transformer_preds) + 
                      0.4 * np.array(lstm_preds) + 
                      0.2 * np.array(tabular_preds))
    
    print(f"Transformer predictions: {np.array(transformer_preds).min():.2f} to {np.array(transformer_preds).max():.2f}")
    print(f"LSTM predictions: {np.array(lstm_preds).min():.2f} to {np.array(lstm_preds).max():.2f}")
    print(f"Tabular predictions: {np.array(tabular_preds).min():.2f} to {np.array(tabular_preds).max():.2f}")
    print(f"Ensemble predictions: {ensemble_preds.min():.2f} to {ensemble_preds.max():.2f}")
    
    return ensemble_preds

def create_deep_submission(test_features, predictions, filename='deep_learning_submission.csv'):
    """Create submission file from deep learning predictions"""
    print(f"Creating deep learning submission...")
    
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"Deep learning submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== DEEP LEARNING DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create sequence features
        train_sessions, test_sessions, train_targets, vocab_size = create_sequence_features(train_df, test_df)
        
        # Create tabular features
        train_features, test_features = create_advanced_tabular_features(train_df, test_df)
        
        # Train deep learning models
        transformer_model, lstm_model, tabular_model, scaler, tabular_cols = train_deep_models(
            train_sessions, test_sessions, train_targets, vocab_size, 
            train_features, test_features, epochs=30
        )
        
        # Make predictions
        predictions = make_deep_predictions(
            transformer_model, lstm_model, tabular_model,
            test_sessions, test_features, scaler, tabular_cols
        )
        
        # Create submission
        submission = create_deep_submission(test_features, predictions)
        
        print("\n=== DEEP LEARNING SOLUTION COMPLETED! ===")
        print("Your deep learning submission is ready!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
