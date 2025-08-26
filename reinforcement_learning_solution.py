#!/usr/bin/env python3
"""
Reinforcement Learning Solution for Datathon
Using Q-learning and policy gradient methods for session optimization
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import random
import warnings
warnings.filterwarnings('ignore')

class SessionEnvironment:
    """Environment for reinforcement learning"""
    def __init__(self, train_data):
        self.train_data = train_data
        self.reset()
        
        # Define actions (event types)
        self.actions = ['VIEW', 'ADD_CART', 'REMOVE_CART', 'BUY']
        self.action_space = len(self.actions)
        
        # Define states (session characteristics)
        self.state_features = [
            'event_count', 'product_count', 'category_count', 
            'duration_minutes', 'hour', 'day_of_week'
        ]
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_session = None
        self.current_step = 0
        self.session_history = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_session is None:
            return np.zeros(len(self.state_features))
        
        # Extract state features from current session
        session_data = self.train_data[self.train_data['user_session'] == self.current_session]
        
        state = []
        state.append(len(session_data))  # event_count
        state.append(session_data['product_id'].nunique())  # product_count
        state.append(session_data['category_id'].nunique())  # category_count
        
        # Duration
        if len(session_data) > 1:
            duration = (pd.to_datetime(session_data['event_time'].max()) - 
                       pd.to_datetime(session_data['event_time'].min())).total_seconds() / 60
        else:
            duration = 0
        state.append(duration)
        
        # Time features
        if len(session_data) > 0:
            hour = pd.to_datetime(session_data['event_time'].iloc[0]).hour
            day_of_week = pd.to_datetime(session_data['event_time'].iloc[0]).dayofweek
        else:
            hour = 0
            day_of_week = 0
        
        state.append(hour)
        state.append(day_of_week)
        
        return np.array(state)
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        if self.current_session is None:
            # Start new session
            available_sessions = self.train_data['user_session'].unique()
            self.current_session = np.random.choice(available_sessions)
            self.current_step = 0
        
        # Get current session data
        session_data = self.train_data[self.train_data['user_session'] == self.current_session]
        
        # Calculate reward based on action and session outcome
        reward = self._calculate_reward(action, session_data)
        
        # Move to next step
        self.current_step += 1
        
        # Check if session is done
        done = self.current_step >= len(session_data) or self.current_step >= 50
        
        if done:
            self.current_session = None
            self.current_step = 0
        
        next_state = self._get_state()
        
        return next_state, reward, done
    
    def _calculate_reward(self, action, session_data):
        """Calculate reward for action in current session"""
        if len(session_data) == 0:
            return 0
        
        # Get actual events in session
        actual_events = session_data['event_type'].tolist()
        
        # Reward based on action alignment with session goal
        if action == 'BUY' and 'BUY' in actual_events:
            reward = 10  # High reward for successful purchase
        elif action == 'ADD_CART' and 'ADD_CART' in actual_events:
            reward = 5   # Good reward for adding to cart
        elif action == 'VIEW' and 'VIEW' in actual_events:
            reward = 2   # Basic reward for viewing
        elif action == 'REMOVE_CART' and 'REMOVE_CART' in actual_events:
            reward = 1   # Low reward for removing from cart
        else:
            reward = -1  # Penalty for misaligned action
        
        # Additional reward for session value
        if 'session_value' in session_data.columns:
            session_value = session_data['session_value'].iloc[0]
            reward += session_value / 100  # Scale session value
        
        return reward

class QLearningAgent:
    """Q-learning agent for session optimization"""
    def __init__(self, state_size, action_size, learning_rate=0.1, epsilon=0.1, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tuple for hashing
        state_tuple = tuple(state)
        
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_tuple])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_tuple = tuple(state)
            next_state_tuple = tuple(next_state)
            
            # Current Q-value
            current_q = self.q_table[state_tuple][action]
            
            # Next Q-value
            if done:
                next_q = 0
            else:
                if next_state_tuple not in self.q_table:
                    self.q_table[next_state_tuple] = np.zeros(self.action_size)
                next_q = np.max(self.q_table[next_state_tuple])
            
            # Update Q-value
            new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q)
            self.q_table[state_tuple][action] = new_q
    
    def train(self, env, episodes=1000, max_steps=100):
        """Train the agent"""
        print("=== TRAINING Q-LEARNING AGENT ===")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train on batch of experiences
            if len(self.memory) > 32:
                self.replay(32)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
    
    def get_session_policy(self, session_data):
        """Get optimal action policy for a session"""
        # Convert session data to state
        state = self._session_to_state(session_data)
        state_tuple = tuple(state)
        
        if state_tuple not in self.q_table:
            return np.random.choice(self.action_size)
        
        return np.argmax(self.q_table[state_tuple])
    
    def _session_to_state(self, session_data):
        """Convert session data to state vector"""
        state = []
        state.append(len(session_data))  # event_count
        state.append(session_data['product_id'].nunique())  # product_count
        state.append(session_data['category_id'].nunique())  # category_count
        
        # Duration
        if len(session_data) > 1:
            duration = (pd.to_datetime(session_data['event_time'].max()) - 
                       pd.to_datetime(session_data['event_time'].min())).total_seconds() / 60
        else:
            duration = 0
        state.append(duration)
        
        # Time features
        if len(session_data) > 0:
            hour = pd.to_datetime(session_data['event_time'].iloc[0]).hour
            day_of_week = pd.to_datetime(session_data['event_time'].iloc[0]).dayofweek
        else:
            hour = 0
            day_of_week = 0
        
        state.append(hour)
        state.append(day_of_week)
        
        return np.array(state)

class PolicyGradientAgent:
    """Policy gradient agent for session optimization"""
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Simple policy network weights
        self.weights = np.random.randn(state_size, action_size) * 0.01
        
        # Experience buffer
        self.memory = []
        
    def get_action(self, state, training=True):
        """Sample action from policy"""
        # Get action probabilities
        logits = np.dot(state, self.weights)
        probs = self._softmax(logits)
        
        if training:
            # Sample action
            action = np.random.choice(self.action_size, p=probs)
        else:
            # Choose best action
            action = np.argmax(probs)
        
        return action, probs
    
    def _softmax(self, x):
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def remember(self, state, action, reward, probs):
        """Store experience"""
        self.memory.append((state, action, reward, probs))
    
    def train(self, env, episodes=500, max_steps=100):
        """Train the agent using policy gradient"""
        print("=== TRAINING POLICY GRADIENT AGENT ===")
        
        for episode in range(episodes):
            state = env.reset()
            episode_rewards = []
            episode_probs = []
            
            for step in range(max_steps):
                action, probs = self.get_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                episode_rewards.append(reward)
                episode_probs.append(probs)
                
                state = next_state
                
                if done:
                    break
            
            # Update policy using episode rewards
            self._update_policy(episode_rewards, episode_probs)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    def _update_policy(self, rewards, probs):
        """Update policy weights using policy gradient"""
        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Update weights
        for i, (reward, prob) in enumerate(zip(discounted_rewards, probs)):
            # Simple policy gradient update
            gradient = reward * prob
            self.weights += self.learning_rate * gradient
    
    def _discount_rewards(self, rewards, gamma=0.95):
        """Calculate discounted rewards"""
        discounted = []
        running_sum = 0
        
        for reward in reversed(rewards):
            running_sum = reward + gamma * running_sum
            discounted.insert(0, running_sum)
        
        return np.array(discounted)

def create_rl_features(train_df, test_df, q_agent, pg_agent):
    """Create features using reinforcement learning agents"""
    print("=== CREATING RL FEATURES ===")
    
    # Create features for training data
    train_rl_features = []
    
    for session_id in train_df['user_session'].unique():
        session_data = train_df[train_df['user_session'] == session_id]
        
        # Get Q-learning policy
        q_action = q_agent.get_session_policy(session_data)
        
        # Get policy gradient action
        pg_action, pg_probs = pg_agent.get_action(q_agent._session_to_state(session_data), training=False)
        
        # Create RL features
        rl_features = {
            'user_session': session_id,
            'q_learning_action': q_action,
            'policy_gradient_action': pg_action,
            'pg_action_prob': pg_probs[pg_action],
            'action_consensus': 1 if q_action == pg_action else 0,
            'session_complexity': len(session_data),
            'optimal_path_score': _calculate_optimal_path_score(session_data)
        }
        
        train_rl_features.append(rl_features)
    
    # Create features for test data
    test_rl_features = []
    
    for session_id in test_df['user_session'].unique():
        session_data = test_df[test_df['user_session'] == session_id]
        
        # Get Q-learning policy
        q_action = q_agent.get_session_policy(session_data)
        
        # Get policy gradient action
        pg_action, pg_probs = pg_agent.get_action(q_agent._session_to_state(session_data), training=False)
        
        # Create RL features
        rl_features = {
            'user_session': session_id,
            'q_learning_action': q_action,
            'policy_gradient_action': pg_action,
            'pg_action_prob': pg_probs[pg_action],
            'action_consensus': 1 if q_action == pg_action else 0,
            'session_complexity': len(session_data),
            'optimal_path_score': _calculate_optimal_path_score(session_data)
        }
        
        test_rl_features.append(rl_features)
    
    # Convert to DataFrames
    train_rl_df = pd.DataFrame(train_rl_features)
    test_rl_df = pd.DataFrame(test_rl_features)
    
    return train_rl_df, test_rl_df

def _calculate_optimal_path_score(session_data):
    """Calculate optimal path score for session"""
    events = session_data['event_type'].tolist()
    
    # Define optimal path: VIEW -> ADD_CART -> BUY
    optimal_path = ['VIEW', 'ADD_CART', 'BUY']
    
    score = 0
    for i, event in enumerate(events):
        if i < len(optimal_path) and event == optimal_path[i]:
            score += 1
        elif event in optimal_path:
            # Partial credit for having the event
            score += 0.5
    
    return score / len(optimal_path) if len(optimal_path) > 0 else 0

def create_rl_submission(train_df, test_df, train_rl_features, test_rl_features, filename='rl_submission.csv'):
    """Create submission using RL features"""
    print("=== CREATING RL SUBMISSION ===")
    
    # Get training targets
    train_targets = train_df.groupby('user_session')['session_value'].first()
    
    # Merge RL features with basic features
    train_basic = train_df.groupby('user_session').agg({
        'event_type': 'count',
        'product_id': 'nunique',
        'category_id': 'nunique'
    }).reset_index()
    
    test_basic = test_df.groupby('user_session').agg({
        'event_type': 'count',
        'product_id': 'nunique',
        'category_id': 'nunique'
    }).reset_index()
    
    # Merge features
    train_final = train_basic.merge(train_rl_features, on='user_session', how='left')
    test_final = test_basic.merge(test_rl_features, on='user_session', how='left')
    
    # Select features
    feature_cols = ['event_type', 'product_id', 'category_id', 'q_learning_action', 
                   'policy_gradient_action', 'pg_action_prob', 'action_consensus', 
                   'session_complexity', 'optimal_path_score']
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    
    X = train_final[feature_cols].fillna(0)
    y = train_targets
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Make predictions
    X_test = test_final[feature_cols].fillna(0)
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_final['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative values
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    submission.to_csv(filename, index=False)
    print(f"RL submission saved to {filename}")
    
    print(f"Submission shape: {submission.shape}")
    print(f"Value range: {submission['session_value'].min():.4f} to {submission['session_value'].max():.4f}")
    
    return submission

def main():
    """Main function"""
    print("=== REINFORCEMENT LEARNING DATATHON SOLUTION ===\n")
    
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # Create environment
        env = SessionEnvironment(train_df)
        
        # Initialize agents
        state_size = len(env.state_features)
        action_size = env.action_space
        
        q_agent = QLearningAgent(state_size, action_size)
        pg_agent = PolicyGradientAgent(state_size, action_size)
        
        # Train agents
        print("Training agents...")
        q_agent.train(env, episodes=500)
        pg_agent.train(env, episodes=300)
        
        # Create RL features
        train_rl_features, test_rl_features = create_rl_features(train_df, test_df, q_agent, pg_agent)
        
        # Create submission
        submission = create_rl_submission(train_df, test_df, train_rl_features, test_rl_features)
        
        print("\n=== REINFORCEMENT LEARNING SOLUTION COMPLETED! ===")
        print("Your RL submission is ready!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
