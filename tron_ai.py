"""
Tron Learning Agent
A reinforcement learning agent that learns to play Tron through self-play.

Uses a simple Deep Q-Network (DQN) with experience replay.
"""

import numpy as np
import random
from collections import deque
import pickle
import os
from tron_game import Direction, TronGame


class NeuralNetwork:
    """Simple feedforward neural network using numpy."""
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of outputs (4 for directions)
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Forward pass through network."""
        self.activations = [x]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:  # Hidden layers
                a = self.relu(z)
            else:  # Output layer (linear)
                a = z
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, x, y_true):
        """Backward pass and weight update."""
        m = x.shape[0]
        
        # Forward pass to get activations
        y_pred = self.forward(x)
        
        # Output layer gradient
        delta = y_pred - y_true
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b
            
            # Propagate to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.relu_derivative(self.activations[i])
    
    def predict(self, x):
        """Predict output for input x."""
        return self.forward(x)


class DQNAgent:
    """
    Deep Q-Network agent for Tron.
    
    Features:
    - Experience replay for stable learning
    - Epsilon-greedy exploration
    - Target network for stable Q-learning
    - State encoding with local view
    """
    
    def __init__(self, 
                 state_size: int = 84,
                 hidden_sizes: list = [128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state representation
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            memory_size: Size of replay memory
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = 4  # Four directions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.name = "DQN"
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.model = NeuralNetwork(state_size, hidden_sizes, self.action_size, learning_rate)
        self.target_model = NeuralNetwork(state_size, hidden_sizes, self.action_size, learning_rate)
        self.update_target_model()
        
        # Training stats
        self.training_step = 0
    
    def update_target_model(self):
        """Copy weights from model to target model."""
        self.target_model.weights = [w.copy() for w in self.model.weights]
        self.target_model.biases = [b.copy() for b in self.model.biases]
    
    def encode_state(self, obs, player_num):
        """
        Encode game observation into neural network input.
        
        Features:
        - Local grid view (11x11 centered on player)
        - Distance to walls
        - Distance to opponent
        - Current direction
        - Available space estimate
        
        Args:
            obs: Game observation
            player_num: Player number (1 or 2)
            
        Returns:
            Encoded state vector
        """
        grid = obs['grid']
        my_pos = obs['p1_pos'] if player_num == 1 else obs['p2_pos']
        opp_pos = obs['p2_pos'] if player_num == 1 else obs['p1_pos']
        my_dir = obs['p1_dir'] if player_num == 1 else obs['p2_dir']
        
        height, width = grid.shape
        features = []
        
        # Local grid view (11x11 = 121 cells, but flatten to smaller representation)
        view_size = 5
        for dr in range(-view_size, view_size + 1):
            for dc in range(-view_size, view_size + 1):
                r, c = my_pos[0] + dr, my_pos[1] + dc
                
                if r < 0 or r >= height or c < 0 or c >= width:
                    features.append(-1)  # Wall
                elif grid[r, c] == 0:
                    features.append(0)  # Empty
                elif grid[r, c] == player_num:
                    features.append(0.5)  # My trail
                else:
                    features.append(1)  # Opponent trail
        
        # Distance to walls (4 values)
        features.append(my_pos[0] / height)  # Distance to top
        features.append((height - 1 - my_pos[0]) / height)  # Distance to bottom
        features.append(my_pos[1] / width)  # Distance to left
        features.append((width - 1 - my_pos[1]) / width)  # Distance to right
        
        # Distance to opponent (2 values)
        features.append((my_pos[0] - opp_pos[0]) / height)  # Vertical distance
        features.append((my_pos[1] - opp_pos[1]) / width)  # Horizontal distance
        
        # Current direction (one-hot encoded, 4 values)
        dir_encoding = [0, 0, 0, 0]
        dir_encoding[int(my_dir)] = 1
        features.extend(dir_encoding)
        
        # Check safety in each direction (4 values)
        for direction in Direction:
            next_pos = self._get_next_pos(my_pos, direction)
            features.append(1 if self._is_safe(grid, next_pos) else 0)
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _get_next_pos(self, pos, direction):
        """Calculate next position."""
        row, col = pos
        if direction == Direction.UP:
            return [row - 1, col]
        elif direction == Direction.DOWN:
            return [row + 1, col]
        elif direction == Direction.LEFT:
            return [row, col - 1]
        elif direction == Direction.RIGHT:
            return [row, col + 1]
    
    def _is_safe(self, grid, pos):
        """Check if position is safe."""
        row, col = pos
        height, width = grid.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return grid[row, col] == 0
    
    def _get_valid_actions(self, current_dir):
        """Get valid actions (excluding 180-degree turn)."""
        all_actions = list(range(4))
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        opp_action = int(opposite[Direction(current_dir)])
        return [a for a in all_actions if a != opp_action]
    
    def get_action(self, obs, player_num, training=False):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            obs: Game observation
            player_num: Player number
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Direction to move
        """
        state = self.encode_state(obs, player_num)
        current_dir = obs['p1_dir'] if player_num == 1 else obs['p2_dir']
        valid_actions = self._get_valid_actions(current_dir)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            q_values = self.model.predict(state)[0]
            # Mask invalid actions
            masked_q = q_values.copy()
            for a in range(4):
                if a not in valid_actions:
                    masked_q[a] = -np.inf
            action = np.argmax(masked_q)
        
        return Direction(action)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch from replay memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.vstack([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Predict Q-values
        current_q = self.model.predict(states)
        next_q = self.target_model.predict(next_states)
        
        # Update Q-values with Bellman equation
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train model
        self.model.backward(states, target_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % 100 == 0:
            self.update_target_model()
    
    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'weights': self.model.weights,
            'biases': self.model.biases,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model.weights = model_data['weights']
            self.model.biases = model_data['biases']
            self.epsilon = model_data['epsilon']
            self.training_step = model_data['training_step']
            self.update_target_model()
            print(f"Model loaded from {filepath}")
            return True
        return False


def train_agent(agent, opponent, episodes=1000, save_every=100, save_path="tron_model.pkl"):
    """
    Train the DQN agent against an opponent.
    
    Args:
        agent: DQN agent to train
        opponent: Opponent agent (can be expert or another DQN)
        episodes: Number of training episodes
        save_every: Save model every N episodes
        save_path: Path to save model
    """
    game = TronGame(width=30, height=30, headless=True)
    
    print(f"Training DQN agent for {episodes} episodes...")
    print(f"Opponent: {opponent.name if hasattr(opponent, 'name') else 'Unknown'}")
    
    wins = 0
    total_rewards = []
    
    for episode in range(episodes):
        obs = game.reset()
        done = False
        episode_reward = 0
        
        states = []
        actions = []
        rewards_p1 = []
        
        while not done:
            # Agent is player 1
            state = agent.encode_state(obs, 1)
            action = agent.get_action(obs, 1, training=True)
            
            # Opponent is player 2
            opp_action = opponent.get_action(obs, 2)
            
            # Step
            next_obs, reward_p1, reward_p2, done = game.step(action, opp_action)
            
            # Store experience
            next_state = agent.encode_state(next_obs, 1)
            agent.remember(state, int(action), reward_p1, next_state, done)
            
            episode_reward += reward_p1
            obs = next_obs
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
        
        total_rewards.append(episode_reward)
        if game.winner == 1:
            wins += 1
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            win_rate = wins / (episode + 1)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Wins: {wins} ({win_rate:.1%}) | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Save model
        if (episode + 1) % save_every == 0:
            agent.save(save_path)
    
    # Final save
    agent.save(save_path)
    
    print(f"\nTraining complete!")
    print(f"Final win rate: {wins/episodes:.1%}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    game.close()
    return agent


if __name__ == "__main__":
    from tron_expert import ExpertAgent, SimpleSnakeAgent
    from tron_game import run_tournament
    
    print("DQN Agent Training Demo")
    print("="*50)
    
    # Create agents
    dqn_agent = DQNAgent(
        state_size=135,  # 11x11 grid (121) + walls (4) + opp_dist (2) + dir (4) + safety (4)
        hidden_sizes=[128, 64],
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64
    )
    
    # Try to load existing model
    if dqn_agent.load("tron_model.pkl"):
        print("Loaded existing model. Continuing training...\n")
    else:
        print("No existing model found. Training from scratch...\n")
    
    # Train against simple opponent first
    print("Phase 1: Training against SimpleSnake")
    simple_opponent = SimpleSnakeAgent()
    train_agent(dqn_agent, simple_opponent, episodes=500, save_every=100)
    
    # Train against expert
    print("\nPhase 2: Training against Expert")
    expert_opponent = ExpertAgent(lookahead=3, aggression=0.5)
    train_agent(dqn_agent, expert_opponent, episodes=500, save_every=100)
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluation")
    print("="*50)
    
    # Set epsilon to 0 for evaluation (pure exploitation)
    eval_agent = DQNAgent(state_size=135)
    eval_agent.load("tron_model.pkl")
    eval_agent.epsilon = 0.0
    
    print("\nDQN vs SimpleSnake:")
    run_tournament(eval_agent, simple_opponent, num_games=100, headless=True, verbose=True)
    
    print("\nDQN vs Expert:")
    run_tournament(eval_agent, expert_opponent, num_games=100, headless=True, verbose=True)
    
    # Visual match
    print("\nVisual match: DQN vs Expert")
    game = TronGame(width=30, height=30, headless=False)
    stats = game.play_match(eval_agent, expert_opponent, fps=10, verbose=True)
    game.close()
