# Tron Game AI

An AI competition framework for the classic Tron game, featuring both rule-based expert systems and learning agents that improve through self-play.

## Overview

This project pits two different AI approaches against each other:
- **Expert System**: Uses static evaluation, pathfinding, and space control heuristics
- **Learning Agent**: A Deep Q-Network (DQN) that learns strategy through reinforcement learning

Watch them battle it out and compare classical AI vs modern machine learning approaches!

## Features

- **Complete Game Engine** (`tron-game.py`)
  - Headless mode for fast training
  - Visual mode with pygame for watching matches
  - **Enhanced metrics system:**
    - Controlled territory (flood fill for actual accessible space)
    - Death type analysis (wall, own trail, opponent trail, head-on)
    - Directional distribution tracking
    - Trail length vs controlled area
  - Tournament system for running multiple games
  
- **Expert System Agent** (`tron-expert.py`)
  - Flood fill for space evaluation
  - Multiple difficulty levels (defensive, balanced, aggressive)
  - Advanced territory control strategies
  - Simple snake baseline agent
  
- **Learning Agent** (`tron-ai.py`)
  - Deep Q-Network with experience replay
  - Custom state encoding
  - Progressive training curriculum
  - Model saving and loading

- **Phased Training System** (`train_phased_ai.py`)
  - **Phase 1**: Train AI to match/beat Expert
  - **Phase 2**: Train enhanced AI to beat Phase 1
  - Progressive difficulty scaling
  - Comprehensive evaluation tournaments
  - Final 3-way showdown

## Installation

### Requirements
- Python 3.7+
- NumPy
- Pygame (optional, for visualization)

### Install Dependencies

```bash
pip install numpy pygame
```

Or without visualization:
```bash
pip install numpy
```

## Quick Start

### 1. Watch Expert Agents Battle

```bash
python tron-expert.py
```

This will:
- Run tournaments between different expert configurations
- Show enhanced metrics (controlled territory, death analysis)
- Display a visual match between advanced and aggressive agents

### 2. Phased AI Training (Recommended)

```bash
python train_phased_ai.py --full
```

This runs the complete training pipeline:
- **Phase 1**: Train AI to beat Expert system → saves `models/phase1_model.pkl`
- **Phase 2**: Train enhanced AI to beat Phase 1 → saves `models/phase2_model.pkl`
- **Ultimate Showdown**: All three agents compete in round-robin tournament

Or train phases individually:
```bash
# Phase 1 only (1000 episodes)
python train_phased_ai.py --phase 1 --episodes 1000

# Phase 2 only (800 episodes)
python train_phased_ai.py --phase 2 --episodes 800

# Just the showdown (requires trained models)
python train_phased_ai.py --showdown
```

### 3. Basic AI Training

```bash
python tron-ai.py
```

This will:
- Train a DQN agent from scratch
- Simple curriculum: SimpleSnake → Expert
- Evaluate and save model

### 4. Run Custom Matches

```python
from tron_game import TronGame, run_tournament
from tron_expert import ExpertAgent, AdvancedExpertAgent
from tron_ai import DQNAgent

# Create agents
expert = ExpertAgent(lookahead=3, aggression=0.5)
advanced = AdvancedExpertAgent(lookahead=5, aggression=0.5)

# Single visual match
game = TronGame(width=30, height=30, headless=False)
stats = game.play_match(expert, advanced, fps=10, verbose=True)
game.close()

# Tournament (headless)
run_tournament(expert, advanced, num_games=100)
```

## Architecture

### Game Engine (`tron-game.py`)

**TronGame Class:**
- Grid-based game state (default 40x40)
- Collision detection
- Observation API for agents
- **Enhanced metrics collection:**
  - **Controlled Territory**: Flood fill algorithm calculates actual accessible space for each player
  - **Death Type Analysis**: Categorizes losses as wall collision, self-collision, opponent trail hit, or head-on crash
  - **Directional Distribution**: Tracks percentage of time spent moving in each direction
  - **Trail Length**: Number of cells traversed (different from controlled territory)

**Key Methods:**
- `reset()`: Start new game
- `step(p1_action, p2_action)`: Execute one turn
- `get_observation()`: Get current state for agents
- `play_match(agent1, agent2)`: Run complete match with detailed statistics
- `render()`: Visualize game state

**Tournament System:**
- `run_tournament(agent1, agent2, num_games)`: Run multiple games with aggregated statistics
- Death analysis across all games
- Territory control comparison
- Performance metrics

### Phased Training System (`train_phased_ai.py`)

**Training Pipeline:**

1. **Phase 1 - Beat the Expert** (1000 episodes)
   - Stage 1.1: Learn basics against SimpleSnake (200 episodes)
   - Stage 1.2: Face defensive expert (300 episodes)
   - Stage 1.3: Master balanced expert (300 episodes)
   - Stage 1.4: Challenge advanced expert (200 episodes)
   - Goal: Achieve 45%+ win rate vs expert
   - Save as `models/phase1_model.pkl`

2. **Phase 2 - Surpass Phase 1** (800 episodes)
   - Uses deeper network (256-128-64 vs 128-64)
   - Lower learning rate for fine-tuning
   - Cross-training to prevent overfitting
   - Goal: Achieve 55%+ win rate vs Phase 1
   - Save as `models/phase2_model.pkl`

3. **Ultimate Showdown**
   - Round-robin tournament: Expert vs Phase1 vs Phase2
   - Detailed performance comparison
   - Visual demonstration match

**Architecture Differences:**
- Phase 1: [128, 64] hidden layers, LR=0.001
- Phase 2: [256, 128, 64] hidden layers, LR=0.0005, larger memory buffer

### Expert System (`tron-expert.py`)

**ExpertAgent:**
- Flood fill for space evaluation
- Lookahead planning
- Configurable aggression parameter

**Strategy:**
1. Avoid immediate collisions
2. Maximize reachable space (flood fill)
3. Maintain multiple escape routes
4. Control center territory (early game)
5. Adjust aggression based on space advantage

**AdvancedExpertAgent:**
- Territory control evaluation
- Opponent path cutting
- Space partitioning

**SimpleSnakeAgent:**
- Basic survival logic
- Useful as baseline opponent

### Learning Agent (`tron-ai.py`)

**DQNAgent:**
- Neural network: Input → [128] → [64] → 4 outputs (directions)
- Experience replay buffer (10,000 transitions)
- Epsilon-greedy exploration (1.0 → 0.01)
- Target network for stable learning

**State Encoding (135 features):**
- 11×11 local grid view (121 features)
- Distance to walls (4 features)
- Distance to opponent (2 features)
- Current direction one-hot (4 features)
- Direction safety checks (4 features)

**Training Process:**
1. Initialize with high exploration (ε=1.0)
2. Play games, store experiences
3. Train on batches from replay buffer
4. Gradually reduce exploration
5. Update target network every 100 steps

## Usage Examples

### Example 1: Compare Different Expert Strategies

```python
from tron_expert import ExpertAgent
from tron_game import run_tournament

defensive = ExpertAgent(aggression=0.2)
aggressive = ExpertAgent(aggression=0.8)

results = run_tournament(defensive, aggressive, num_games=200)
print(f"Defensive win rate: {results['agent1_win_rate']:.1%}")
```

### Example 2: Train Custom DQN

```python
from tron_ai import DQNAgent, train_agent
from tron_expert import SimpleSnakeAgent

# Create agent with custom parameters
agent = DQNAgent(
    hidden_sizes=[256, 128, 64],  # Deeper network
    learning_rate=0.0005,
    epsilon_decay=0.998  # Slower exploration decay
)

# Train
opponent = SimpleSnakeAgent()
train_agent(agent, opponent, episodes=1000, save_path="my_model.pkl")
```

### Example 3: Load and Evaluate Trained Agent

```python
from tron_ai import DQNAgent
from tron_expert import AdvancedExpertAgent
from tron_game import run_tournament

# Load trained agent
agent = DQNAgent(state_size=135)
agent.load("tron_model.pkl")
agent.epsilon = 0.0  # No exploration for evaluation

# Evaluate
expert = AdvancedExpertAgent()
results = run_tournament(agent, expert, num_games=100)
```

### Example 4: Custom Grid Size

```python
from tron_game import TronGame
from tron_expert import ExpertAgent

# Larger arena
game = TronGame(width=60, height=60, headless=False)
agent1 = ExpertAgent(aggression=0.3)
agent2 = ExpertAgent(aggression=0.7)

stats = game.play_match(agent1, agent2, max_turns=2000, fps=15)
game.close()
```

## Training Tips

### For DQN Agent:

1. **Start Simple**: Train against SimpleSnake before Expert
2. **Progressive Difficulty**: Gradually increase opponent strength
3. **Longer Training**: 1000+ episodes for noticeable improvement
4. **Hyperparameter Tuning**:
   - Increase `hidden_sizes` for more complex strategy
   - Lower `learning_rate` (0.0001-0.001) for stability
   - Adjust `epsilon_decay` to balance exploration/exploitation
   - Increase `memory_size` for more diverse experiences

5. **Monitor Progress**: Watch win rate and average reward trends

### Curriculum Learning Example:

```python
# Phase 1: Learn basics
train_agent(dqn, SimpleSnakeAgent(), episodes=300)

# Phase 2: Learn tactics
train_agent(dqn, ExpertAgent(aggression=0.3), episodes=400)

# Phase 3: Master strategy
train_agent(dqn, AdvancedExpertAgent(), episodes=500)
```

## Performance Benchmarks

### Expected Results After Training

**Expert Agents (baseline):**
- AdvancedExpert vs SimpleSnake: ~95% win rate
- AdvancedExpert vs Balanced Expert: ~65% win rate
- Expert agents show consistent performance

**Phase 1 AI (after 1000 episodes):**
- vs SimpleSnake: ~80-90% win rate
- vs Balanced Expert: ~45-55% win rate  
- vs Advanced Expert: ~40-50% win rate

**Phase 2 AI (after 800 additional episodes):**
- vs Phase 1 AI: ~55-65% win rate
- vs Advanced Expert: ~50-60% win rate
- vs SimpleSnake: ~90-95% win rate

**Death Analysis Patterns:**
- SimpleSnake: 100% wall collisions (basic pathfinding)
- Defensive Expert: High own-trail deaths (too conservative)
- Aggressive Expert: Mixed wall/opponent trail deaths (risky play)
- Learning Agents: Evolving patterns as training progresses

**Territory Control:**
- Winners typically control 800-1200 cells on default 40×40 grid
- Controlled territory is the true measure of dominance
- Trail length alone is misleading (both players move equally)

*Note: Results vary based on grid size, training duration, and random seed.*

## Phased Training Methodology

### Why Phased Training?

Traditional approach: Train one AI until it's "good enough"
**Problem**: Plateaus quickly, lacks competitive pressure

Phased approach: Each generation must surpass the previous
**Benefits**:
- Progressive difficulty scaling
- Competitive evolution
- Clear performance milestones
- Multiple agent variants for comparison

### Training Curriculum

**Phase 1 - Foundation Building:**
1. SimpleSnake (basic survival) → Learn collision avoidance
2. Defensive Expert (conservative) → Learn space management
3. Balanced Expert (moderate) → Develop tactical play
4. Advanced Expert (sophisticated) → Master strategic planning

**Phase 2 - Competitive Evolution:**
1. Phase 1 AI (primary opponent) → Exploit learned weaknesses
2. Advanced Expert (cross-training) → Prevent overfitting
3. Phase 1 AI (refinement) → Solidify superiority

### Architecture Evolution

| | Phase 1 | Phase 2 |
|---|---|---|
| Network | [128, 64] | [256, 128, 64] |
| Learning Rate | 0.001 | 0.0005 |
| Memory Size | 15,000 | 20,000 |
| Batch Size | 64 | 128 |
| Gamma | 0.95 | 0.97 |
| Focus | Beat expert | Beat Phase 1 |

### Expected Training Time

On a typical modern CPU:
- Phase 1 (1000 episodes): ~15-25 minutes
- Phase 2 (800 episodes): ~20-30 minutes  
- Total pipeline: ~45-60 minutes

Headless mode achieves 30-50 games/second.

## Project Structure

```
tron-ai/
├── README.md                 # This file
├── tron-game.py             # Game engine with enhanced metrics
├── tron-expert.py           # Expert system agents
├── tron-ai.py               # DQN learning agent
├── train_phased_ai.py       # Phased training pipeline (NEW!)
├── models/                  # Saved AI models (created during training)
│   ├── phase1_model.pkl     # Phase 1 AI (beats expert)
│   └── phase2_model.pkl     # Phase 2 AI (beats Phase 1)
└── Tron Game Files/         # Original reference implementation
    └── onefile/
        └── tron.py
```

## Enhanced Metrics System

The game now tracks comprehensive gameplay statistics:

### Territory Metrics
- **Trail Length**: Number of cells traversed (always equal for both players or differs by 1)
- **Controlled Territory**: Actual accessible space calculated via flood fill algorithm
  - This is the true measure of board control
  - Calculated at game end to determine winner if max turns reached
  - Winner has larger controlled territory, not just longer trail

### Death Analysis
Every loss is categorized by type:
- **Wall**: Crashed into arena boundary
- **Own Trail**: Ran into own path (suicide/trapped)
- **Opponent Trail**: Hit opponent's path (got cut off)
- **Head Collision**: Both players collided head-on

### Movement Patterns
- **Directional Distribution**: Percentage of moves in each direction (UP/RIGHT/DOWN/LEFT)
- Helps identify movement biases and strategies
- Useful for analyzing learned behaviors

### Tournament Statistics
When running tournaments, you get:
- Win rates and tie percentages
- Average controlled territory per player
- Death type distribution across all games
- Average game length (turns)
- Games per second performance metric

## Customization

### Creating Your Own Agent

Any agent must implement:

```python
class MyAgent:
    def get_action(self, obs, player_num):
        """
        Args:
            obs: Dictionary with game state
            player_num: 1 or 2
            
        Returns:
            Direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        # Your strategy here
        return Direction.RIGHT
```

### Observation Dictionary

```python
obs = {
    'grid': np.array,          # Game grid (height × width)
    'p1_pos': [row, col],      # Player 1 position
    'p2_pos': [row, col],      # Player 2 position
    'p1_dir': Direction,       # Player 1 direction
    'p2_dir': Direction,       # Player 2 direction
    'turn': int,               # Current turn number
    'p1_trail_length': int,    # P1 cells traversed
    'p2_trail_length': int,    # P2 cells traversed
}
```

Note: Controlled territory is calculated at game end, not included in real-time observations.

## Future Enhancements

Potential improvements:
- [ ] Add more learning algorithms (PPO, A3C)
- [ ] Implement self-play training
- [ ] Add tournament bracket system
- [ ] Create web interface
- [ ] Support for 3+ players
- [ ] Add replay saving/loading
- [ ] Implement Monte Carlo Tree Search agent
- [ ] Add adaptive difficulty

## Contributing

Feel free to experiment with:
- New agent strategies
- Different neural network architectures
- Alternative state representations
- Training curricula
- Visualization improvements

## License

This is an educational project for learning AI and game development.

## Acknowledgments

- Original Tron game concept
- Inspired by classic arcade gameplay
- Built for comparing AI approaches in competitive environments

