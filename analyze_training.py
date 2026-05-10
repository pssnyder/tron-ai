"""
Training Results Analyzer
Evaluate and visualize the performance of trained models
"""

import pickle
import numpy as np
from tron_game import TronGame
from tron_expert import ExpertAgent, AdvancedExpertAgent, SimpleSnakeAgent
from tron_ai import DQNAgent


def analyze_model(model_path, model_name, architecture):
    """Analyze a trained model's performance"""
    print("\n" + "=" * 70)
    print(f"ANALYZING: {model_name}")
    print("=" * 70)
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✓ Model loaded successfully")
        print(f"Training step: {data.get('training_step', 'unknown')}")
        print(f"Final epsilon: {data.get('epsilon', 'unknown')}")
        
        # Load model into agent
        agent = DQNAgent(state_size=135, hidden_sizes=architecture)
        agent.model.weights = data['weights']
        agent.model.biases = data['biases']
        agent.epsilon = 0.0  # Greedy evaluation
        
        # Test against various opponents
        opponents = [
            ("SimpleSnake", SimpleSnakeAgent()),
            ("Expert (Defensive)", ExpertAgent(aggression=0.3)),
            ("Expert (Balanced)", ExpertAgent(aggression=0.5)),
            ("Expert (Aggressive)", ExpertAgent(aggression=0.8)),
            ("Advanced Expert", AdvancedExpertAgent()),
        ]
        
        print(f"\nTesting {model_name} (greedy policy, ε=0.0):")
        print("-" * 70)
        
        game = TronGame(headless=True)
        
        for opp_name, opponent in opponents:
            wins = 0
            losses = 0
            ties = 0
            total_territory = 0
            total_turns = 0
            
            num_games = 100
            for _ in range(num_games):
                game.reset()
                done = False
                
                while not done:
                    obs = game.get_observation()
                    action1 = agent.get_action(obs, player_num=1)
                    action2 = opponent.get_action(obs, player_num=2)
                    done, winner = game.step(action1, action2)
                
                stats = game.get_game_stats()
                total_turns += stats['turns']
                total_territory += stats['p1_territory']
                
                if stats['winner'] == 1:
                    wins += 1
                elif stats['winner'] == 2:
                    losses += 1
                else:
                    ties += 1
            
            win_rate = wins / num_games
            avg_territory = total_territory / num_games
            avg_turns = total_turns / num_games
            
            status = "✓" if win_rate >= 0.45 else "⚠" if win_rate >= 0.30 else "✗"
            print(f"{status} vs {opp_name:25} Win: {win_rate*100:5.1f}%  " +
                  f"Territory: {avg_territory:6.1f}  Turns: {avg_turns:5.1f}")
        
        return agent
        
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def compare_models():
    """Compare Phase 1 vs Phase 2 performance"""
    print("\n" + "=" * 70)
    print("PHASE 1 vs PHASE 2 COMPARISON")
    print("=" * 70)
    
    try:
        # Load both models
        with open('models/phase1_model.pkl', 'rb') as f:
            phase1_data = pickle.load(f)
        with open('models/phase2_model.pkl', 'rb') as f:
            phase2_data = pickle.load(f)
        
        phase1 = DQNAgent(state_size=135, hidden_sizes=[128, 64])
        phase1.model.weights = phase1_data['weights']
        phase1.model.biases = phase1_data['biases']
        phase1.epsilon = 0.0
        
        phase2 = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
        phase2.model.weights = phase2_data['weights']
        phase2.model.biases = phase2_data['biases']
        phase2.epsilon = 0.0
        
        print("\nHead-to-head: Phase 2 vs Phase 1 (100 games)")
        
        game = TronGame(headless=True)
        p2_wins = 0
        p1_wins = 0
        ties = 0
        
        for _ in range(100):
            game.reset()
            done = False
            
            while not done:
                obs = game.get_observation()
                action1 = phase2.get_action(obs, player_num=1)  # Phase 2 as P1
                action2 = phase1.get_action(obs, player_num=2)  # Phase 1 as P2
                done, winner = game.step(action1, action2)
            
            stats = game.get_game_stats()
            if stats['winner'] == 1:
                p2_wins += 1
            elif stats['winner'] == 2:
                p1_wins += 1
            else:
                ties += 1
        
        print(f"Phase 2 wins: {p2_wins} ({p2_wins}%)")
        print(f"Phase 1 wins: {p1_wins} ({p1_wins}%)")
        print(f"Ties: {ties}")
        
        if p2_wins >= 55:
            print("\n✓ Phase 2 achieves target performance (55%+ vs Phase 1)")
        else:
            print(f"\n⚠ Phase 2 below target (need 55%, got {p2_wins}%)")
        
    except FileNotFoundError as e:
        print(f"✗ Could not load models: {e}")


def show_network_details():
    """Display neural network architecture details"""
    print("\n" + "=" * 70)
    print("NEURAL NETWORK ARCHITECTURES")
    print("=" * 70)
    
    models = [
        ("Phase 1", "models/phase1_model.pkl", [128, 64]),
        ("Phase 2", "models/phase2_model.pkl", [256, 128, 64]),
    ]
    
    for name, path, expected_arch in models:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n{name} Model:")
            print(f"  Input size: 135 features")
            
            weights = data['weights']
            biases = data['biases']
            total_params = 0
            
            for i, (W, b) in enumerate(zip(weights, biases)):
                params = W.size + b.size
                total_params += params
                print(f"  Layer {i+1}: {W.shape[1]} → {W.shape[0]} ({params:,} parameters)")
            
            print(f"  Total parameters: {total_params:,}")
            
        except FileNotFoundError:
            print(f"\n{name} Model: Not found")


if __name__ == "__main__":
    print("=" * 70)
    print("TRON AI TRAINING ANALYSIS")
    print("=" * 70)
    
    # Show network architectures
    show_network_details()
    
    # Analyze Phase 1
    phase1_agent = analyze_model(
        "models/phase1_model.pkl",
        "Phase 1 AI",
        [128, 64]
    )
    
    # Analyze Phase 2
    phase2_agent = analyze_model(
        "models/phase2_model.pkl",
        "Phase 2 AI",
        [256, 128, 64]
    )
    
    # Compare them
    if phase1_agent and phase2_agent:
        compare_models()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  python tournament_system.py          # Run full tournament")
    print("  python tournament_system.py --quick  # Quick test (10 games/matchup)")
