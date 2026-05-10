"""
Phased Training Script for Tron AI

Training progression:
- Phase 1: Train AI to beat/match the expert system → save as phase1_model.pkl
- Phase 2: Train enhanced AI to beat Phase 1 AI → save as phase2_model.pkl

Final result: 3 competitive implementations
- Expert: Rule-based static evaluation
- Phase1 AI: Learning agent trained to beat expert
- Phase2 AI: Advanced learning agent trained to beat Phase1
"""

import numpy as np
import os
from tron_game import TronGame, run_tournament
from tron_expert import ExpertAgent, AdvancedExpertAgent, SimpleSnakeAgent
from tron_ai import DQNAgent, train_agent


def phase1_training(episodes=1000, save_path="models/phase1_model.pkl"):
    """
    Phase 1: Train AI to beat the expert system.
    
    Strategy:
    - Start with simple opponent (SimpleSnake) to learn basics
    - Progress to defensive expert
    - Finally train against balanced expert
    - Goal: Achieve 50%+ win rate against expert
    
    Args:
        episodes: Total training episodes
        save_path: Path to save the trained model
    
    Returns:
        Trained DQNAgent
    """
    print("="*70)
    print("PHASE 1 TRAINING: Learning to Beat the Expert")
    print("="*70)
    
    # Create Phase 1 agent
    phase1_agent = DQNAgent(
        state_size=135,
        hidden_sizes=[128, 64],
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=15000,
        batch_size=64
    )
    
    # Try to load existing Phase 1 model
    if os.path.exists(save_path):
        print(f"\nFound existing Phase 1 model at {save_path}")
        response = input("Continue training from checkpoint? (y/n): ")
        if response.lower() == 'y':
            phase1_agent.load(save_path)
            print("Loaded existing model. Continuing training...\n")
    
    # Training curriculum
    print("\n" + "-"*70)
    print("Stage 1.1: Learning Basics against SimpleSnake")
    print("-"*70)
    simple_opponent = SimpleSnakeAgent()
    train_agent(phase1_agent, simple_opponent, 
                episodes=int(episodes * 0.2), 
                save_every=50, 
                save_path=save_path)
    
    print("\n" + "-"*70)
    print("Stage 1.2: Facing Defensive Expert")
    print("-"*70)
    defensive_expert = ExpertAgent(lookahead=3, aggression=0.3)
    train_agent(phase1_agent, defensive_expert, 
                episodes=int(episodes * 0.3), 
                save_every=50, 
                save_path=save_path)
    
    print("\n" + "-"*70)
    print("Stage 1.3: Mastering Balanced Expert")
    print("-"*70)
    balanced_expert = ExpertAgent(lookahead=3, aggression=0.5)
    train_agent(phase1_agent, balanced_expert, 
                episodes=int(episodes * 0.3), 
                save_every=50, 
                save_path=save_path)
    
    print("\n" + "-"*70)
    print("Stage 1.4: Final Training against Advanced Expert")
    print("-"*70)
    advanced_expert = AdvancedExpertAgent(lookahead=5, aggression=0.5)
    train_agent(phase1_agent, advanced_expert, 
                episodes=int(episodes * 0.2), 
                save_every=50, 
                save_path=save_path)
    
    # Final save
    phase1_agent.save(save_path)
    
    # Evaluation
    print("\n" + "="*70)
    print("PHASE 1 EVALUATION")
    print("="*70)
    
    eval_agent = DQNAgent(state_size=135)
    eval_agent.load(save_path)
    eval_agent.epsilon = 0.0  # Pure exploitation
    
    print("\nPhase 1 AI vs SimpleSnake:")
    run_tournament(eval_agent, simple_opponent, num_games=100, headless=True, verbose=True)
    
    print("\nPhase 1 AI vs Balanced Expert:")
    results = run_tournament(eval_agent, balanced_expert, num_games=100, headless=True, verbose=True)
    
    print("\nPhase 1 AI vs Advanced Expert:")
    run_tournament(eval_agent, advanced_expert, num_games=100, headless=True, verbose=True)
    
    # Check if ready for Phase 2
    if results['agent1_win_rate'] >= 0.45:
        print("\n✓ Phase 1 COMPLETE! Win rate against expert: {:.1%}".format(results['agent1_win_rate']))
        print("  Ready for Phase 2 training!")
    else:
        print("\n⚠ Phase 1 win rate: {:.1%}".format(results['agent1_win_rate']))
        print("  Consider more training before Phase 2")
    
    return eval_agent


def phase2_training(episodes=800, save_path="models/phase2_model.pkl"):
    """
    Phase 2: Train enhanced AI to beat Phase 1 AI.
    
    Strategy:
    - Load Phase 1 agent as opponent
    - Use larger network and more sophisticated training
    - Focus on exploiting Phase 1's weaknesses
    - Goal: Achieve 60%+ win rate against Phase 1
    
    Args:
        episodes: Training episodes
        save_path: Path to save Phase 2 model
    
    Returns:
        Trained Phase 2 DQNAgent
    """
    print("\n" + "="*70)
    print("PHASE 2 TRAINING: Beating Phase 1 AI")
    print("="*70)
    
    # Load Phase 1 agent as opponent
    phase1_path = "models/phase1_model.pkl"
    if not os.path.exists(phase1_path):
        print(f"\nError: Phase 1 model not found at {phase1_path}")
        print("Please complete Phase 1 training first!")
        return None
    
    phase1_opponent = DQNAgent(state_size=135)
    phase1_opponent.load(phase1_path)
    phase1_opponent.epsilon = 0.1  # Small exploration to avoid predictability
    print(f"\nLoaded Phase 1 opponent from {phase1_path}")
    
    # Create Phase 2 agent with enhanced architecture
    phase2_agent = DQNAgent(
        state_size=135,
        hidden_sizes=[256, 128, 64],  # Deeper network
        learning_rate=0.0005,  # Lower learning rate for fine-tuning
        gamma=0.97,  # Higher discount for long-term planning
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.996,  # Slower decay
        memory_size=20000,  # Larger memory
        batch_size=128  # Larger batches
    )
    
    # Try to load existing Phase 2 model
    if os.path.exists(save_path):
        print(f"\nFound existing Phase 2 model at {save_path}")
        response = input("Continue training from checkpoint? (y/n): ")
        if response.lower() == 'y':
            phase2_agent.load(save_path)
            print("Loaded existing model. Continuing training...\n")
    
    # Training curriculum
    print("\n" + "-"*70)
    print("Stage 2.1: Learning Phase 1's Patterns")
    print("-"*70)
    train_agent(phase2_agent, phase1_opponent, 
                episodes=int(episodes * 0.4), 
                save_every=50, 
                save_path=save_path)
    
    # Mix in expert opponents to prevent overfitting
    print("\n" + "-"*70)
    print("Stage 2.2: Cross-training with Expert (prevent overfitting)")
    print("-"*70)
    advanced_expert = AdvancedExpertAgent(lookahead=5, aggression=0.6)
    train_agent(phase2_agent, advanced_expert, 
                episodes=int(episodes * 0.2), 
                save_every=50, 
                save_path=save_path)
    
    # Back to Phase 1 opponent
    print("\n" + "-"*70)
    print("Stage 2.3: Final Refinement against Phase 1")
    print("-"*70)
    train_agent(phase2_agent, phase1_opponent, 
                episodes=int(episodes * 0.4), 
                save_every=50, 
                save_path=save_path)
    
    # Final save
    phase2_agent.save(save_path)
    
    # Evaluation
    print("\n" + "="*70)
    print("PHASE 2 EVALUATION")
    print("="*70)
    
    eval_agent = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
    eval_agent.load(save_path)
    eval_agent.epsilon = 0.0
    
    # Reload Phase 1 for fair evaluation
    phase1_eval = DQNAgent(state_size=135)
    phase1_eval.load(phase1_path)
    phase1_eval.epsilon = 0.0
    
    print("\nPhase 2 AI vs Phase 1 AI:")
    results_vs_phase1 = run_tournament(eval_agent, phase1_eval, num_games=100, headless=True, verbose=True)
    
    print("\nPhase 2 AI vs Advanced Expert:")
    run_tournament(eval_agent, advanced_expert, num_games=100, headless=True, verbose=True)
    
    print("\nPhase 2 AI vs SimpleSnake (baseline):")
    simple_opponent = SimpleSnakeAgent()
    run_tournament(eval_agent, simple_opponent, num_games=100, headless=True, verbose=True)
    
    if results_vs_phase1['agent1_win_rate'] >= 0.55:
        print("\n✓ Phase 2 COMPLETE! Win rate vs Phase 1: {:.1%}".format(results_vs_phase1['agent1_win_rate']))
    else:
        print("\n⚠ Phase 2 win rate vs Phase 1: {:.1%}".format(results_vs_phase1['agent1_win_rate']))
        print("  Consider more training")
    
    return eval_agent


def ultimate_showdown():
    """
    Final tournament: All three implementations compete.
    
    Compares:
    - Expert System (AdvancedExpert)
    - Phase 1 AI
    - Phase 2 AI
    """
    print("\n" + "="*70)
    print("ULTIMATE SHOWDOWN: All Agents Compete!")
    print("="*70)
    
    # Load all agents
    expert = AdvancedExpertAgent(lookahead=5, aggression=0.5)
    expert.name = "Advanced Expert"
    
    phase1_agent = DQNAgent(state_size=135)
    if not phase1_agent.load("models/phase1_model.pkl"):
        print("Error: Phase 1 model not found!")
        return
    phase1_agent.epsilon = 0.0
    phase1_agent.name = "Phase 1 AI"
    
    phase2_agent = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
    if not phase2_agent.load("models/phase2_model.pkl"):
        print("Error: Phase 2 model not found!")
        return
    phase2_agent.epsilon = 0.0
    phase2_agent.name = "Phase 2 AI"
    
    # Round robin tournament
    print("\n" + "-"*70)
    print("Match 1: Expert vs Phase 1 AI")
    print("-"*70)
    results_1 = run_tournament(expert, phase1_agent, num_games=100, headless=True, verbose=True)
    
    print("\n" + "-"*70)
    print("Match 2: Expert vs Phase 2 AI")
    print("-"*70)
    results_2 = run_tournament(expert, phase2_agent, num_games=100, headless=True, verbose=True)
    
    print("\n" + "-"*70)
    print("Match 3: Phase 1 AI vs Phase 2 AI")
    print("-"*70)
    results_3 = run_tournament(phase1_agent, phase2_agent, num_games=100, headless=True, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL STANDINGS")
    print("="*70)
    
    # Calculate overall scores
    expert_score = results_1['agent1_win_rate'] + results_2['agent1_win_rate']
    phase1_score = (1 - results_1['agent1_win_rate']) + results_3['agent1_win_rate']
    phase2_score = (1 - results_2['agent1_win_rate']) + (1 - results_3['agent1_win_rate'])
    
    standings = [
        ("Advanced Expert", expert_score),
        ("Phase 1 AI", phase1_score),
        ("Phase 2 AI", phase2_score)
    ]
    standings.sort(key=lambda x: x[1], reverse=True)
    
    print("\nOverall Performance (combined win rates):")
    for i, (name, score) in enumerate(standings, 1):
        print(f"{i}. {name}: {score:.3f}")
    
    # Visual demonstration
    print("\n" + "-"*70)
    print("Visual Match: Phase 2 AI vs Expert")
    print("-"*70)
    print("Watch the final showdown!")
    
    game = TronGame(width=35, height=35, headless=False)
    stats = game.play_match(phase2_agent, expert, fps=12, verbose=True)
    game.close()


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phased Training for Tron AI")
    parser.add_argument('--phase', type=int, choices=[1, 2], 
                       help='Training phase (1 or 2)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    parser.add_argument('--showdown', action='store_true',
                       help='Run ultimate showdown with all agents')
    parser.add_argument('--full', action='store_true',
                       help='Run complete training pipeline (Phase 1 + Phase 2)')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    if args.full:
        # Full training pipeline
        print("Starting FULL TRAINING PIPELINE")
        print("This will take a while...\n")
        
        # Phase 1
        phase1_episodes = args.episodes if args.episodes else 1000
        phase1_training(episodes=phase1_episodes)
        
        # Phase 2
        phase2_episodes = int(phase1_episodes * 0.8)
        phase2_training(episodes=phase2_episodes)
        
        # Ultimate showdown
        ultimate_showdown()
        
    elif args.phase == 1:
        # Phase 1 only
        episodes = args.episodes if args.episodes else 1000
        phase1_training(episodes=episodes)
        
    elif args.phase == 2:
        # Phase 2 only
        episodes = args.episodes if args.episodes else 800
        phase2_training(episodes=episodes)
        
    elif args.showdown:
        # Just the showdown
        ultimate_showdown()
        
    else:
        # Interactive mode
        print("="*70)
        print("TRON AI - PHASED TRAINING SYSTEM")
        print("="*70)
        print("\nOptions:")
        print("1. Phase 1 Training (AI learns to beat Expert)")
        print("2. Phase 2 Training (Enhanced AI learns to beat Phase 1)")
        print("3. Ultimate Showdown (All agents compete)")
        print("4. Full Pipeline (Phase 1 + Phase 2 + Showdown)")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == '1':
            episodes = int(input("Training episodes (default 1000): ") or "1000")
            phase1_training(episodes=episodes)
        elif choice == '2':
            episodes = int(input("Training episodes (default 800): ") or "800")
            phase2_training(episodes=episodes)
        elif choice == '3':
            ultimate_showdown()
        elif choice == '4':
            episodes = int(input("Phase 1 episodes (default 1000): ") or "1000")
            phase1_training(episodes=episodes)
            phase2_training(episodes=int(episodes * 0.8))
            ultimate_showdown()
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
