"""
Tron AI Tournament System
Comprehensive tournament framework for testing multiple AI variants with detailed metrics
"""

import numpy as np
from tron_game import TronGame, Direction
from tron_expert import ExpertAgent, AdvancedExpertAgent, SimpleSnakeAgent
from tron_ai import DQNAgent
import pickle
import time
from collections import defaultdict
import json


class PlayerVariant:
    """Represents a specific player configuration"""
    def __init__(self, name, agent, description=""):
        self.name = name
        self.agent = agent
        self.description = description
        

class TournamentStats:
    """Tracks comprehensive tournament statistics"""
    def __init__(self):
        self.matchups = defaultdict(lambda: {
            'games': 0,
            'p1_wins': 0,
            'p2_wins': 0,
            'ties': 0,
            'total_turns': 0,
            'p1_territory': 0,
            'p2_territory': 0,
            'p1_deaths': defaultdict(int),
            'p2_deaths': defaultdict(int),
            'p1_directions': defaultdict(int),
            'p2_directions': defaultdict(int),
            'close_games': 0,  # games decided by <50 turns
            'long_games': 0,   # games >200 turns
        })
        
    def record_game(self, p1_name, p2_name, result):
        """Record a single game result"""
        key = f"{p1_name}_vs_{p2_name}"
        stats = self.matchups[key]
        
        stats['games'] += 1
        if result['winner'] == 1:
            stats['p1_wins'] += 1
        elif result['winner'] == 2:
            stats['p2_wins'] += 1
        else:
            stats['ties'] += 1
            
        stats['total_turns'] += result['turns']
        stats['p1_territory'] += result['p1_territory']
        stats['p2_territory'] += result['p2_territory']
        
        # Death analysis
        if result['p1_death_type']:
            stats['p1_deaths'][result['p1_death_type']] += 1
        if result['p2_death_type']:
            stats['p2_deaths'][result['p2_death_type']] += 1
            
        # Direction tracking
        for direction, count in result['p1_directions'].items():
            stats['p1_directions'][direction] += count
        for direction, count in result['p2_directions'].items():
            stats['p2_directions'][direction] += count
            
        # Game length classification
        if result['turns'] < 50:
            stats['close_games'] += 1
        elif result['turns'] > 200:
            stats['long_games'] += 1
            
    def get_summary(self, p1_name, p2_name):
        """Get summary statistics for a matchup"""
        key = f"{p1_name}_vs_{p2_name}"
        stats = self.matchups[key]
        
        if stats['games'] == 0:
            return None
            
        return {
            'games': stats['games'],
            'p1_win_rate': stats['p1_wins'] / stats['games'],
            'p2_win_rate': stats['p2_wins'] / stats['games'],
            'tie_rate': stats['ties'] / stats['games'],
            'avg_turns': stats['total_turns'] / stats['games'],
            'avg_p1_territory': stats['p1_territory'] / stats['games'],
            'avg_p2_territory': stats['p2_territory'] / stats['games'],
            'p1_deaths': dict(stats['p1_deaths']),
            'p2_deaths': dict(stats['p2_deaths']),
            'p1_direction_preference': self._get_direction_preference(stats['p1_directions']),
            'p2_direction_preference': self._get_direction_preference(stats['p2_directions']),
            'close_games': stats['close_games'],
            'long_games': stats['long_games'],
        }
        
    def _get_direction_preference(self, direction_counts):
        """Calculate direction usage percentages"""
        total = sum(direction_counts.values())
        if total == 0:
            return {}
        return {d: count/total for d, count in direction_counts.items()}
        
    def save_to_file(self, filename):
        """Save tournament stats to JSON"""
        data = {}
        for matchup, stats in self.matchups.items():
            # Convert defaultdicts to regular dicts for JSON serialization
            data[matchup] = {
                'games': stats['games'],
                'p1_wins': stats['p1_wins'],
                'p2_wins': stats['p2_wins'],
                'ties': stats['ties'],
                'avg_turns': stats['total_turns'] / stats['games'] if stats['games'] > 0 else 0,
                'p1_deaths': dict(stats['p1_deaths']),
                'p2_deaths': dict(stats['p2_deaths']),
                'p1_directions': dict(stats['p1_directions']),
                'p2_directions': dict(stats['p2_directions']),
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            

def create_player_variants():
    """Create multiple variants of each player type"""
    variants = []
    
    # === EXPERT VARIANTS ===
    variants.append(PlayerVariant(
        "Snake_Baseline",
        SimpleSnakeAgent(),
        "Basic survival logic - baseline opponent"
    ))
    
    variants.append(PlayerVariant(
        "Expert_Defensive",
        ExpertAgent(aggression=0.3, lookahead=3),
        "Conservative expert - prioritizes survival"
    ))
    
    variants.append(PlayerVariant(
        "Expert_Balanced",
        ExpertAgent(aggression=0.5, lookahead=3),
        "Balanced expert - default strategy"
    ))
    
    variants.append(PlayerVariant(
        "Expert_Aggressive",
        ExpertAgent(aggression=0.8, lookahead=3),
        "Aggressive expert - seeks opponent cuts"
    ))
    
    variants.append(PlayerVariant(
        "Expert_Advanced",
        AdvancedExpertAgent(aggression=0.6),
        "Advanced tactics with territory control"
    ))
    
    # === DQN VARIANTS - PHASE 1 ===
    try:
        # Load base Phase 1 model
        with open('models/phase1_model.pkl', 'rb') as f:
            phase1_data = pickle.load(f)
        
        # Standard Phase 1 (greedy)
        phase1_greedy = DQNAgent(state_size=135, hidden_sizes=[128, 64])
        phase1_greedy.model.weights = phase1_data['weights']
        phase1_greedy.model.biases = phase1_data['biases']
        phase1_greedy.epsilon = 0.0  # Greedy
        variants.append(PlayerVariant(
            "Phase1_Greedy",
            phase1_greedy,
            "Phase 1 AI - pure exploitation (ε=0)"
        ))
        
        # Phase 1 with slight exploration
        phase1_explore = DQNAgent(state_size=135, hidden_sizes=[128, 64])
        phase1_explore.model.weights = phase1_data['weights']
        phase1_explore.model.biases = phase1_data['biases']
        phase1_explore.epsilon = 0.05  # Small exploration
        variants.append(PlayerVariant(
            "Phase1_Cautious",
            phase1_explore,
            "Phase 1 AI - 5% exploration for variety"
        ))
        
        # Phase 1 moderate exploration
        phase1_balanced = DQNAgent(state_size=135, hidden_sizes=[128, 64])
        phase1_balanced.model.weights = phase1_data['weights']
        phase1_balanced.model.biases = phase1_data['biases']
        phase1_balanced.epsilon = 0.10
        variants.append(PlayerVariant(
            "Phase1_Adaptive",
            phase1_balanced,
            "Phase 1 AI - 10% exploration for adaptability"
        ))
        
    except FileNotFoundError:
        print("⚠ Phase 1 model not found - skipping Phase 1 variants")
        
    # === DQN VARIANTS - PHASE 2 ===
    try:
        with open('models/phase2_model.pkl', 'rb') as f:
            phase2_data = pickle.load(f)
        
        # Standard Phase 2 (greedy)
        phase2_greedy = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
        phase2_greedy.model.weights = phase2_data['weights']
        phase2_greedy.model.biases = phase2_data['biases']
        phase2_greedy.epsilon = 0.0
        variants.append(PlayerVariant(
            "Phase2_Greedy",
            phase2_greedy,
            "Phase 2 AI - maximum performance (ε=0)"
        ))
        
        # Phase 2 with slight exploration
        phase2_explore = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
        phase2_explore.model.weights = phase2_data['weights']
        phase2_explore.model.biases = phase2_data['biases']
        phase2_explore.epsilon = 0.05
        variants.append(PlayerVariant(
            "Phase2_Cautious",
            phase2_explore,
            "Phase 2 AI - 5% exploration"
        ))
        
        # Phase 2 moderate exploration
        phase2_balanced = DQNAgent(state_size=135, hidden_sizes=[256, 128, 64])
        phase2_balanced.model.weights = phase2_data['weights']
        phase2_balanced.model.biases = phase2_data['biases']
        phase2_balanced.epsilon = 0.10
        variants.append(PlayerVariant(
            "Phase2_Adaptive",
            phase2_balanced,
            "Phase 2 AI - 10% exploration for unpredictability"
        ))
        
    except FileNotFoundError:
        print("⚠ Phase 2 model not found - skipping Phase 2 variants")
    
    return variants


def run_matchup(variant1, variant2, num_games=100, verbose=False):
    """Run a matchup between two variants"""
    game = TronGame(headless=True)
    results = []
    
    for i in range(num_games):
        stats = game.play_match(variant1.agent, variant2.agent, verbose=False)
        
        # Convert direction percentages to move counts for this game
        p1_dir_counts = {}
        p2_dir_counts = {}
        for direction, pct in stats['p1_direction_distribution'].items():
            # Convert percentage back to approximate move count
            move_count = int(pct * stats['turns'] / 100)
            p1_dir_counts[str(direction)] = move_count
        for direction, pct in stats['p2_direction_distribution'].items():
            move_count = int(pct * stats['turns'] / 100)
            p2_dir_counts[str(direction)] = move_count
        
        # Convert to format expected by tournament stats
        result = {
            'winner': stats['winner'],
            'turns': stats['turns'],
            'p1_territory': stats['p1_controlled_territory'],
            'p2_territory': stats['p2_controlled_territory'],
            'p1_death_type': stats['p1_death_type'],
            'p2_death_type': stats['p2_death_type'],
            'p1_directions': p1_dir_counts,
            'p2_directions': p2_dir_counts,
        }
        results.append(result)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{num_games} games")
    
    return results


def run_round_robin_tournament(variants, games_per_matchup=100, save_results=True):
    """Run full round-robin tournament"""
    print("=" * 70)
    print("TRON AI ROUND-ROBIN TOURNAMENT")
    print("=" * 70)
    print(f"\nPlayers: {len(variants)}")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Total games: {len(variants) * (len(variants) - 1) * games_per_matchup}")
    print()
    
    tournament_stats = TournamentStats()
    matchup_count = 0
    total_matchups = len(variants) * (len(variants) - 1)
    
    start_time = time.time()
    
    for i, variant1 in enumerate(variants):
        for j, variant2 in enumerate(variants):
            if i == j:
                continue  # Skip self-matches
                
            matchup_count += 1
            print(f"\n[{matchup_count}/{total_matchups}] {variant1.name} vs {variant2.name}")
            print(f"  {variant1.description}")
            print(f"  vs {variant2.description}")
            
            results = run_matchup(variant1, variant2, games_per_matchup, verbose=True)
            
            # Record all games
            for result in results:
                tournament_stats.record_game(variant1.name, variant2.name, result)
            
            # Print quick summary
            summary = tournament_stats.get_summary(variant1.name, variant2.name)
            print(f"  ✓ {variant1.name}: {summary['p1_win_rate']*100:.1f}% | " +
                  f"{variant2.name}: {summary['p2_win_rate']*100:.1f}%")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TOURNAMENT COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Games/sec: {(total_matchups * games_per_matchup) / elapsed:.1f}")
    
    if save_results:
        filename = f"tournament_results_{int(time.time())}.json"
        tournament_stats.save_to_file(filename)
        print(f"\n✓ Results saved to {filename}")
    
    return tournament_stats


def print_leaderboard(tournament_stats, variants):
    """Print overall leaderboard across all matchups"""
    print("\n" + "=" * 70)
    print("OVERALL LEADERBOARD")
    print("=" * 70)
    
    # Calculate overall stats for each player
    player_stats = {}
    
    for variant in variants:
        total_wins = 0
        total_losses = 0
        total_ties = 0
        total_games = 0
        total_territory = 0
        death_types = defaultdict(int)
        
        # As player 1
        for other in variants:
            if variant.name == other.name:
                continue
            summary = tournament_stats.get_summary(variant.name, other.name)
            if summary:
                games = summary['games']
                total_games += games
                total_wins += int(summary['p1_win_rate'] * games)
                total_losses += int(summary['p2_win_rate'] * games)
                total_ties += int(summary['tie_rate'] * games)
                total_territory += summary['avg_p1_territory'] * games
                for death_type, count in summary['p1_deaths'].items():
                    death_types[death_type] += count
        
        # As player 2
        for other in variants:
            if variant.name == other.name:
                continue
            summary = tournament_stats.get_summary(other.name, variant.name)
            if summary:
                games = summary['games']
                total_games += games
                total_wins += int(summary['p2_win_rate'] * games)
                total_losses += int(summary['p1_win_rate'] * games)
                total_ties += int(summary['tie_rate'] * games)
                total_territory += summary['avg_p2_territory'] * games
                for death_type, count in summary['p2_deaths'].items():
                    death_types[death_type] += count
        
        if total_games > 0:
            player_stats[variant.name] = {
                'win_rate': total_wins / total_games,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'total_ties': total_ties,
                'total_games': total_games,
                'avg_territory': total_territory / total_games,
                'death_types': dict(death_types),
            }
    
    # Sort by win rate
    sorted_players = sorted(player_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Player':<25} {'Win Rate':<12} {'W-L-T':<15} {'Avg Territory':<15}")
    print("-" * 70)
    
    for rank, (name, stats) in enumerate(sorted_players, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<3} {name:<25} {stats['win_rate']*100:>6.1f}%     " +
              f"{stats['total_wins']:>3}-{stats['total_losses']:>3}-{stats['total_ties']:>3}      " +
              f"{stats['avg_territory']:>8.1f}")
    
    return sorted_players


def print_detailed_analysis(tournament_stats, player1_name, player2_name):
    """Print detailed head-to-head analysis"""
    summary = tournament_stats.get_summary(player1_name, player2_name)
    
    if not summary:
        print(f"No data for {player1_name} vs {player2_name}")
        return
    
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS: {player1_name} vs {player2_name}")
    print("=" * 70)
    
    print(f"\nGames Played: {summary['games']}")
    print(f"\nWin Rates:")
    print(f"  {player1_name}: {summary['p1_win_rate']*100:.1f}%")
    print(f"  {player2_name}: {summary['p2_win_rate']*100:.1f}%")
    print(f"  Ties: {summary['tie_rate']*100:.1f}%")
    
    print(f"\nAverage Game Length: {summary['avg_turns']:.1f} turns")
    print(f"  Close games (<50 turns): {summary['close_games']}")
    print(f"  Long games (>200 turns): {summary['long_games']}")
    
    print(f"\nAverage Controlled Territory:")
    print(f"  {player1_name}: {summary['avg_p1_territory']:.1f} cells")
    print(f"  {player2_name}: {summary['avg_p2_territory']:.1f} cells")
    
    print(f"\n{player1_name} Death Analysis:")
    for death_type, count in summary['p1_deaths'].items():
        pct = (count / summary['games']) * 100
        print(f"  {death_type}: {count} ({pct:.1f}%)")
    
    print(f"\n{player2_name} Death Analysis:")
    for death_type, count in summary['p2_deaths'].items():
        pct = (count / summary['games']) * 100
        print(f"  {death_type}: {count} ({pct:.1f}%)")
    
    print(f"\n{player1_name} Direction Preference:")
    for direction, pct in summary['p1_direction_preference'].items():
        print(f"  {direction}: {pct*100:.1f}%")
    
    print(f"\n{player2_name} Direction Preference:")
    for direction, pct in summary['p2_direction_preference'].items():
        print(f"  {direction}: {pct*100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tron AI Tournament System")
    parser.add_argument('--games', type=int, default=100, help='Games per matchup')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 games)')
    parser.add_argument('--analysis', nargs=2, metavar=('PLAYER1', 'PLAYER2'),
                       help='Detailed analysis of specific matchup')
    
    args = parser.parse_args()
    
    games_per_matchup = 10 if args.quick else args.games
    
    print("Loading player variants...")
    variants = create_player_variants()
    print(f"✓ Loaded {len(variants)} player variants\n")
    
    for i, variant in enumerate(variants, 1):
        print(f"{i}. {variant.name}")
        print(f"   {variant.description}")
    
    if args.analysis:
        # Load previous tournament results
        import glob
        result_files = sorted(glob.glob("tournament_results_*.json"))
        if result_files:
            print(f"\nLoading results from {result_files[-1]}...")
            with open(result_files[-1], 'r') as f:
                data = json.load(f)
            
            # Reconstruct tournament stats
            stats = TournamentStats()
            # This is simplified - in production you'd want to store/load properly
            print(f"Analysis requires running a fresh tournament first")
        else:
            print("No tournament results found. Run a tournament first.")
    else:
        # Run tournament
        tournament_stats = run_round_robin_tournament(variants, games_per_matchup)
        
        # Print leaderboard
        leaderboard = print_leaderboard(tournament_stats, variants)
        
        # Print some interesting matchups
        if len(variants) >= 2:
            print("\n" + "=" * 70)
            print("KEY MATCHUP ANALYSES")
            print("=" * 70)
            
            # Top 2 players
            if len(leaderboard) >= 2:
                print_detailed_analysis(tournament_stats, 
                                       leaderboard[0][0], 
                                       leaderboard[1][0])
