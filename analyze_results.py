import json
from collections import defaultdict

# Load results
with open('tournament_results_1778459597.json', 'r') as f:
    data = json.load(f)

# Calculate overall stats
player_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'games': 0})

for matchup, stats in data.items():
    p1, p2 = matchup.split('_vs_')
    
    player_stats[p1]['games'] += stats['games']
    player_stats[p1]['wins'] += stats['p1_wins']
    player_stats[p1]['losses'] += stats['p2_wins']
    player_stats[p1]['ties'] += stats['ties']
    
    player_stats[p2]['games'] += stats['games']
    player_stats[p2]['wins'] += stats['p2_wins']
    player_stats[p2]['losses'] += stats['p1_wins']
    player_stats[p2]['ties'] += stats['ties']

# Calculate win rates
results = []
for player, stats in player_stats.items():
    win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
    results.append((player, win_rate, stats['wins'], stats['losses'], stats['ties']))

# Sort by win rate
results.sort(key=lambda x: x[1], reverse=True)

# Print leaderboard
print('='*80)
print('FINAL TOURNAMENT LEADERBOARD')
print('='*80)
print(f"{'Rank':<6} {'Player':<28} {'Win Rate':<12} {'Record (W-L-T)':<20}")
print('-'*80)

for rank, (player, wr, w, l, t) in enumerate(results, 1):
    medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
    print(f"{medal} {rank:<3} {player:<28} {wr*100:>6.1f}%      {w:>3}-{l:>3}-{t:>2}")

print()
print('='*80)
print('KEY FINDINGS')
print('='*80)

# Find interesting patterns
print("\nTop 3 Performers:")
for i, (player, wr, w, l, t) in enumerate(results[:3], 1):
    print(f"  {i}. {player}: {wr*100:.1f}% win rate")

print("\nBottom 3 Performers:")
for i, (player, wr, w, l, t) in enumerate(results[-3:], 1):
    print(f"  {i}. {player}: {wr*100:.1f}% win rate")
