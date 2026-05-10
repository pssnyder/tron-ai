"""
Tron Game Engine
A complete game engine for Tron with AI agent support, headless/visual modes, and metrics collection.
"""

import numpy as np
from enum import IntEnum
from typing import Tuple, List, Optional, Dict
import time


class Direction(IntEnum):
    """Movement directions"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class TronGame:
    """
    Tron game engine with API for AI agents.
    
    Features:
    - Headless mode for fast training
    - Visual mode with pygame for watching matches
    - Complete game state observation
    - Collision detection
    - Metrics and statistics collection
    """
    
    def __init__(self, width: int = 40, height: int = 40, headless: bool = True):
        """
        Initialize the Tron game.
        
        Args:
            width: Grid width
            height: Grid height
            headless: If True, run without visualization
        """
        self.width = width
        self.height = height
        self.headless = headless
        
        # Game state
        self.grid = np.zeros((height, width), dtype=np.int8)  # 0=empty, 1=p1, 2=p2, -1=both crashed
        self.reset()
        
        # Visualization
        self.display = None
        self.clock = None
        if not headless:
            self._init_display()
    
    def reset(self) -> Dict:
        """
        Reset the game to initial state.
        
        Returns:
            Initial observation dictionary
        """
        self.grid.fill(0)
        
        # Player 1 starts on left side
        self.p1_pos = [self.height // 2, self.width // 4]
        self.p1_dir = Direction.RIGHT
        
        # Player 2 starts on right side
        self.p2_pos = [self.height // 2, 3 * self.width // 4]
        self.p2_dir = Direction.LEFT
        
        # Mark starting positions
        self.grid[self.p1_pos[0], self.p1_pos[1]] = 1
        self.grid[self.p2_pos[0], self.p2_pos[1]] = 2
        
        # Game state
        self.game_over = False
        self.winner = None
        self.turn = 0
        
        # Statistics - trail length (cells traversed)
        self.p1_trail_length = 1
        self.p2_trail_length = 1
        
        # Directional tracking
        self.p1_direction_counts = {Direction.UP: 0, Direction.RIGHT: 0, Direction.DOWN: 0, Direction.LEFT: 0}
        self.p2_direction_counts = {Direction.UP: 0, Direction.RIGHT: 0, Direction.DOWN: 0, Direction.LEFT: 0}
        self.p1_direction_counts[Direction.RIGHT] = 1
        self.p2_direction_counts[Direction.LEFT] = 1
        
        # Death tracking
        self.p1_death_type = None  # 'wall', 'own_trail', 'opponent_trail', 'head_collision'
        self.p2_death_type = None
        
        return self.get_observation()
    
    def get_observation(self) -> Dict:
        """
        Get current game state for AI agents.
        
        Returns:
            Dictionary containing:
            - grid: numpy array of the game board
            - p1_pos: [row, col] position of player 1
            - p2_pos: [row, col] position of player 2
            - p1_dir: current direction of player 1
            - p2_dir: current direction of player 2
            - turn: current turn number
            - p1_territory: cells controlled by player 1
            - p2_territory: cells controlled by player 2
        """
        return {
            'grid': self.grid.copy(),
            'p1_pos': self.p1_pos.copy(),
            'p2_pos': self.p2_pos.copy(),
            'p1_dir': self.p1_dir,
            'p2_dir': self.p2_dir,
            'turn': self.turn,
            'p1_trail_length': self.p1_trail_length,
            'p2_trail_length': self.p2_trail_length,
        }
    
    def is_valid_move(self, pos: List[int], direction: Direction, player: int) -> bool:
        """
        Check if a move is valid (not turning 180 degrees).
        
        Args:
            pos: Current position
            direction: Proposed direction
            player: Player number (1 or 2)
            
        Returns:
            True if the move is valid
        """
        current_dir = self.p1_dir if player == 1 else self.p2_dir
        # Can't turn 180 degrees
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return direction != opposite[current_dir]
    
    def get_next_pos(self, pos: List[int], direction: Direction) -> List[int]:
        """Get the next position given current position and direction."""
        row, col = pos
        if direction == Direction.UP:
            return [row - 1, col]
        elif direction == Direction.DOWN:
            return [row + 1, col]
        elif direction == Direction.LEFT:
            return [row, col - 1]
        elif direction == Direction.RIGHT:
            return [row, col + 1]
    
    def is_collision(self, pos: List[int]) -> bool:
        """Check if position is a collision (wall or trail)."""
        row, col = pos
        # Wall collision
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        # Trail collision
        if self.grid[row, col] != 0:
            return True
        return False
    
    def _calculate_reachable_territory(self, start_pos: List[int]) -> int:
        """
        Calculate reachable empty cells from a position using flood fill.
        
        Args:
            start_pos: Starting position [row, col]
            
        Returns:
            Number of reachable empty cells
        """
        from collections import deque
        
        visited = set()
        queue = deque([tuple(start_pos)])
        visited.add(tuple(start_pos))
        count = 0
        
        while queue:
            row, col = queue.popleft()
            
            # Check if this cell is accessible (empty or our current position)
            if self.grid[row, col] == 0 or [row, col] == start_pos:
                count += 1
            
            # Check all 4 directions
            for direction in Direction:
                next_pos = self._get_next_pos([row, col], direction)
                next_tuple = tuple(next_pos)
                
                if next_tuple not in visited:
                    # Check if in bounds
                    if (0 <= next_pos[0] < self.height and 
                        0 <= next_pos[1] < self.width):
                        # Check if accessible (empty space)
                        if self.grid[next_pos[0], next_pos[1]] == 0:
                            visited.add(next_tuple)
                            queue.append(next_tuple)
        
        return count
    
    def _get_next_pos(self, pos: List[int], direction: Direction) -> List[int]:
        """Helper method for internal use (handles list or tuple positions)."""
        row, col = pos[0], pos[1]
        if direction == Direction.UP:
            return [row - 1, col]
        elif direction == Direction.DOWN:
            return [row + 1, col]
        elif direction == Direction.LEFT:
            return [row, col - 1]
        elif direction == Direction.RIGHT:
            return [row, col + 1]
    
    def _determine_death_type(self, pos: List[int], player: int) -> str:
        """
        Determine how a player died.
        
        Args:
            pos: Position where collision occurred
            player: Player number (1 or 2)
            
        Returns:
            Death type: 'wall', 'own_trail', 'opponent_trail', 'head_collision'
        """
        row, col = pos
        
        # Check if hit wall
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return 'wall'
        
        # Check what's at the collision point
        cell_value = self.grid[row, col]
        
        if cell_value == player:
            return 'own_trail'
        elif cell_value != 0:
            return 'opponent_trail'
        
        return 'unknown'
    
    def step(self, p1_action: Direction, p2_action: Direction) -> Tuple[Dict, float, float, bool]:
        """
        Execute one game step with both players' actions.
        
        Args:
            p1_action: Direction for player 1
            p2_action: Direction for player 2
            
        Returns:
            Tuple of (observation, p1_reward, p2_reward, done)
        """
        if self.game_over:
            obs = self.get_observation()
            return obs, 0, 0, True
        
        # Validate moves (prevent 180-degree turns)
        if not self.is_valid_move(self.p1_pos, p1_action, 1):
            p1_action = self.p1_dir
        if not self.is_valid_move(self.p2_pos, p2_action, 2):
            p2_action = self.p2_dir
        
        # Update directions and track them
        self.p1_dir = p1_action
        self.p2_dir = p2_action
        self.p1_direction_counts[p1_action] += 1
        self.p2_direction_counts[p2_action] += 1
        
        # Calculate next positions
        p1_next = self.get_next_pos(self.p1_pos, p1_action)
        p2_next = self.get_next_pos(self.p2_pos, p2_action)
        
        # Check collisions
        p1_crashed = self.is_collision(p1_next)
        p2_crashed = self.is_collision(p2_next)
        
        # Head-to-head collision
        head_collision = False
        if p1_next == p2_next:
            p1_crashed = True
            p2_crashed = True
            head_collision = True
        
        # Determine outcome and rewards
        p1_reward = 0
        p2_reward = 0
        
        if p1_crashed and p2_crashed:
            self.game_over = True
            self.winner = 0  # Tie
            p1_reward = -1
            p2_reward = -1
            
            # Record death types
            if head_collision:
                self.p1_death_type = 'head_collision'
                self.p2_death_type = 'head_collision'
            else:
                self.p1_death_type = self._determine_death_type(p1_next, 1)
                self.p2_death_type = self._determine_death_type(p2_next, 2)
                
        elif p1_crashed:
            self.game_over = True
            self.winner = 2
            p1_reward = -10
            p2_reward = 10
            self.p1_death_type = self._determine_death_type(p1_next, 1)
            
        elif p2_crashed:
            self.game_over = True
            self.winner = 1
            p1_reward = 10
            p2_reward = -10
            self.p2_death_type = self._determine_death_type(p2_next, 2)
            
        else:
            # Both alive - small step reward
            p1_reward = 0.1
            p2_reward = 0.1
            
            # Update positions and grid
            self.p1_pos = p1_next
            self.p2_pos = p2_next
            
            self.grid[p1_next[0], p1_next[1]] = 1
            self.grid[p2_next[0], p2_next[1]] = 2
            
            self.p1_trail_length += 1
            self.p2_trail_length += 1
        
        self.turn += 1
        
        return self.get_observation(), p1_reward, p2_reward, self.game_over
    
    def _init_display(self):
        """Initialize pygame display for visualization."""
        try:
            import pygame
            pygame.init()
            
            self.cell_size = 15
            self.screen_width = self.width * self.cell_size
            self.screen_height = self.height * self.cell_size + 60  # Extra space for stats
            
            self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Tron AI Battle')
            self.clock = pygame.time.Clock()
            
            # Colors
            self.COLOR_BG = (0, 0, 0)
            self.COLOR_P1 = (255, 50, 50)  # Red
            self.COLOR_P2 = (50, 150, 255)  # Blue
            self.COLOR_GRID = (30, 30, 30)
            self.COLOR_TEXT = (200, 200, 200)
            
        except ImportError:
            print("Warning: pygame not installed. Running in headless mode.")
            self.headless = True
    
    def render(self, fps: int = 10):
        """
        Render the current game state.
        
        Args:
            fps: Frames per second for visualization
        """
        if self.headless or self.display is None:
            return
        
        import pygame
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Clear screen
        self.display.fill(self.COLOR_BG)
        
        # Draw grid
        for row in range(self.height):
            for col in range(self.width):
                x = col * self.cell_size
                y = row * self.cell_size
                
                # Draw cell
                if self.grid[row, col] == 1:
                    pygame.draw.rect(self.display, self.COLOR_P1, 
                                   (x, y, self.cell_size, self.cell_size))
                elif self.grid[row, col] == 2:
                    pygame.draw.rect(self.display, self.COLOR_P2,
                                   (x, y, self.cell_size, self.cell_size))
                
                # Draw grid lines
                pygame.draw.rect(self.display, self.COLOR_GRID,
                               (x, y, self.cell_size, self.cell_size), 1)
        
        # Draw stats
        font = pygame.font.Font(None, 24)
        stats_y = self.height * self.cell_size + 10
        
        # Player 1 stats
        p1_text = f"P1 (Red): {self.p1_trail_length} cells"
        text_surface = font.render(p1_text, True, self.COLOR_P1)
        self.display.blit(text_surface, (10, stats_y))
        
        # Player 2 stats
        p2_text = f"P2 (Blue): {self.p2_trail_length} cells"
        text_surface = font.render(p2_text, True, self.COLOR_P2)
        self.display.blit(text_surface, (10, stats_y + 25))
        
        # Turn counter
        turn_text = f"Turn: {self.turn}"
        text_surface = font.render(turn_text, True, self.COLOR_TEXT)
        self.display.blit(text_surface, (self.width * self.cell_size - 120, stats_y))
        
        # Game over message
        if self.game_over:
            big_font = pygame.font.Font(None, 48)
            if self.winner == 1:
                msg = "Player 1 (Red) Wins!"
                color = self.COLOR_P1
            elif self.winner == 2:
                msg = "Player 2 (Blue) Wins!"
                color = self.COLOR_P2
            else:
                msg = "Tie Game!"
                color = self.COLOR_TEXT
            
            text_surface = big_font.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            
            # Draw semi-transparent background
            s = pygame.Surface((self.screen_width, self.screen_height))
            s.set_alpha(128)
            s.fill((0, 0, 0))
            self.display.blit(s, (0, 0))
            
            # Draw text
            self.display.blit(text_surface, text_rect)
        
        pygame.display.flip()
        self.clock.tick(fps)
    
    def play_match(self, agent1, agent2, max_turns: int = 1000, fps: int = 10, verbose: bool = False):
        """
        Play a complete match between two agents.
        
        Args:
            agent1: First agent (must have get_action(obs, player_num) method)
            agent2: Second agent
            max_turns: Maximum number of turns before declaring a tie
            fps: Frames per second for visualization
            verbose: Print game progress
            
        Returns:
            Dictionary with match statistics
        """
        obs = self.reset()
        done = False
        
        while not done and self.turn < max_turns:
            # Get actions from both agents
            p1_action = agent1.get_action(obs, player_num=1)
            p2_action = agent2.get_action(obs, player_num=2)
            
            # Execute step
            obs, p1_reward, p2_reward, done = self.step(p1_action, p2_action)
            
            # Render if visual mode
            if not self.headless:
                self.render(fps)
            
            if verbose and self.turn % 100 == 0:
                print(f"Turn {self.turn}: P1={self.p1_trail_length} cells, P2={self.p2_trail_length} cells")
        
        # Max turns reached
        if not done:
            # Determine winner by controlled territory (reachable space)
            p1_territory = self._calculate_reachable_territory(self.p1_pos)
            p2_territory = self._calculate_reachable_territory(self.p2_pos)
            
            if p1_territory > p2_territory:
                self.winner = 1
            elif p2_territory > p1_territory:
                self.winner = 2
            else:
                self.winner = 0
            self.game_over = True
        
        # Calculate final controlled territories
        p1_controlled_territory = self._calculate_reachable_territory(self.p1_pos) if self.winner != 2 else 0
        p2_controlled_territory = self._calculate_reachable_territory(self.p2_pos) if self.winner != 1 else 0
        
        # Calculate directional percentages
        p1_total_moves = sum(self.p1_direction_counts.values())
        p2_total_moves = sum(self.p2_direction_counts.values())
        
        p1_direction_pct = {k: v / p1_total_moves * 100 if p1_total_moves > 0 else 0 
                           for k, v in self.p1_direction_counts.items()}
        p2_direction_pct = {k: v / p2_total_moves * 100 if p2_total_moves > 0 else 0 
                           for k, v in self.p2_direction_counts.items()}
        
        # Compile statistics
        stats = {
            'winner': self.winner,
            'turns': self.turn,
            'p1_trail_length': self.p1_trail_length,
            'p2_trail_length': self.p2_trail_length,
            'p1_controlled_territory': p1_controlled_territory,
            'p2_controlled_territory': p2_controlled_territory,
            'p1_death_type': self.p1_death_type,
            'p2_death_type': self.p2_death_type,
            'p1_direction_distribution': p1_direction_pct,
            'p2_direction_distribution': p2_direction_pct,
            'p1_survival_rate': self.p1_trail_length / max(self.turn, 1),
            'p2_survival_rate': self.p2_trail_length / max(self.turn, 1),
        }
        
        if verbose:
            print("\n=== Match Results ===")
            if self.winner == 1:
                print("Winner: Player 1 (Red)")
            elif self.winner == 2:
                print("Winner: Player 2 (Blue)")
            else:
                print("Winner: Tie")
            print(f"Turns: {stats['turns']}")
            print(f"\nTrail Lengths:")
            print(f"  P1: {stats['p1_trail_length']} cells")
            print(f"  P2: {stats['p2_trail_length']} cells")
            print(f"\nControlled Territory:")
            print(f"  P1: {stats['p1_controlled_territory']} cells")
            print(f"  P2: {stats['p2_controlled_territory']} cells")
            if stats['p1_death_type']:
                print(f"\nP1 Death: {stats['p1_death_type']}")
            if stats['p2_death_type']:
                print(f"P2 Death: {stats['p2_death_type']}")
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if not self.headless and self.display is not None:
            import pygame
            pygame.quit()


def run_tournament(agent1, agent2, num_games: int = 100, headless: bool = True, verbose: bool = False):
    """
    Run a tournament between two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        headless: Run without visualization
        verbose: Print detailed statistics
        
    Returns:
        Dictionary with tournament statistics
    """
    game = TronGame(headless=headless)
    
    wins = {1: 0, 2: 0, 0: 0}  # 0 is tie
    total_turns = 0
    total_p1_controlled = 0
    total_p2_controlled = 0
    death_types_p1 = {'wall': 0, 'own_trail': 0, 'opponent_trail': 0, 'head_collision': 0, None: 0}
    death_types_p2 = {'wall': 0, 'own_trail': 0, 'opponent_trail': 0, 'head_collision': 0, None: 0}
    
    print(f"\nRunning tournament: {num_games} games")
    start_time = time.time()
    
    for i in range(num_games):
        stats = game.play_match(agent1, agent2, verbose=False)
        
        wins[stats['winner']] += 1
        total_turns += stats['turns']
        total_p1_controlled += stats['p1_controlled_territory']
        total_p2_controlled += stats['p2_controlled_territory']
        death_types_p1[stats['p1_death_type']] += 1
        death_types_p2[stats['p2_death_type']] += 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_games} games")
    
    elapsed = time.time() - start_time
    
    # Compile results
    results = {
        'agent1_wins': wins[1],
        'agent2_wins': wins[2],
        'ties': wins[0],
        'total_games': num_games,
        'agent1_win_rate': wins[1] / num_games,
        'agent2_win_rate': wins[2] / num_games,
        'avg_turns': total_turns / num_games,
        'avg_p1_controlled_territory': total_p1_controlled / num_games,
        'avg_p2_controlled_territory': total_p2_controlled / num_games,
        'p1_death_distribution': {k: v/num_games*100 for k, v in death_types_p1.items() if k is not None},
        'p2_death_distribution': {k: v/num_games*100 for k, v in death_types_p2.items() if k is not None},
        'elapsed_time': elapsed,
        'games_per_second': num_games / elapsed,
    }
    
    print("\n=== Tournament Results ===")
    print(f"Total Games: {num_games}")
    print(f"Agent 1 Wins: {wins[1]} ({results['agent1_win_rate']:.1%})")
    print(f"Agent 2 Wins: {wins[2]} ({results['agent2_win_rate']:.1%})")
    print(f"Ties: {wins[0]} ({wins[0]/num_games:.1%})")
    print(f"Average Turns: {results['avg_turns']:.1f}")
    print(f"Average Controlled Territory - P1: {results['avg_p1_controlled_territory']:.1f}, P2: {results['avg_p2_controlled_territory']:.1f}")
    print(f"\nDeath Analysis:")
    print(f"  Agent 1 Deaths: Wall={death_types_p1['wall']}, Own Trail={death_types_p1['own_trail']}, Opponent Trail={death_types_p1['opponent_trail']}, Head-on={death_types_p1['head_collision']}")
    print(f"  Agent 2 Deaths: Wall={death_types_p2['wall']}, Own Trail={death_types_p2['own_trail']}, Opponent Trail={death_types_p2['opponent_trail']}, Head-on={death_types_p2['head_collision']}")
    print(f"Elapsed Time: {elapsed:.2f}s ({results['games_per_second']:.1f} games/sec)")
    
    game.close()
    return results


if __name__ == "__main__":
    # Demo with random agents
    class RandomAgent:
        """Simple random agent for testing."""
        def get_action(self, obs, player_num):
            import random
            return random.choice(list(Direction))
    
    print("Tron Game Engine - Demo Mode")
    print("Playing a match between two random agents...")
    
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    
    # Single visual match
    game = TronGame(width=30, height=30, headless=False)
    stats = game.play_match(agent1, agent2, fps=15, verbose=True)
    game.close()
    
    # Tournament
    print("\n" + "="*50)
    print("Running headless tournament...")
    run_tournament(agent1, agent2, num_games=50, headless=True, verbose=True)
