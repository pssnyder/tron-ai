"""
Tron Expert System Agent
A rule-based AI using classic pathfinding and space evaluation heuristics.
"""

import numpy as np
from collections import deque
from tron_game import Direction, TronGame


class ExpertAgent:
    """
    Expert system agent using static evaluation and heuristics.
    
    Strategy:
    - Avoid collisions with walls and trails
    - Maximize available space (flood fill)
    - Prefer directions with more open space
    - Stay away from opponent when possible
    - Use lookahead to avoid traps
    """
    
    def __init__(self, lookahead: int = 3, aggression: float = 0.3):
        """
        Initialize the expert agent.
        
        Args:
            lookahead: How many moves to look ahead
            aggression: How aggressive to be (0=defensive, 1=aggressive)
                       Higher values favor cutting off opponent
        """
        self.lookahead = lookahead
        self.aggression = aggression
        self.name = "Expert"
    
    def get_action(self, obs, player_num):
        """
        Get the best action based on expert heuristics.
        
        Args:
            obs: Game observation dictionary
            player_num: Player number (1 or 2)
            
        Returns:
            Direction to move
        """
        grid = obs['grid']
        my_pos = obs['p1_pos'] if player_num == 1 else obs['p2_pos']
        my_dir = obs['p1_dir'] if player_num == 1 else obs['p2_dir']
        opp_pos = obs['p2_pos'] if player_num == 1 else obs['p1_pos']
        
        # Get valid directions (can't turn 180 degrees)
        valid_dirs = self._get_valid_directions(my_dir)
        
        # Evaluate each direction
        scores = {}
        for direction in valid_dirs:
            score = self._evaluate_direction(grid, my_pos, opp_pos, direction, player_num)
            scores[direction] = score
        
        # Choose best direction
        if scores:
            best_dir = max(scores, key=scores.get)
            return best_dir
        else:
            # No good moves, just go straight
            return my_dir
    
    def _get_valid_directions(self, current_dir):
        """Get valid directions (excluding 180-degree turn)."""
        all_dirs = list(Direction)
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return [d for d in all_dirs if d != opposite[current_dir]]
    
    def _get_next_pos(self, pos, direction):
        """Calculate next position given current position and direction."""
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
        """Check if a position is safe (empty and in bounds)."""
        row, col = pos
        height, width = grid.shape
        
        # Check bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        # Check if empty
        return grid[row, col] == 0
    
    def _flood_fill(self, grid, start_pos, max_depth=None):
        """
        Count reachable empty cells using flood fill.
        
        Args:
            grid: Game grid
            start_pos: Starting position
            max_depth: Maximum depth to search (None for unlimited)
            
        Returns:
            Number of reachable cells
        """
        if not self._is_safe(grid, start_pos):
            return 0
        
        height, width = grid.shape
        visited = set()
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited.add((start_pos[0], start_pos[1]))
        count = 0
        
        while queue:
            row, col, depth = queue.popleft()
            count += 1
            
            # Stop at max depth
            if max_depth is not None and depth >= max_depth:
                continue
            
            # Check all 4 directions
            for direction in Direction:
                next_pos = self._get_next_pos([row, col], direction)
                next_tuple = (next_pos[0], next_pos[1])
                
                if next_tuple not in visited and self._is_safe(grid, next_pos):
                    visited.add(next_tuple)
                    queue.append((next_pos[0], next_pos[1], depth + 1))
        
        return count
    
    def _evaluate_direction(self, grid, my_pos, opp_pos, direction, player_num):
        """
        Evaluate how good a direction is.
        
        Heuristics:
        1. Immediate safety (don't crash)
        2. Available space (flood fill)
        3. Distance from opponent
        4. Lookahead safety
        
        Args:
            grid: Game grid
            my_pos: Current position
            opp_pos: Opponent position
            direction: Direction to evaluate
            player_num: Player number
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Calculate next position
        next_pos = self._get_next_pos(my_pos, direction)
        
        # 1. Immediate safety check
        if not self._is_safe(grid, next_pos):
            return -1000  # Instant death
        
        # Create a copy of the grid with our move applied
        test_grid = grid.copy()
        test_grid[next_pos[0], next_pos[1]] = player_num
        
        # 2. Available space (most important metric)
        available_space = self._flood_fill(test_grid, next_pos)
        score += available_space * 10
        
        # 3. Short-term lookahead space
        short_space = self._flood_fill(test_grid, next_pos, max_depth=5)
        score += short_space * 5
        
        # 4. Distance from opponent
        my_row, my_col = next_pos
        opp_row, opp_col = opp_pos
        distance = abs(my_row - opp_row) + abs(my_col - opp_col)
        
        # Aggressive: get closer to opponent
        # Defensive: stay away from opponent
        if self.aggression > 0.5:
            # Aggressive: prefer being closer (cut them off)
            score -= distance * self.aggression * 2
        else:
            # Defensive: prefer more distance
            score += distance * (1 - self.aggression) * 2
        
        # 5. Center preference (early game)
        height, width = grid.shape
        center_row, center_col = height // 2, width // 2
        dist_to_center = abs(my_row - center_row) + abs(my_col - center_col)
        
        # Prefer center early in game
        empty_cells = np.sum(grid == 0)
        total_cells = height * width
        if empty_cells > total_cells * 0.7:  # Early game
            score -= dist_to_center * 0.5
        
        # 6. Lookahead: check if we have multiple escape routes
        escape_routes = 0
        for next_dir in Direction:
            future_pos = self._get_next_pos(next_pos, next_dir)
            if self._is_safe(test_grid, future_pos):
                escape_routes += 1
        
        score += escape_routes * 3
        
        # 7. Wall proximity penalty
        wall_distance = min(
            next_pos[0],  # Distance to top
            height - 1 - next_pos[0],  # Distance to bottom
            next_pos[1],  # Distance to left
            width - 1 - next_pos[1]  # Distance to right
        )
        score += wall_distance * 0.5
        
        return score
    
    def __str__(self):
        return f"ExpertAgent(lookahead={self.lookahead}, aggression={self.aggression})"


class AdvancedExpertAgent(ExpertAgent):
    """
    Advanced expert agent with additional strategic heuristics.
    
    Enhancements:
    - Territory control evaluation
    - Cutting off opponent paths
    - Voronoi-like space partitioning
    """
    
    def __init__(self, lookahead: int = 5, aggression: float = 0.5):
        super().__init__(lookahead, aggression)
        self.name = "AdvancedExpert"
    
    def _evaluate_direction(self, grid, my_pos, opp_pos, direction, player_num):
        """Enhanced evaluation with territory control."""
        # Get base score from parent
        score = super()._evaluate_direction(grid, my_pos, opp_pos, direction, player_num)
        
        if score <= -1000:  # Immediate death
            return score
        
        # Calculate next position
        next_pos = self._get_next_pos(my_pos, direction)
        
        # Territory control: compare my reachable space vs opponent's
        test_grid = grid.copy()
        test_grid[next_pos[0], next_pos[1]] = player_num
        
        my_space = self._flood_fill(test_grid, next_pos)
        opp_space = self._flood_fill(test_grid, opp_pos)
        
        # Prefer moves that give us more space than opponent
        space_advantage = my_space - opp_space
        score += space_advantage * 5
        
        # If we have significantly more space, we can be more aggressive
        if my_space > opp_space * 1.5:
            # Cut off opponent
            opp_next_positions = []
            for opp_dir in Direction:
                opp_next = self._get_next_pos(opp_pos, opp_dir)
                if self._is_safe(test_grid, opp_next):
                    opp_next_positions.append(opp_next)
            
            # Prefer moves that reduce opponent's options
            score -= len(opp_next_positions) * 3
        
        return score


class SimpleSnakeAgent:
    """
    Very simple snake-like agent that just tries to survive.
    Useful as a baseline.
    """
    
    def __init__(self):
        self.name = "SimpleSnake"
    
    def get_action(self, obs, player_num):
        """Choose the safest direction with most open space ahead."""
        grid = obs['grid']
        my_pos = obs['p1_pos'] if player_num == 1 else obs['p2_pos']
        my_dir = obs['p1_dir'] if player_num == 1 else obs['p2_dir']
        
        # Try to continue straight if possible
        straight_pos = self._get_next_pos(my_pos, my_dir)
        if self._is_safe(grid, straight_pos):
            # Look ahead a few steps
            safe_count = 0
            test_pos = straight_pos
            test_dir = my_dir
            for _ in range(3):
                test_pos = self._get_next_pos(test_pos, test_dir)
                if self._is_safe(grid, test_pos):
                    safe_count += 1
                else:
                    break
            
            if safe_count >= 2:  # Good path ahead
                return my_dir
        
        # Otherwise, find a safe turn
        valid_dirs = self._get_valid_directions(my_dir)
        
        for direction in valid_dirs:
            next_pos = self._get_next_pos(my_pos, direction)
            if self._is_safe(grid, next_pos):
                return direction
        
        # No good option, just go straight
        return my_dir
    
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
    
    def _get_valid_directions(self, current_dir):
        """Get valid directions (excluding 180-degree turn)."""
        all_dirs = list(Direction)
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return [d for d in all_dirs if d != opposite[current_dir]]


if __name__ == "__main__":
    from tron_game import TronGame, run_tournament
    
    print("Expert Agent Demo")
    print("="*50)
    
    # Create different expert agents
    defensive_agent = ExpertAgent(lookahead=3, aggression=0.2)
    balanced_agent = ExpertAgent(lookahead=3, aggression=0.5)
    aggressive_agent = ExpertAgent(lookahead=3, aggression=0.8)
    advanced_agent = AdvancedExpertAgent(lookahead=5, aggression=0.5)
    simple_agent = SimpleSnakeAgent()
    
    # Test 1: Defensive vs Aggressive
    print("\n1. Defensive Expert vs Aggressive Expert")
    run_tournament(defensive_agent, aggressive_agent, num_games=100, headless=True, verbose=True)
    
    # Test 2: Balanced vs Advanced
    print("\n2. Balanced Expert vs Advanced Expert")
    run_tournament(balanced_agent, advanced_agent, num_games=100, headless=True, verbose=True)
    
    # Test 3: Advanced vs Simple Snake
    print("\n3. Advanced Expert vs Simple Snake")
    run_tournament(advanced_agent, simple_agent, num_games=100, headless=True, verbose=True)
    
    # Visual match
    print("\n4. Visual Match: Advanced vs Aggressive")
    game = TronGame(width=30, height=30, headless=False)
    stats = game.play_match(advanced_agent, aggressive_agent, fps=10, verbose=True)
    game.close()
