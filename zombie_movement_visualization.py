#!/usr/bin/env python3
"""
Zombie Movement Visualization Script

This script demonstrates how to visualize zombie movement predictions from our laws
vs actual environment behavior. It creates a scenario with a zombie near the player,
samples multiple transitions, and overlays probability distributions on the rendered game.

COORDINATE MAPPING SYSTEM:
==========================

The Crafter environment uses multiple coordinate systems that need to be mapped:

1. WORLD COORDINATES:
   - Absolute positions in the game world (e.g., player at (16, 16))
   - Used by entities, laws, and state representations
   - Can be any integer values

2. VIEW COORDINATES:
   - Local coordinates within the player's view window
   - View is typically 9x9 tiles centered on the player
   - Player is at view center (4, 4) in a 9x9 view
   - Range: [0, view_width) x [0, view_height)

3. PIXEL COORDINATES:
   - Actual pixel positions in the rendered image
   - Image size: typically 256x256 pixels
   - Each view tile maps to ~28x28 pixels (256/9 ≈ 28)

COORDINATE TRANSFORMATION PIPELINE:
===================================

World → View → Pixel

1. World to View:
   - Calculate relative position from player: (world_x - player_x, world_y - player_y)
   - Add view center offset: (view_center_x + rel_x, view_center_y + rel_y)
   - Check bounds: position must be within [0, view_width) x [0, view_height)

2. View to Pixel:
   - Calculate unit size: pixel_size = render_size / view_size
   - Map to pixels: (view_x * unit_x, view_y * unit_y)
   - Draw square: [pixel_x : pixel_x + unit_x, pixel_y : pixel_y + unit_y]

EXAMPLE:
========
Player at world (16, 16), zombie at world (17, 16):
- Relative position: (17-16, 16-16) = (1, 0)
- View coordinates: (4+1, 4+0) = (5, 4) in 9x9 view
- Pixel coordinates: (5*28, 4*28) = (140, 112) in 256x256 image
- Draw 28x28 pixel square at (140, 112)
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

# Import crafter components
from crafter.functional_env import (
    initial_state,
    transition,
    observation,
    export_world_state,
)
from crafter.state_export import WorldState, ZombieState
from crafter.state_reconstruction import reconstruct_world_from_state
from crafter import objects, constants
from crafter.testing_helpers import world_utils, player_utils

# Import our method components
from distant_sunburn.our_method.crafter.handwritten_laws import (
    CorrectEntityAILaw,
    LawFunctionWrapper,
)
from distant_sunburn.poe_world.core import DiscreteDistribution


# ============================================================================
# COORDINATE MAPPING HELPER FUNCTIONS
# ============================================================================


def world_to_view_coordinates(
    world_x: int,
    world_y: int,
    player_x: int,
    player_y: int,
    view_width: int,
    view_height: int,
) -> tuple[int, int] | None:
    """
    Convert world coordinates to view coordinates.

    Args:
        world_x, world_y: World position to convert
        player_x, player_y: Player's world position
        view_width, view_height: Dimensions of the view window

    Returns:
        (view_x, view_y) if position is within view bounds, None otherwise
    """
    # Calculate relative position from player
    rel_x = world_x - player_x
    rel_y = world_y - player_y

    # Convert to view coordinates (view is centered on player)
    view_center_x = view_width // 2
    view_center_y = view_height // 2
    view_x = view_center_x + rel_x
    view_y = view_center_y + rel_y

    # Check if position is within view bounds
    if 0 <= view_x < view_width and 0 <= view_y < view_height:
        return (view_x, view_y)
    else:
        return None


def view_to_pixel_coordinates(
    view_x: int,
    view_y: int,
    view_width: int,
    view_height: int,
    render_width: int,
    render_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert view coordinates to pixel coordinates and calculate square size.

    Args:
        view_x, view_y: View position to convert
        view_width, view_height: Dimensions of the view window
        render_width, render_height: Dimensions of the rendered image

    Returns:
        (pixel_x, pixel_y, square_width, square_height)
    """
    # Calculate unit size for each grid cell
    unit_x = render_width // view_width
    unit_y = render_height // view_height

    # Calculate pixel coordinates
    pixel_x = view_x * unit_x
    pixel_y = view_y * unit_y

    return (pixel_x, pixel_y, unit_x, unit_y)


def world_to_pixel_coordinates(
    world_x: int,
    world_y: int,
    player_x: int,
    player_y: int,
    view_width: int,
    view_height: int,
    render_width: int,
    render_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Convert world coordinates directly to pixel coordinates.

    Args:
        world_x, world_y: World position to convert
        player_x, player_y: Player's world position
        view_width, view_height: Dimensions of the view window
        render_width, render_height: Dimensions of the rendered image

    Returns:
        (pixel_x, pixel_y, square_width, square_height) if within view, None otherwise
    """
    # First convert to view coordinates
    view_coords = world_to_view_coordinates(
        world_x, world_y, player_x, player_y, view_width, view_height
    )

    if view_coords is None:
        return None

    view_x, view_y = view_coords

    # Then convert to pixel coordinates
    return view_to_pixel_coordinates(
        view_x, view_y, view_width, view_height, render_width, render_height
    )


# ============================================================================
# MAIN CLASSES
# ============================================================================


class ZombiePlacementHelper:
    """Helper class to place zombies near the player for testing."""

    @staticmethod
    def place_zombie_near_player(state: WorldState, distance: int = 3) -> WorldState:
        """
        Place a zombie near the player in the given state.

        Args:
            state: The world state to modify
            distance: Distance from player to place zombie

        Returns:
            Modified state with zombie placed near player
        """
        # Reconstruct world from state
        world = reconstruct_world_from_state(state)

        # Get player
        player = None
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                player = obj
                break

        if player is None:
            raise ValueError("No player found in world")

        # Clear all entities from the world (except the player)
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                continue
            world.remove(obj)

        # Set all tiles to grass for walkability
        for x in range(state.size[0]):
            for y in range(state.size[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        # Place player at center
        player_utils.set_player_position(
            player, (state.size[0] // 2, state.size[1] // 2)
        )

        # Add a zombie near the player
        zombie_pos = (player.pos[0] + distance, player.pos[1])
        zombie = objects.Zombie(world, zombie_pos, player)
        world.add(zombie)

        print(f"Placed zombie at {zombie_pos} near player at {player.pos}")

        # Export the modified state
        return export_world_state(world, view=state.view, step_count=state.step_count)


class DistributionVisualizer:
    """Helper class to visualize probability distributions on the game render."""

    @staticmethod
    def create_distribution_overlay(
        base_image: np.ndarray,
        distributions: dict[tuple[int, int], float],
        view_dims: tuple[int, int],
        render_size: tuple[int, int],
        player_pos: tuple[int, int],
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Create an overlay showing probability distributions as colored squares.

        Args:
            base_image: The base rendered game image
            distributions: Dict mapping (x, y) world positions to probability values
            view_dims: View dimensions of the game
            render_size: Render size of the image
            player_pos: Player's world position (x, y)
            alpha: Transparency of the overlay

        Returns:
            Image with distribution overlay
        """
        # Create a copy of the base image
        overlay_image = base_image.copy().astype(np.float32)

        view_width, view_height = view_dims
        render_width, render_height = render_size
        player_x, player_y = player_pos

        # Create overlay for each distribution position
        for (world_x, world_y), prob in distributions.items():
            # Use helper function to convert world coordinates to pixel coordinates
            pixel_coords = world_to_pixel_coordinates(
                world_x,
                world_y,
                player_x,
                player_y,
                view_width,
                view_height,
                render_width,
                render_height,
            )

            if pixel_coords is not None:
                pixel_x, pixel_y, square_width, square_height = pixel_coords

                # Create colored square based on probability
                color = DistributionVisualizer._prob_to_color(prob)

                # Draw square
                for dx in range(square_width):
                    for dy in range(square_height):
                        px, py = pixel_x + dx, pixel_y + dy
                        if (
                            0 <= px < overlay_image.shape[1]
                            and 0 <= py < overlay_image.shape[0]
                        ):
                            # Blend with existing pixel
                            overlay_image[py, px] = (
                                alpha * np.array(color)
                                + (1 - alpha) * overlay_image[py, px]
                            )

        return overlay_image.astype(np.uint8)

    @staticmethod
    def _prob_to_color(prob: float) -> tuple[int, int, int]:
        """Convert probability to RGB color (red for high, blue for low)."""
        # Clamp probability to [0, 1]
        prob = max(0, min(1, prob))

        # Red for high probability, blue for low probability
        red = int(255 * prob)
        blue = int(255 * (1 - prob))
        green = 0

        return (red, green, blue)


class ZombieMovementAnalyzer:
    """Main class for analyzing zombie movement predictions vs reality."""

    def __init__(self):
        self.law = LawFunctionWrapper.from_non_runtime_created(CorrectEntityAILaw())

    def sample_environment_transitions(
        self, initial_state: WorldState, action: str, n_samples: int = 50
    ) -> list[tuple[int, int]]:
        """
        Sample multiple transitions from the environment to get true zombie positions.

        Args:
            initial_state: Starting state
            action: Action to take
            n_samples: Number of samples to take

        Returns:
            List of (x, y) positions where zombie ended up
        """
        # Convert action string to action index
        action_idx = None
        for i, action_name in enumerate(constants.actions):
            if action_name == action:
                action_idx = i
                break

        if action_idx is None:
            raise ValueError(f"Unknown action: {action}")
        positions = []

        for i in range(n_samples):
            # Create a copy of the initial state for each sample
            state_copy = copy.deepcopy(initial_state)

            # Take the transition
            next_state, _ = transition(state_copy, action_idx)

            # Find zombie position in the next state
            for obj in next_state.objects:
                if isinstance(obj, ZombieState):
                    positions.append((obj.position.x, obj.position.y))
                    break

        return positions

    def get_law_predictions(
        self, initial_state: WorldState, action: str
    ) -> dict[tuple[int, int], float]:
        """
        Get zombie movement predictions from our law.

        Args:
            initial_state: Starting state
            action: Action to take

        Returns:
            Dict mapping (x, y) positions to predicted probabilities
        """
        # Create a copy of the state
        state_copy = copy.deepcopy(initial_state)

        # Apply the law
        if self.law.precondition(state_copy, action):
            self.law.effect(state_copy, action)

        # Find zombie and extract its position distribution
        for obj in state_copy.objects:
            if isinstance(obj, ZombieState):
                # Check if position has DiscreteDistribution
                if isinstance(obj.position.x, DiscreteDistribution) and isinstance(
                    obj.position.y, DiscreteDistribution
                ):
                    # Extract distribution
                    x_dist = obj.position.x
                    y_dist = obj.position.y

                    # Convert to probability dict
                    predictions = {}
                    for i, x_val in enumerate(x_dist.support):
                        for j, y_val in enumerate(y_dist.support):
                            # Calculate combined probability
                            x_prob = np.exp(x_dist.log_probs[i])
                            y_prob = np.exp(y_dist.log_probs[j])
                            combined_prob = x_prob * y_prob

                            predictions[(int(x_val), int(y_val))] = combined_prob

                    return predictions

        return {}

    def create_test_visualization(self, initial_state: WorldState) -> None:
        """
        Create a test visualization with hand-specified distributions to test the rendering code.

        Args:
            initial_state: Starting state with zombie placed near player
        """
        print("Creating test visualization with hand-specified distributions...")

        # Render the initial state
        base_image = observation(initial_state, render_size=(256, 256))
        print(f"Base image shape: {base_image.shape}")

        # Get player position for coordinate mapping
        player_pos = initial_state.player.position
        print(f"Player position: ({player_pos.x}, {player_pos.y})")

        # Create a simple test: draw a blue square at a fixed position
        test_overlay = base_image.copy().astype(np.float32)

        # Draw a bright blue square in the center of the image
        center_x, center_y = base_image.shape[1] // 2, base_image.shape[0] // 2
        square_size = 32

        for dx in range(square_size):
            for dy in range(square_size):
                x = center_x - square_size // 2 + dx
                y = center_y - square_size // 2 + dy
                if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
                    test_overlay[y, x] = [0, 0, 255]  # Bright blue

        # Test the distribution overlay function with debug info
        # Use world coordinates relative to the player position
        test_distribution = {
            (player_pos.x, player_pos.y): 1.0,  # Player position
            (player_pos.x + 1, player_pos.y): 0.5,  # One tile right of player
            (player_pos.x, player_pos.y + 1): 0.3,  # One tile down from player
            (player_pos.x - 1, player_pos.y): 0.2,  # One tile left of player
        }

        print(f"Testing distribution overlay with {len(test_distribution)} positions")

        # Test the distribution overlay
        dist_overlay = DistributionVisualizer.create_distribution_overlay(
            base_image,
            test_distribution,
            (9, 9),
            (256, 256),
            (player_pos.x, player_pos.y),
        )

        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Base image
        axes[0].imshow(base_image)
        axes[0].set_title("Base Game State")
        axes[0].axis("off")

        # Test overlay with blue square
        axes[1].imshow(test_overlay.astype(np.uint8))
        axes[1].set_title("Test: Blue Square Overlay")
        axes[1].axis("off")

        # Test distribution overlay
        axes[2].imshow(dist_overlay)
        axes[2].set_title("Test: Distribution Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("zombie_movement_test.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\nTest: Drew blue square at center ({center_x}, {center_y})")
        print(
            f"Test: Distribution overlay created with {len(test_distribution)} positions"
        )
        print("Test visualization complete! Check 'zombie_movement_test.png'")


def main():
    """Main function to run the zombie movement visualization."""
    print("Creating initial game state...")

    # Create initial state
    state = initial_state(area=(32, 32), view=(9, 9), seed=42)

    # Place a zombie near the player
    state_with_zombie = ZombiePlacementHelper.place_zombie_near_player(
        state, distance=3
    )

    # Create analyzer
    analyzer = ZombieMovementAnalyzer()

    # Create test visualization with hand-specified distributions
    analyzer.create_test_visualization(state_with_zombie)


if __name__ == "__main__":
    main()
