# PRD: Hybrid Evaluation Framework for Crafter (Revised Architecture)

## 1. Overview

This document outlines the engineering requirements for extending our hybrid evaluation framework to the Crafter environment. This revised version incorporates a cleaner, more functional architecture based on critical feedback, ensuring better code isolation and clarity for the implementer.

The primary goal is to create a robust, configurable, and maintainable testing suite for Crafter world models. This involves two key workstreams:
1.  **Trajectory Collection**: Implementing flexible strategies for collecting interesting and diverse state transitions from the environment.
2.  **Distractor Generation**: Creating a structured and categorized set of state mutations to test a world model's fine-grained understanding of game mechanics.

**Reference Document:** [Hybrid Evaluation Framework for Symbolic WMs](docs/1-hybrid-evaluation-framework-for-symbolic-wms.md)

## 2. Core Architecture

We will continue to use the hexagonal architecture defined in `src/distant_sunburn/evaluator/core.py`. The main task is to implement concrete versions of the protocols for the Crafter environment (`crafter.state_export.WorldState`).

## 3. Workstream 1: Trajectory Collection

To ensure robustness and testability, we will adopt a more functional approach. Instead of passing a mutable `Env` object between components, scenarios will be responsible for describing a desired `WorldState`, which the collection strategy will then execute.

### 3.1. State Extraction Helper

The `crafter.Env` object does not have a simple `.get_state()` method. We must use the `export_world_state` function. To simplify this, a helper function should be created.

```python
# In a new file: src/distant_sunburn/evaluator/crafter/utils.py

from crafter.env import Env
from crafter.state_export import WorldState, export_world_state

def get_world_state(env: Env) -> WorldState:
    """Exports the WorldState from a crafter.Env instance."""
    return export_world_state(
        env._world, 
        view=env._config.view, 
        step_count=env._step
    )
```

### 3.2. Architecture: Strategy-Based Collection

The `CrafterTrajectoryCollector` will instantiate and run a given collection strategy.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py

from crafter.functional_env import EnvConfig
from ..core import SymbolicTransition, TrajectoryCollector

class CollectionStrategy(Protocol):
    """A protocol for a single method of collecting transitions in Crafter."""
    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        ...

class CrafterTrajectoryCollector(TrajectoryCollector[WorldState]):
    def __init__(self, strategy: CollectionStrategy):
        self.strategy = strategy

    def collect_transitions(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        return self.strategy.collect(num_transitions)
```

### 3.3. Strategy 1: Random Movement Policy

This strategy will now create its own internal `Env` to generate states.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py
from .utils import get_world_state

class RandomMovementStrategy(CollectionStrategy):
    def __init__(self, env_config: EnvConfig, seed: int):
        self.env_config = env_config
        self.rng = random.Random(seed)
        self.movement_actions = ["move_left", "move_right", "move_up", "move_down"]

    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        env = Env(self.env_config)
        env.reset()
        
        transitions = []
        state = get_world_state(env)

        for _ in range(num_transitions):
            action = self.rng.choice(self.movement_actions)
            prev_state = state
            
            env.step(action)
            state = get_world_state(env)
            
            transitions.append(SymbolicTransition(prev_state, action, state))
        
        return transitions
```

### 3.4. Strategy 2: Scenario-Based Collection

This strategy uses `Scenario` objects that are now stateless factories for `WorldState` objects.

**New `Scenario` Protocol:**
```python
# In: src/distant_sunburn/evaluator/crafter/scenarios.py
from crafter.functional_env import EnvConfig
from crafter.state_export import WorldState

class Scenario(Protocol):
    @property
    def name(self) -> str: ...

    def get_initial_state(self, env_config: EnvConfig) -> WorldState:
        """Creates and returns the specific starting WorldState for this scenario."""
        ...

    def get_actions(self) -> list[str]: ...
```

**New `ScenarioBasedStrategy`:** This strategy is now much cleaner. It uses the pure `crafter.functional_env.transition` function to step through the scenario, ensuring no side effects.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py
from crafter.functional_env import transition
from .scenarios import Scenario

class ScenarioBasedStrategy(CollectionStrategy):
    def __init__(self, scenarios: list[Scenario], env_config: EnvConfig):
        self.scenarios = scenarios
        self.env_config = env_config

    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        transitions = []
        for scenario in self.scenarios:
            initial_state = scenario.get_initial_state(self.env_config)
            actions = scenario.get_actions()

            state = initial_state
            for action in actions:
                prev_state = state
                state = transition(prev_state, action)
                transitions.append(SymbolicTransition(prev_state, action, state))
        
        return transitions
```

## 4. Workstream 2: Distractor Generation (Unchanged)
The architecture for distractor generation remains the same as it is already functional and decoupled.

## 5. Factory and Final Assembly

The factory will be updated to reflect the new stateless strategy architecture.

```python
# In: src/distant_sunburn/evaluator/crafter/factory.py

from .components import (
    CrafterTrajectoryCollector, 
    ScenarioBasedStrategy,
    CrafterDistractorGenerator
)
from .scenarios import CraftWoodenPickaxeScenario, CowMovementScenario

class CrafterEvaluationFactory:
    def __init__(self, env_config: EnvConfig, policy_seed: int = 42):
        self.env_config = env_config
        self.policy_seed = policy_seed

    def create_context(
        self, config: EvaluationConfig, num_transitions: int
    ) -> EvaluationContext[WorldState]:

        scenarios = [CraftWoodenPickaxeScenario(), CowMovementScenario()]
        strategy = ScenarioBasedStrategy(scenarios, self.env_config)
        
        collector = CrafterTrajectoryCollector(strategy)
        test_transitions = collector.collect_transitions(num_transitions)

        distractor_generator = CrafterDistractorGenerator(seed=self.policy_seed)

        return EvaluationContext(
            config=config,
            test_transitions=test_transitions,
            distractor_generator=distractor_generator,
            edit_distance_calculator=JSONPatchEditDistance(),
        )
```

## 6. Implementation Plan (Revised)

1.  **Setup:**
    *   Create `src/distant_sunburn/evaluator/crafter/utils.py` and implement the `get_world_state` helper function.
    *   Create `src/distant_sunburn/evaluator/crafter/scenarios.py` and `.../mutators.py`.
    *   Ensure test helpers from `external/crafter_refactored` are accessible.
2.  **Implement Trajectory Collection:**
    *   Update `CollectionStrategy` protocol in `components.py` to remove the `env` argument.
    *   Implement the revised `RandomMovementStrategy`.
    *   Define the new `Scenario` protocol in `scenarios.py`.
    *   Implement at least two scenarios (e.g., `CraftWoodenPickaxeScenario`) using the new stateless approach.
    *   Implement the revised `ScenarioBasedStrategy`.
    *   Implement the simplified `CrafterTrajectoryCollector`.
3.  **Implement Distractor Generation:** (No change in plan)
4.  **Update Factory:** Modify `CrafterEvaluationFactory` to match the new design.
5.  **Testing:** (No change in plan)

## 7. Appendix: Detailed Scenario Implementation Guide (Revised)

This guide demonstrates creating a scenario with the new, cleaner architecture.

### Step 1 & 2: Define Goal and Pre-conditions (Unchanged)
- **Goal:** Test crafting a wooden pickaxe.
- **Pre-conditions:** Player needs 2 wood, must be next to a table.

### Step 3: Code the Scenario

The scenario now encapsulates its own setup logic, creating a temporary environment to produce the desired starting state.

```python
# In: src/distant_sunburn/evaluator/crafter/scenarios.py

from crafter.env import Env
from crafter.functional_env import EnvConfig
from crafter.state_export import WorldState
from . import Scenario 
from .utils import get_world_state # Import the new helper
from tests.helpers import player_utils, world_utils

class CraftWoodenPickaxeScenario(Scenario):
    @property
    def name(self) -> str:
        return "craft_wooden_pickaxe"

    def get_initial_state(self, env_config: EnvConfig) -> WorldState:
        """
        Creates a temporary environment, configures it to the desired
        starting conditions, and returns the resulting WorldState.
        """
        # Create a temporary, isolated env for setup.
        env = Env(env_config)
        env.reset()

        # Perform setup logic on the temporary env.
        player = env._player
        world = env._world
        player_utils.set_player_position(player, (5, 5))
        world_utils.set_tile_material(world, (5, 6), "table")
        player_utils.set_player_inventory_item(player, "wood", 2)
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)

        # Use the helper to export the configured state.
        # The temporary env is then discarded.
        return get_world_state(env)

    def get_actions(self) -> list[str]:
        return ["make_wood_pickaxe"]
```

### Step 4: Integrate the Scenario (Unchanged)

The integration in the factory remains the same conceptually: instantiate the scenario and add it to the strategy's list. The code in the factory is now simpler as it doesn't need to manage an `Env` instance directly.