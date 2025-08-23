# PRD: Hybrid Evaluation Framework Implementation

## Context

Implementation of the hybrid evaluation framework for symbolic world models described in `1-hybrid-evaluation-framework-for-symbolic-wms.md`. This system evaluates world models by combining generative and discriminative tests to measure both utility (for planning) and scientific accuracy (probability distribution understanding).

## Architecture 

### Hexagonal Design with Dependency Injection
- **Core Domain**: Evaluation orchestration, metrics calculation
- **Injected Components**: Trajectory collection, edit distance calculation, distractor generation  
- **Environment Adapters**: Environment-specific implementations of injected components

Module location: `src/distant_sunburn/evaluator/`
Test location: `tests/evaluator/`

## Core Interfaces

### World Model Protocol
```python
class EvaluatableWorldModel(Protocol[MetadataT]):
    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT:
        """Generate single prediction by sampling from posterior P(s_next | s, a)"""
        ...
    
    def evaluate_log_probability(self, 
                                next_state: MetadataT, 
                                current_state: MetadataT, 
                                action: Any) -> float:
        """Compute log P(next_state | current_state, action)"""
        ...
```

### Environment Protocol (Minimal)
```python
class SymbolicEnvironment(Protocol[MetadataT]):
    def transition(self, state: MetadataT, action: Any) -> MetadataT:
        """True transition function: (s, a) -> s'"""
        ...
```

### Component Protocols (Injected Dependencies)

#### Trajectory Collection
```python
class TrajectoryCollector(Protocol[MetadataT]):
    def collect_transitions(self, 
                          environment: SymbolicEnvironment[MetadataT], 
                          num_transitions: int) -> list[SymbolicTransition[MetadataT]]:
        """Collect symbolic transitions using environment-specific policy"""
        ...

@dataclass(frozen=True)
class SymbolicTransition(Generic[MetadataT]):
    prev_metadata: MetadataT
    action: Any
    next_metadata: MetadataT
```

#### Edit Distance Calculation
```python
class EditDistanceCalculator(Protocol[MetadataT]):
    def compute_distance(self, state1: MetadataT, state2: MetadataT) -> float:
        """Compute structured edit distance between two states"""
        ...
```

#### Distractor Generation
```python
class DistractorGenerator(Protocol[MetadataT]):
    def generate_distractors(self, 
                           transition: SymbolicTransition[MetadataT], 
                           all_transitions: list[SymbolicTransition[MetadataT]], 
                           num_distractors: int) -> list[MetadataT]:
        """Generate plausible but incorrect next states"""
        ...
```

### Configuration and Results
```python
@dataclass(frozen=True)
class EvaluationConfig:
    num_transitions: int = 100
    num_distractors: int = 5
    random_seed: int = 42

@dataclass(frozen=True)
class EvaluationResults:
    mean_generative_error: float
    discriminative_accuracy: float
    discriminative_accuracy_by_distractor_type: dict[str, float]
    total_transitions_evaluated: int
```

## Core Evaluation Logic

### Dependency-Injected Evaluator
```python
class HybridEvaluator:
    def __init__(self, 
                 config: EvaluationConfig,
                 trajectory_collector: TrajectoryCollector[MetadataT],
                 edit_distance_calc: EditDistanceCalculator[MetadataT], 
                 distractor_generator: DistractorGenerator[MetadataT]):
        self.config = config
        self.trajectory_collector = trajectory_collector
        self.edit_distance_calc = edit_distance_calc
        self.distractor_generator = distractor_generator
    
    def evaluate(self, 
                world_model: EvaluatableWorldModel[MetadataT], 
                environment: SymbolicEnvironment[MetadataT]) -> EvaluationResults:
        """Core evaluation logic - environment agnostic"""
        
        # 1. Collect transitions using injected collector
        transitions = self.trajectory_collector.collect_transitions(
            environment, self.config.num_transitions
        )
        
        generative_errors = []
        discriminative_successes = []
        distractor_type_results = defaultdict(list)
        
        for transition in transitions:
            # 2. Generate prediction
            pred_state = world_model.sample_next_state(
                transition.prev_metadata, transition.action
            )
            
            # 3. Measure generative error using injected calculator
            gen_error = self.edit_distance_calc.compute_distance(
                pred_state, transition.next_metadata
            )
            generative_errors.append(gen_error)
            
            # 4. Generate distractors using injected generator
            distractors = self.distractor_generator.generate_distractors(
                transition, transitions, self.config.num_distractors
            )
            
            # 5. Construct candidate set
            candidates = [transition.next_metadata, pred_state] + distractors
            
            # 6. Evaluate log probabilities
            log_probs = {
                candidate: world_model.evaluate_log_probability(
                    candidate, transition.prev_metadata, transition.action
                )
                for candidate in candidates
            }
            
            # 7. Check discriminative success
            max_prob = max(log_probs.values())
            true_state_prob = log_probs[transition.next_metadata]
            discriminative_successes.append(true_state_prob == max_prob)
        
        return EvaluationResults(
            mean_generative_error=np.mean(generative_errors),
            discriminative_accuracy=np.mean(discriminative_successes),
            discriminative_accuracy_by_distractor_type=dict(distractor_type_results),
            total_transitions_evaluated=len(transitions)
        )
```

---

# Implementation Strategies

## Environment Adapters

### Design Principle
Each environment provides implementations of the injected component protocols. This separates environment-specific concerns from the core evaluation logic.

### 1D Environment Adapter Example
```python
class Environment1DAdapter:
    """Complete adapter for 1D benchmark environment"""
    
    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
    
    def create_environment(self) -> SymbolicEnvironment[GameState]:
        return Environment1DWrapper(self.config, self.seed)
    
    def create_trajectory_collector(self) -> TrajectoryCollector[GameState]:
        return RandomPolicy1DTrajectoryCollector(self.rng)
    
    def create_edit_distance_calculator(self) -> EditDistanceCalculator[GameState]:
        return JSONPatchEditDistance()
    
    def create_distractor_generator(self) -> DistractorGenerator[GameState]:
        return Semantic1DDistractorGenerator(self.config)

class Environment1DWrapper:
    """Minimal environment wrapper - only transition function"""
    
    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.base_seed = seed
    
    def transition(self, state: GameState, action: Action) -> GameState:
        return transition_function(state, action, DEFAULT_LAWS)
```

## Component Implementation Strategies

### Trajectory Collection Implementations

#### Random Policy Approach
```python
class RandomPolicy1DTrajectoryCollector:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.STAY]
    
    def collect_transitions(self, environment: SymbolicEnvironment[GameState], 
                          num_transitions: int) -> list[SymbolicTransition[GameState]]:
        transitions = []
        state = initial_state(seed=self.rng.randint(0, 2**31-1))
        
        for _ in range(num_transitions):
            action = self.rng.choice(self.actions)
            next_state = environment.transition(state, action)
            transitions.append(SymbolicTransition(state, action, next_state))
            state = next_state
        
        return transitions
```

### Edit Distance Implementations

#### JSON Patch Distance (for serializable states)
```python
class JSONPatchEditDistance:
    def compute_distance(self, state1: GameState, state2: GameState) -> float:
        json1 = self._to_json(state1)
        json2 = self._to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))
    
    def _to_json(self, state: GameState) -> dict:
        return {
            "player_position": state.player.position,
            "lights": [(light.position, light.is_on) for light in state.lights]
            # Exclude non-serializable fields like RNG
        }
```

#### Structural Distance (for complex states)
```python
class StructuralEditDistance:
    def compute_distance(self, state1: WorldState, state2: WorldState) -> float:
        # Focus on semantically meaningful differences
        distance = 0
        
        # Player differences
        if state1.player.position != state2.player.position:
            distance += abs(state1.player.position.x - state2.player.position.x)
            distance += abs(state1.player.position.y - state2.player.position.y)
        
        # Inventory differences
        for item in state1.player.inventory.model_fields:
            diff = abs(getattr(state1.player.inventory, item) - 
                      getattr(state2.player.inventory, item))
            distance += diff
        
        # Entity count differences
        distance += abs(len(state1.objects) - len(state2.objects))
        
        return distance
```

### Distractor Generation Implementations

#### Temporal Distractors
```python
class TemporalDistractorGenerator:
    def __init__(self, gap: int = 50):
        self.gap = gap
    
    def generate_distractors(self, transition: SymbolicTransition[MetadataT], 
                           all_transitions: list[SymbolicTransition[MetadataT]], 
                           num_distractors: int) -> list[MetadataT]:
        current_idx = all_transitions.index(transition)
        eligible_indices = [i for i in range(len(all_transitions)) 
                           if abs(i - current_idx) > self.gap]
        
        if len(eligible_indices) < num_distractors:
            return [transitions[i].next_metadata for i in eligible_indices]
        
        selected_indices = random.sample(eligible_indices, num_distractors)
        return [all_transitions[i].next_metadata for i in selected_indices]
```

#### Semantic Mutators
```python
class Semantic1DDistractorGenerator:
    def __init__(self, config: WorldConfig):
        self.config = config
        self.mutators = [
            self._mutate_player_position,
            self._mutate_light_states,
        ]
    
    def generate_distractors(self, transition: SymbolicTransition[GameState], 
                           all_transitions: list[SymbolicTransition[GameState]], 
                           num_distractors: int) -> list[GameState]:
        distractors = []
        for _ in range(num_distractors):
            mutator = random.choice(self.mutators)
            distractor = mutator(transition.next_metadata)
            distractors.append(distractor)
        return distractors
    
    def _mutate_player_position(self, state: GameState) -> GameState:
        new_state = copy.deepcopy(state)
        new_state.player.position = random.choice([
            state.player.position + 2,  # Jump too far
            -1,  # Out of bounds
            self.config.width  # Out of bounds
        ])
        return new_state
    
    def _mutate_light_states(self, state: GameState) -> GameState:
        new_state = copy.deepcopy(state)
        for light in new_state.lights:
            if random.random() < 0.5:
                light.is_on = not light.is_on
        return new_state
```

## Usage Examples

### Creating an Evaluator for 1D Environment
```python
# Setup environment adapter
adapter = Environment1DAdapter(
    config=WorldConfig(width=12, switch_point=6), 
    seed=42
)

# Create components
environment = adapter.create_environment()
trajectory_collector = adapter.create_trajectory_collector()
edit_distance_calc = adapter.create_edit_distance_calculator()
distractor_generator = adapter.create_distractor_generator()

# Create evaluator with injected dependencies
evaluator = HybridEvaluator(
    config=EvaluationConfig(num_transitions=100, num_distractors=5),
    trajectory_collector=trajectory_collector,
    edit_distance_calc=edit_distance_calc,
    distractor_generator=distractor_generator
)

# Evaluate a world model
results = evaluator.evaluate(world_model, environment)
```

### For More Complex Environments (Crafter)
```python
# Different implementations for complex state spaces
class CrafterAdapter:
    def create_edit_distance_calculator(self) -> EditDistanceCalculator[WorldState]:
        return StructuralEditDistance()  # Not JSON patch
    
    def create_trajectory_collector(self) -> TrajectoryCollector[WorldState]:
        return CrafterRandomPolicyCollector(action_space_size=17)
    
    def create_distractor_generator(self) -> DistractorGenerator[WorldState]:
        return CrafterSemanticMutators()  # Knows about inventory constraints

# Usage is identical - implementation details are hidden
crafter_adapter = CrafterAdapter(...)
crafter_evaluator = HybridEvaluator(
    config=EvaluationConfig(),
    trajectory_collector=crafter_adapter.create_trajectory_collector(),
    edit_distance_calc=crafter_adapter.create_edit_distance_calculator(),
    distractor_generator=crafter_adapter.create_distractor_generator()
)
```

---

# Testing and Validation

## Sanity Check Design

### Ground Truth vs Baseline Comparison
```python
def test_true_vs_null_world_model():
    """True transition function should vastly outperform null model"""
    
    # Setup components
    adapter = Environment1DAdapter(config=WorldConfig(width=12, switch_point=6), seed=42)
    environment = adapter.create_environment()
    
    # Create world models
    true_model = TrueTransitionWorldModel(environment)  # Perfect model
    null_model = NullWorldModel()  # Always predicts no change
    
    # Create evaluator with injected components
    evaluator = HybridEvaluator(
        config=EvaluationConfig(num_transitions=50),
        trajectory_collector=adapter.create_trajectory_collector(),
        edit_distance_calc=adapter.create_edit_distance_calculator(),
        distractor_generator=adapter.create_distractor_generator()
    )
    
    # Evaluate both models
    true_results = evaluator.evaluate(true_model, environment)
    null_results = evaluator.evaluate(null_model, environment)
    
    # Assertions - true model should dominate
    assert true_results.mean_generative_error < null_results.mean_generative_error
    assert true_results.discriminative_accuracy > null_results.discriminative_accuracy
    assert true_results.discriminative_accuracy > 0.9  # Near perfect
    assert null_results.discriminative_accuracy < 0.3  # Poor performance
```

### Baseline World Models for Testing
```python
class TrueTransitionWorldModel:
    """Perfect world model using actual transition function"""
    
    def __init__(self, environment: SymbolicEnvironment[MetadataT]):
        self.environment = environment
    
    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT:
        return self.environment.transition(current_state, action)
    
    def evaluate_log_probability(self, next_state: MetadataT, 
                               current_state: MetadataT, action: Any) -> float:
        # Perfect model: probability 1 for correct transition, 0 otherwise
        true_next = self.environment.transition(current_state, action)
        return 0.0 if self._states_equal(next_state, true_next) else -math.inf

class NullWorldModel:
    """Baseline model that predicts no state changes"""
    
    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT:
        return copy.deepcopy(current_state)
    
    def evaluate_log_probability(self, next_state: MetadataT, 
                               current_state: MetadataT, action: Any) -> float:
        if self._states_equal(next_state, current_state):
            return 0.0
        else:
            return -5.0  # Low but not impossible probability for changes
```

## Implementation Considerations

### Random State Management
**Critical for stochastic environments**: Ensure deterministic evaluation results by proper random state handling:

```python
class DeterministicEnvironmentWrapper:
    def __init__(self, base_environment, seed: int):
        self.base_environment = base_environment
        self.seed = seed
    
    def transition(self, state: MetadataT, action: Any) -> MetadataT:
        # Ensure reproducible transitions
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self.base_environment.transition(state, action)
```


## Success Criteria
**Sanity Test Passage**: True model >> Null model on both metrics  