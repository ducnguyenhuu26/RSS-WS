# PRD: Hybrid Evaluation Framework Implementation

## Context

Implementation of the hybrid evaluation framework for symbolic world models described in `1-hybrid-evaluation-framework-for-symbolic-wms.md`. This system evaluates world models by combining generative and discriminative tests to measure both utility (for planning) and scientific accuracy (probability distribution understanding).

## Architecture 

### Hexagonal Design
- **Core Domain**: Evaluation orchestration, metrics calculation, distractor generation
- **Ports**: World model protocol, environment protocol, persistence protocol  
- **Adapters**: 1D environment wrapper, JSON serialization, file-based storage

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

### Environment Protocol
```python
class SymbolicEnvironment(Protocol[MetadataT]):
    def transition(self, state: MetadataT, action: Any) -> MetadataT:
        """True transition function: (s, a) -> s'"""
        ...
    
    def initial_state(self, seed: int) -> MetadataT:
        """Generate reproducible initial state"""
        ...
    
    def random_action(self) -> Any:
        """Sample random action from action space"""
        ...
```

### Configuration Objects
```python
@dataclass(frozen=True)
class EvaluationConfig:
    num_transitions: int = 100
    num_distractors: int = 5
    temporal_distractor_gap: int = 50
    random_seed: int = 42

@dataclass(frozen=True)
class DistractorConfig:
    temporal_enabled: bool = True
    semantic_enabled: bool = True
    mutator_functions: list[Callable[[MetadataT], MetadataT]] = field(default_factory=list)

@dataclass(frozen=True)
class EvaluationResults:
    mean_generative_error: float
    discriminative_accuracy: float
    discriminative_accuracy_by_distractor_type: dict[str, float]
    total_transitions_evaluated: int
```

## Test Environment: 1D Slippery World

### Assessment of benchmark_1d/environment.py
**Sufficient for testing.** Key properties:
- **Stochasticity**: MovementLaw slip probability (10%), LightLaw toggle probability (20%)
- **Multiple state components**: Player position, light states
- **Deterministic base rules** with probabilistic perturbations
- **Bounded state space**: Clear domain for likelihood calculation

### Environment Wrapper Implementation
```python
class Environment1DWrapper:
    def __init__(self, config: WorldConfig, seed: int):
        self.base_environment = # 1D environment setup
        self.rng = random.Random(seed)
    
    def transition(self, state: GameState, action: Action) -> GameState:
        # Ensure rng state matches between true environment and world model
        return transition_function(state, action, DEFAULT_LAWS)
    
    def to_json_serializable(self, state: GameState) -> dict:
        """Convert GameState dataclass to JSON-serializable dict"""
        return {
            "config": {"width": state.config.width, "switch_point": state.config.switch_point},
            "player": {"position": state.player.position},
            "lights": [{"position": light.position, "is_on": light.is_on} for light in state.lights]
            # Exclude rng from serialization
        }
```

## Likelihood Function Design

### Direct Equality Approach
For the true transition function wrapped as a world model:
```python
class TrueTransitionWorldModel:
    def __init__(self, environment: Environment1DWrapper):
        self.environment = environment
        # CRITICAL: Hold reference to true transition function to maintain rng synchronization
        self._transition_function = environment.transition
    
    def evaluate_log_probability(self, next_state: GameState, current_state: GameState, action: Action) -> float:
        # Run true transition function with same rng state
        true_next_state = self._transition_function(current_state, action)
        
        # Direct equality check (works because we have access to true rng)
        if self._states_equal(next_state, true_next_state):
            return 0.0  # log(1) = 0 for perfect match
        else:
            return -math.inf  # log(0) = -inf for mismatch
```

**Rationale**: Direct equality works here because:
1. We control the true transition function and its random state
2. Generated states from true model should exactly match expected states
3. This is harsh for learned models but perfect for ground truth testing

## Edit Distance Implementation

### JSON Patch Distance
```python
import jsonpatch

def compute_edit_distance(state1: GameState, state2: GameState, wrapper: Environment1DWrapper) -> int:
    json1 = wrapper.to_json_serializable(state1)
    json2 = wrapper.to_json_serializable(state2)
    patch = jsonpatch.make_patch(json1, json2)
    return len(list(patch))
```

## Distractor Generation

### Temporal Distractors
```python
def generate_temporal_distractors(transitions: list[SymbolicTransition], 
                                 current_idx: int, 
                                 gap: int, 
                                 count: int) -> list[MetadataT]:
    """Sample states from transitions with |t - current_t| > gap"""
    eligible_indices = [i for i in range(len(transitions)) 
                       if abs(i - current_idx) > gap]
    return random.sample([transitions[i].next_metadata for i in eligible_indices], count)
```

### Semantic Mutators for 1D Environment
```python
def mutate_player_position(state: GameState) -> GameState:
    """Move player to invalid position"""
    new_state = copy.deepcopy(state)
    new_state.player.position = random.choice([
        state.player.position + 2,  # Jump too far
        -1,  # Out of bounds
        state.config.width  # Out of bounds
    ])
    return new_state

def mutate_light_states(state: GameState) -> GameState:
    """Randomly toggle light states without action justification"""
    new_state = copy.deepcopy(state)
    for light in new_state.lights:
        if random.random() < 0.5:
            light.is_on = not light.is_on
    return new_state
```

## Evaluation Protocol Implementation

```python
class HybridEvaluator:
    def __init__(self, config: EvaluationConfig, distractor_config: DistractorConfig):
        self.config = config
        self.distractor_config = distractor_config
    
    def evaluate(self, 
                world_model: EvaluatableWorldModel[MetadataT], 
                environment: SymbolicEnvironment[MetadataT]) -> EvaluationResults:
        
        # 1. Collect transitions via random policy
        transitions = self._collect_transitions(environment)
        
        generative_errors = []
        discriminative_successes = []
        
        for transition in transitions:
            # 2. Generate prediction
            pred_state = world_model.sample_next_state(transition.prev_metadata, transition.action)
            
            # 3. Measure generative error
            gen_error = compute_edit_distance(pred_state, transition.next_metadata, environment)
            generative_errors.append(gen_error)
            
            # 4. Generate distractors
            distractors = self._generate_distractors(transition, transitions)
            
            # 5. Construct candidate set
            candidates = [transition.next_metadata, pred_state] + distractors
            
            # 6. Evaluate log probabilities
            log_probs = {
                candidate: world_model.evaluate_log_probability(candidate, transition.prev_metadata, transition.action)
                for candidate in candidates
            }
            
            # 7. Check discriminative success
            max_prob = max(log_probs.values())
            true_state_prob = log_probs[transition.next_metadata]
            discriminative_successes.append(true_state_prob == max_prob)
        
        return EvaluationResults(
            mean_generative_error=np.mean(generative_errors),
            discriminative_accuracy=np.mean(discriminative_successes),
            discriminative_accuracy_by_distractor_type=self._analyze_by_distractor_type(),
            total_transitions_evaluated=len(transitions)
        )
```

## Sanity Check Test Design

### Test Scenario
```python
def test_true_vs_null_world_model():
    """True transition function should vastly outperform null model"""
    
    # Setup
    environment = Environment1DWrapper(config=WorldConfig(width=12, switch_point=6), seed=42)
    true_model = TrueTransitionWorldModel(environment)
    null_model = NullWorldModel()  # Always predicts no change
    
    evaluator = HybridEvaluator(
        config=EvaluationConfig(num_transitions=50),
        distractor_config=DistractorConfig()
    )
    
    # Evaluate both models
    true_results = evaluator.evaluate(true_model, environment)
    null_results = evaluator.evaluate(null_model, environment)
    
    # Assertions
    assert true_results.mean_generative_error < null_results.mean_generative_error
    assert true_results.discriminative_accuracy > null_results.discriminative_accuracy
    assert true_results.discriminative_accuracy > 0.9  # True model should be highly accurate
    assert null_results.discriminative_accuracy < 0.3  # Null model should perform poorly
```

### Null World Model
```python
class NullWorldModel:
    """Baseline model that predicts no state changes"""
    
    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT:
        return copy.deepcopy(current_state)  # No change prediction
    
    def evaluate_log_probability(self, next_state: MetadataT, current_state: MetadataT, action: Any) -> float:
        if self._states_equal(next_state, current_state):
            return 0.0
        else:
            return -5.0  # Low but not impossible probability for changes
```

## Critical Implementation Notes

### Random State Synchronization
**CRITICAL**: The true world model wrapper must maintain exact random state synchronization with the environment's transition function. Implementation approaches:
1. **Shared RNG Reference**: World model holds direct reference to environment's RNG
2. **State Copying**: Deep copy environment state before transition function calls  
3. **Seed Management**: Explicit seed coordination between environment and world model

**Failure Mode**: Mismatched random states will cause the "perfect" true model to fail discriminative tests due to stochastic divergence.

### JSON Serialization Edge Cases
- Handle nested dataclasses and enums
- Exclude non-serializable fields (RNG objects)
- Ensure deterministic field ordering for consistent patch generation

### Performance Considerations
- Batch distractor generation for efficiency
- Cache JSON serialization results within evaluation loop
- Parallelize evaluation across transitions if needed

## Success Criteria

1. **Interface Compliance**: All world models implement `EvaluatableWorldModel` protocol
2. **Sanity Test Passage**: True model >> Null model on both metrics
3. **Stochasticity Handling**: Consistent results across runs with same seed
4. **Distractor Quality**: Semantic mutators generate challenging but invalid states
5. **Performance**: Evaluate 100 transitions in <10 seconds

This framework provides the foundation for rigorous symbolic world model evaluation with clear separation of concerns and robust testing mechanisms.
