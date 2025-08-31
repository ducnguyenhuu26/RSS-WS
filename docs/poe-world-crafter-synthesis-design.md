# PoE-World Synthesis for Crafter: A Design Document

**Date**: 2025-01-27  
**Purpose**: To provide a clean, type-safe, and easily implementable design for adapting the PoE-World synthesis algorithm to the Crafter environment, while remaining faithful to the original architecture's limitations.

## 1. Overview

This document outlines the design for a reimplementation of the PoE-World expert synthesis algorithm, specifically adapted for the Crafter environment. The primary goal is to create a baseline for comparison in our research paper. As such, this design intentionally preserves the limitations of the original PoE-World architecture and avoids introducing any novel improvements.

The core principles of this design are:

-   **Faithfulness to PoE-World**: The design will mirror the original's object-centric, modular, and surprise-driven approach.
-   **Minimal Domain Knowledge**: We will avoid hardcoding game mechanics, adhering to the principle of learning from observation with minimal human guidance.
-   **Type Safety**: The design will leverage Pydantic models from `crafter.state_export` for a robust and type-safe implementation.
-   **Limited Observables**: Experts will only be synthesized for a small set of attributes (position and health), as defined in `src/distant_sunburn/poe_world/crafter/observable_extractor.py`.

## 2. Core Interfaces

The following interfaces will form the foundation of our implementation:

### 2.1. `ExpertSynthesizerProtocol`

This protocol defines the core interface for synthesizing experts from a sequence of state transitions.

```python
from typing import Protocol, List, Awaitable
from crafter.state_export import WorldState

class Expert(Protocol):
    """Represents a synthesized expert function."""
    code: str
    object_type: str

class ExpertSynthesizerProtocol(Protocol):
    """Protocol for expert synthesis from state transitions."""

    async def synthesize_experts(
        self,
        transitions: List[Tuple[WorldState, str, WorldState]],
        object_type: str
    ) -> List[Expert]:
        """Synthesize expert programs from state transitions."""
        ...
```

### 2.2. `StateTransition`

A state transition will be represented as a tuple of `(input_state, action, output_state)`, where each state is a `crafter.state_export.WorldState` object.

### 2.3. `ObservableExtractor`

The `ObservableExtractor` from `src/distant_sunburn/poe_world/crafter/observable_extractor.py` will be used to define the scope of our synthesis. We will only synthesize experts for the following attributes:

-   `player_position_x`
-   `player_position_y`
-   `player_health`
-   `entity_{entity_id}_position_x`
-   `entity_{entity_id}_position_y`
-   `entity_{entity_id}_health`

## 3. Synthesis Pipeline

The end-to-end synthesis process is as follows:

### Step 1: Per-Object-Type Processing

For each object type in Crafter (`player`, `cow`, `zombie`, `skeleton`, etc.), we will create a separate `ObjModelLearner`. This learner will be responsible for managing the synthesis process for that specific object type.

```python
# Main loop
for obj_type in ["player", "cow", "zombie", ...]:
    learner = ObjModelLearner(config, obj_type)
    learner.process_transitions(experience_buffer)
```

**Assumption**: Each object type learns independently. This preserves the modularity and limitations of the original PoE-World.

### Step 2: Surprise-Driven Synthesis

Synthesis will only be triggered for "surprising" transitions—those that the current set of experts cannot explain well.

```python
# Inside ObjModelLearner
def process_transitions(self, transitions):
    for transition in transitions:
        if not self._explain_well(transition):
            self._synthesize_for_transition(transition)
```

A transition is considered surprising if the log probability of the observed outcome, according to the current mixture of experts, is below a certain threshold.

### Step 3: Custom View Creation

For each surprising transition, we will create a "custom view" of the state that is specific to the object type being processed. This view will only contain objects of the target type.

```python
# Inside ObjModelLearner
def _create_custom_view(self, state: WorldState) -> WorldState:
    # Create a deep copy to avoid modifying the original state
    custom_state = state.model_copy(deep=True)
    
    # Filter objects to only include the target type
    custom_state.objects = [
        obj for obj in state.objects if obj.name == self.obj_type
    ]
    
    return custom_state
```

**Assumption**: Each object type only sees its own objects. This is a key limitation that prevents the synthesis of cross-object-type rules (e.g., for AoE attacks).

### Step 4: Binary Interaction Synthesizer

To remain faithful to PoE-World's original design, we will include a `BinaryInteractionSynthesizer` that handles "touching" relationships between objects. This synthesizer is fundamental to PoE-World's approach and is not domain-specific knowledge, as it only requires spatial adjacency detection.

```python
class BinaryInteractionSynthesizer:
    """Synthesizer for binary touching relationships between objects."""
    
    def _detect_touching_objects(self, state: WorldState) -> List[Tuple[str, str]]:
        """Detect which objects are touching each other (Manhattan distance = 1)."""
        touching = []
        for obj1 in state.objects:
            for obj2 in state.objects:
                if obj1.entity_id != obj2.entity_id:
                    # Check if objects are adjacent
                    if abs(obj1.position.x - obj2.position.x) + abs(obj1.position.y - obj2.position.y) == 1:
                        touching.append((f"{obj1.name}_{obj1.entity_id}", f"{obj2.name}_{obj2.entity_id}"))
        return touching
    
    def synthesize(self, input_state: WorldState, action: str, output_state: WorldState) -> List[Expert]:
        # Detect touching relationships in input state
        touching_pairs = self._detect_touching_objects(input_state)
        
        # Generate rules based on touching relationships
        # This preserves the original PoE-World approach
        ...
```

This synthesizer enables rules like:
```python
def alter_player_objects(state: WorldState, action: str) -> WorldState:
    if action == "attack":
        player = state.player
        for obj in state.objects:
            if obj.name == "zombie" and obj.touches(player):
                obj.health = max(0, obj.health - 20)
    return state
```

### Step 5: Generic Action Synthesizer

In addition to the binary interaction synthesizer, we will use a single, generic `ActionSynthesizer` for all other object types and mechanics. This synthesizer will be responsible for generating rules based on observed changes to the limited set of attributes (position and health).

```python
# Inside ObjModelLearner
def _synthesize_for_transition(self, transition):
    input_state, action, output_state = transition
    
    # Create custom views
    custom_input_state = self._create_custom_view(input_state)
    custom_output_state = self._create_custom_view(output_state)
    
    # Synthesize experts using both synthesizers
    interaction_synthesizer = BinaryInteractionSynthesizer(self.obj_type, self.llm)
    action_synthesizer = ActionSynthesizer(self.obj_type, self.llm)
    
    interaction_experts = interaction_synthesizer.synthesize(
        custom_input_state, action, custom_output_state
    )
    action_experts = action_synthesizer.synthesize(
        custom_input_state, action, custom_output_state
    )
    
    experts = interaction_experts + action_experts
    
    # Add experts to the model
    self.moe.add_experts(experts)
```

This approach adheres to the constraint of not having specialized synthesizers for different mechanics, which would give the baseline an unfair advantage.

### Step 6: LLM-Based Synthesis

The `ActionSynthesizer` will use a two-stage LLM process to generate experts:

**Stage 1: Observation Generation**

The LLM will be prompted with the custom view of the input and output states and asked to generate natural language descriptions of the changes.

**Example Prompt (for `zombie` type):**

```
Input list of objects:
- zombie object (id = 1) with position = (x=11, y=10) and health = 50

Output list of object changes:
- The zombie object (id = 1) now has health = 30

Please describe the effects of the action on the zombie objects.
```

**Stage 2: Code Generation**

The LLM will be prompted with the generated observations and the action, and asked to generate a Python function that implements the observed effects.

**Example Prompt (for `zombie` type):**

```
We observe that the action 'attack' has the following effects on zombie objects:
- The zombie objects lose 20 health

Please synthesize a Python function that implements this effect. The format should be:

def alter_zombie_objects(state: WorldState, action: str) -> WorldState:
    # your code here
    return state
```

**Generated Expert:**

```python
def alter_zombie_objects(state: WorldState, action: str) -> WorldState:
    if action == "attack":
        zombie_objs = [obj for obj in state.objects if obj.name == "zombie"]
        for zombie in zombie_objs:
            # Note: This is overly broad and affects ALL zombies,
            # which is a limitation we are preserving.
            zombie.health = max(0, zombie.health - 20)
    return state
```

### Step 7: Expert Integration

The generated experts will be added to the Mixture of Experts (MoE) model for the corresponding object type. The weights of the experts will then be fitted to the experience buffer to determine which experts are most accurate.

## 4. Limitations and Faithfulness to PoE-World

This design intentionally preserves the following limitations of the original PoE-World architecture:

-   **No Cross-Object-Type Rules**: Since each object type learns independently, the system cannot generate rules for interactions that affect multiple object types simultaneously (e.g., AoE attacks).
-   **Limited Spatial Reasoning**: The custom view removes information about other object types, preventing the synthesis of rules based on spatial relationships between different types of objects.
-   **No Complex Mechanics**: Without specialized synthesizers, the system will not be able to learn complex mechanics like POMDPs, constraints, or multi-timestep effects.
-   **Over-Generalization**: The generated rules may be overly broad (e.g., an attack affecting all zombies, not just the one being attacked), which is a direct consequence of the limited context provided to the synthesizers.

By adhering to these limitations, this design provides a faithful implementation of the PoE-World algorithm for Crafter, making it a fair and suitable baseline for our research.
