# Issue #001: PoE-World Cannot Express Complex Spatial Relationships Between Object Types

## Summary

PoE-World's current design has a fundamental limitation: **it cannot express complex spatial relationships between different object types**. This severely restricts its ability to model realistic game mechanics like targeted attacks, area-of-effect abilities, and conditional interactions based on proximity.

## Background

PoE-World uses a modular, object-type-specific learning approach where:

1. Each object type (e.g., 'player', 'zombie', 'ladder') gets its own synthesizer
2. Each synthesizer only sees objects of its assigned type
3. Synthesizers generate rules independently for their object type
4. Cross-object-type interactions are not naturally expressible

## Problem Description

### The Zombie Attack Scenario

Consider a Crafter environment where a player attacks a zombie:

**Initial State:**
```python
input_state = WorldState(
    objects=[
        PlayerState(entity_id=0, position=Position(x=10, y=10), health=100, action="attack"),
        ZombieState(entity_id=1, position=Position(x=11, y=10), health=50, cooldown=0),
        ZombieState(entity_id=2, position=Position(x=20, y=20), health=50, cooldown=0)  # Far away zombie
    ]
)
```

**After Attack:**
```python
output_state = WorldState(
    objects=[
        PlayerState(entity_id=0, position=Position(x=10, y=10), health=100, action="attack"),
        ZombieState(entity_id=1, position=Position(x=11, y=10), health=30, cooldown=0),  # Damaged
        ZombieState(entity_id=2, position=Position(x=20, y=20), health=50, cooldown=0)   # Unaffected
    ]
)
```

### How PoE-World Processes This

**Player Synthesizer:**
- Sees: `[PlayerState(entity_id=0, position=Position(x=10, y=10), health=100, action="attack")]`
- Effects: `[]` (no changes to player)
- Result: No rules generated

**Zombie Synthesizer:**
- Sees: `[ZombieState(entity_id=1, position=Position(x=11, y=10), health=50)]` → `[ZombieState(entity_id=1, position=Position(x=11, y=10), health=30)]`
- Effects: `["The zombie object (id = 1) sets health to 30"]`
- Generated Rule:
```python
def alter_zombie_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == "attack":
        zombie_objs = obj_list.get_objs_by_obj_type('zombie')
        for zombie_obj in zombie_objs:
            zombie_obj.health = RandomValues([30, 25, 35])  # Affects ALL zombies!
    return obj_list
```

### The Problem

The generated rule is **overly broad** and **incorrect**:
- It affects **all zombies** in the world, not just the one being attacked
- It doesn't capture that only zombies **within attack range** should be affected
- It doesn't understand the **spatial relationship** between player and zombie

## Root Cause Analysis

### 1. **Object Type Isolation**
Each synthesizer only sees objects of its assigned type:
```python
# Zombie synthesizer sees:
input_objects = self.objects_selector(x.input_state)  # Only zombie objects
output_objects = self.objects_selector(x.output_state)  # Only zombie objects
```

### 2. **Missing Spatial Context**
The synthesizer doesn't know:
- Where the player is located
- Which zombies are within attack range
- Whether there are obstacles between player and zombie
- The spatial relationship between objects

### 3. **No Cross-Object-Type Rules**
PoE-World has no mechanism to express rules that involve multiple object types:
```python
# This type of rule cannot be expressed:
def alter_objects_in_attack_range(obj_list: ObjList, action: str) -> ObjList:
    if action == "attack":
        player = obj_list.get_objs_by_obj_type('player')[0]
        for obj in obj_list.objs:
            if obj.obj_type == 'zombie' and obj.distance(player.position) <= 1:
                obj.health = max(0, obj.health - 20)
    return obj_list
```

## Impact

This limitation affects many realistic game mechanics:

1. **Targeted Attacks**: Cannot express "attack affects only nearby enemies"
2. **Area-of-Effect Abilities**: Cannot express "ability affects all objects within radius R"
3. **Line-of-Sight Effects**: Cannot express "projectile travels until hitting obstacle"
4. **Conditional Interactions**: Cannot express "interact only with closest object"
5. **Spatial Constraints**: Cannot express "cannot move through walls"

## Investigation Tasks

To confirm this limitation exists, an engineer should examine:

### 1. **Synthesizer Input Filtering**
**File:** `external/poe-world/learners/synthesizer.py`
**Check:** How `objects_selector` and `interactions_selector` filter input:
```python
# Look for:
input_target_obj_list = self.objects_selector(x.input_state)  # Only objects of target type
input_target_int_list = self.interactions_selector(x.input_state.get_obj_interactions())
```

**Question:** Does this filtering prevent synthesizers from seeing objects of other types?

### 2. **LLM Prompt Content**
**File:** `external/poe-world/prompts/synthesizer.py`
**Check:** What information is included in synthesis prompts:
```python
# Look for:
prompt.format(input=self._prep_interpret_input(input_target_obj_list, input_target_int_list),
              effects=list_to_bullets(effects))
```

**Question:** Do prompts include spatial relationship information between different object types?

### 3. **Generated Rule Format**
**File:** `external/poe-world/prompts/synthesizer.py`
**Check:** The `explain_event_prompt` template:
```python
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
```

**Question:** Does this format allow rules that reference multiple object types?

### 4. **Interaction Representation**
**File:** `external/poe-world/learners/synthesizer.py`
**Check:** How object interactions are represented:
```python
# Look for:
input_target_int_list = self.interactions_selector(x.input_state.get_obj_interactions())
```

**Question:** Do interactions include spatial relationship information beyond simple touching?

### 5. **Synthesizer Types**
**File:** `external/poe-world/learners/synthesizer.py`
**Check:** Available synthesizer types:
- `ActionSynthesizer`
- `PassiveMovementSynthesizer`
- `MultiTimestepActionSynthesizer`
- etc.

**Question:** Are there any synthesizers designed to handle cross-object-type spatial relationships?

## Suggested Solutions

If the limitation is confirmed, potential solutions include:

### 1. **Enhanced Interaction Information**
Include spatial relationship data in interactions:
```python
interactions = [
    "zombie object (id = 1) is within attack range of player object (id = 0)",
    "zombie object (id = 1) is at distance 1 from player object (id = 0)"
]
```

### 2. **Cross-Object-Type Synthesizers**
New synthesizer types that can handle multiple object types:
```python
class SpatialRelationshipSynthesizer(Synthesizer):
    def synthesize(self, transitions):
        # Look for patterns involving multiple object types
```

### 3. **Spatial Context in Prompts**
Modify prompts to include spatial relationship information:
```python
prompt = """
Given that action {action} affects objects of types {affected_types} 
within distance {radius} of the player, generate rules that capture this spatial relationship.
"""
```

### 4. **Multi-Object-Type Rules**
Allow rules that reference multiple object types:
```python
def alter_objects_in_spatial_relationship(obj_list: ObjList, action: str) -> ObjList:
    # Can reference multiple object types
```

## Conclusion

This spatial relationship limitation represents a significant constraint on PoE-World's ability to model realistic game mechanics. While the modular design provides interpretability and scalability, it sacrifices expressiveness for complex spatial interactions.

**Priority:** High - This affects the system's ability to model realistic game environments.

**Effort:** Medium - Would require significant changes to synthesizer architecture and prompt design.

**Risk:** Medium - Changes could affect existing functionality and model interpretability.
