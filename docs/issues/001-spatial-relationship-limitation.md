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

## Additional Problem: Causal Relationship Limitation

### The Zombie Attack Scenario (Reverse Case)

Consider the reverse scenario where a zombie attacks a player:

**Initial State:**
```python
input_state = WorldState(
    objects=[
        PlayerState(entity_id=0, position=Position(x=10, y=10), health=100),
        ZombieState(entity_id=1, position=Position(x=11, y=10), health=50, action="attack")
    ]
)
```

**After Zombie Attack:**
```python
output_state = WorldState(
    objects=[
        PlayerState(entity_id=0, position=Position(x=10, y=10), health=80),  # Health reduced
        ZombieState(entity_id=1, position=Position(x=11, y=10), health=50)
    ]
)
```

### How PoE-World Processes This

**Player Synthesizer:**
- **Sees:** `[PlayerState(entity_id=0, position=Position(x=10, y=10), health=100)]` → `[PlayerState(entity_id=0, position=Position(x=10, y=10), health=80)]`
- **Effects:** `["The player object (id = 0) sets health to 80"]`
- **Generated Rule:**
```python
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == "attack":  # But this is the zombie's action, not the player's!
        player_objs = obj_list.get_objs_by_obj_type('player')
        for player_obj in player_objs:
            player_obj.health = RandomValues([80, 75, 85])  # Random damage values
    return obj_list
```

**Zombie Synthesizer:**
- **Sees:** `[ZombieState(entity_id=1, position=Position(x=11, y=10), health=50, action="attack")]` → `[ZombieState(entity_id=1, position=Position(x=11, y=10), health=50)]`
- **Effects:** `[]` (no changes to zombie)
- **Result:** No rules generated

### The Causal Relationship Problem

The player synthesizer **has no way to know that it was the zombie that caused the damage** because:

1. **Cannot see the zombie** - it only sees player objects due to object-type filtering:
   ```python
   # Lines 75-85: external/poe-world/learners/synthesizer.py
   input_target_obj_list = self.objects_selector(x.input_state)  # Only player objects
   ```

2. **Cannot see the zombie's action** - it doesn't know the zombie performed an "attack":
   ```python
   # Lines 867-966: external/poe-world/prompts/synthesizer.py
   # Prompts only show interactions as "touching" relationships, not causal actions
   "Interaction -- player object (id = 0) is touching ladder object (id = 2)"
   ```

3. **Cannot understand spatial proximity** - it doesn't know the zombie was adjacent to the player:
   ```python
   # Lines 549-600: external/poe-world/classes/helper.py
   # touches() method only checks collision, doesn't capture attack range or causality
   def touches(self, other, touch_side=-1, touch_percent=0):
       return pygame.Rect(self.x - 1, self.y - 1, self.w + 2, self.h + 2).colliderect(
           pygame.Rect(other.x, other.y, other.w, other.h))
   ```

4. **Cannot determine causality** - it just sees the player's health changed, but doesn't know why

### What the Player Synthesizer Actually Sees

The player synthesizer only sees:
```python
# Input: Only player objects (no zombie context)
[PlayerState(entity_id=0, position=Position(x=10, y=10), health=100)]

# Output: Only player objects (no zombie context)
[PlayerState(entity_id=0, position=Position(x=10, y=10), health=80)]

# Effects: Only player changes (no causal information)
["The player object (id = 0) sets health to 80"]
```

It has **no context** about:
- What other objects exist in the world
- What actions other objects performed
- Whether the player was touching anything
- What caused the health change

### The Generated Rule Would Be Incorrect

The player synthesizer might generate a rule like:
```python
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    # This rule makes no sense - it suggests the player's own action damages them
    if action == "attack":  # But this is the zombie's action!
        player_objs = obj_list.get_objs_by_obj_type('player')
        for player_obj in player_objs:
            player_obj.health = RandomValues([80, 75, 85])
    return obj_list
```

This rule is **completely wrong** because:
1. It suggests the player's own "attack" action damages them
2. It doesn't capture that damage comes from external sources
3. It doesn't understand the spatial relationship (zombie must be adjacent)
4. It doesn't capture the causal relationship (zombie's action → player's damage)

### Root Cause: Object-Type Isolation in Event Processing

The causal relationship limitation stems from the same architectural constraint as the spatial relationship limitation. In `external/poe-world/learners/synthesizer.py`:

```python
# Lines 44-50: Each synthesizer is isolated to its object type
self.objects_selector = ObjTypeObjSelector(self.obj_type)
self.interactions_selector = ObjTypeInteractionSelector(self.obj_type)

# Lines 75-85: Input filtering prevents cross-object-type context
input_target_obj_list = self.objects_selector(x.input_state)  # Only objects of target type
input_target_int_list = self.interactions_selector(x.input_state.get_obj_interactions())  # Only interactions involving target type
```

This isolation means that when an object of type A affects an object of type B, the synthesizer for type B cannot see the causal agent (object A) or its actions.

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

### 3. **Missing Causal Context**
The synthesizer doesn't know:
- What actions other objects performed
- Which object caused a particular effect
- Whether changes are self-inflicted or externally caused
- The causal chain of events

### 4. **No Cross-Object-Type Rules**
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

### 5. **No Causal Relationship Understanding**
PoE-World cannot understand or express causal relationships between different object types:
```python
# This type of rule cannot be expressed:
def alter_player_when_attacked_by_zombie(obj_list: ObjList, action: str) -> ObjList:
    if action == "attack":  # zombie's action
        player = obj_list.get_objs_by_obj_type('player')[0]
        zombies = obj_list.get_objs_by_obj_type('zombie')
        for zombie in zombies:
            if zombie.distance(player.position) <= 1:  # zombie is adjacent
                player.health = max(0, player.health - 20)  # zombie damages player
    return obj_list
```

## Impact

This limitation affects many realistic game mechanics:

1. **Targeted Attacks**: Cannot express "attack affects only nearby enemies"
2. **Area-of-Effect Abilities**: Cannot express "ability affects all objects within radius R"
3. **Line-of-Sight Effects**: Cannot express "projectile travels until hitting obstacle"
4. **Conditional Interactions**: Cannot express "interact only with closest object"
5. **Spatial Constraints**: Cannot express "cannot move through walls"
6. **Causal Relationships**: Cannot express "zombie's attack damages player" or "player's action affects zombie"
7. **External Effects**: Cannot distinguish between self-inflicted and externally-caused changes

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

### 6. **Causal Relationship Processing**
**File:** `external/poe-world/learners/synthesizer.py`
**Check:** How synthesizers handle effects caused by other object types:
```python
# Look for:
effects = self._get_natural_language_effects(x)  # How are effects attributed?
```

**Question:** Can synthesizers distinguish between self-inflicted and externally-caused changes?

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

### 5. **Causal Relationship Modeling**
Add support for causal relationships between object types:
```python
def alter_player_when_attacked_by_zombie(obj_list: ObjList, action: str) -> ObjList:
    # Can express that zombie's action affects player
```

## Conclusion

This spatial relationship limitation represents a significant constraint on PoE-World's ability to model realistic game mechanics. While the modular design provides interpretability and scalability, it sacrifices expressiveness for complex spatial interactions and causal relationships.

**Priority:** High - This affects the system's ability to model realistic game environments.

**Effort:** Medium - Would require significant changes to synthesizer architecture and prompt design.

**Risk:** Medium - Changes could affect existing functionality and model interpretability.

---

## Investigation Results

### Executive Summary

After conducting a thorough investigation of the PoE-World codebase, **all six questions have been answered in the affirmative** - confirming that PoE-World has fundamental limitations in expressing complex spatial relationships and causal relationships between different object types. These limitations are architectural in nature and affect the system's ability to model realistic game mechanics.

### Detailed Findings

#### 1. **Synthesizer Input Filtering** ✅ **CONFIRMED**

**Location:** `external/poe-world/learners/synthesizer.py` (Lines 44-50, 75-85)

**Finding:** Each synthesizer is initialized with object-type-specific selectors that prevent access to objects of other types:

```python
# Each synthesizer only sees objects of its assigned type
self.objects_selector = ObjTypeObjSelector(self.obj_type)
self.interactions_selector = ObjTypeInteractionSelector(self.obj_type)

# Input filtering in synthesis process
input_target_obj_list = self.objects_selector(x.input_state)  # Only objects of target type
input_target_int_list = self.interactions_selector(x.input_state.get_obj_interactions())  # Only interactions involving target type
```

**Impact:** Synthesizers cannot access spatial context from other object types, making it impossible to generate rules that depend on relationships between different object types.

#### 2. **LLM Prompt Content** ✅ **CONFIRMED**

**Location:** `external/poe-world/prompts/synthesizer.py` (Lines 867-966, 5-50)

**Finding:** Prompts only include binary "touching" relationships and basic object properties. No distance, range, or spatial relationship information is provided:

```python
# Only shows interactions as "touching" relationships:
"Interaction -- player object (id = 0) is touching ladder object (id = 2)"

# No spatial context like:
# "player object (id = 0) is at distance 3 from zombie object (id = 1)"
# "zombie object (id = 1) is within attack range of player object (id = 0)"
```

**Impact:** LLMs cannot generate rules that depend on spatial relationships beyond simple touching, as they lack the necessary spatial context.

#### 3. **Generated Rule Format** ✅ **CONFIRMED**

**Location:** `external/poe-world/prompts/synthesizer.py` (Lines 5-50)

**Finding:** The rule generation template is designed to operate on a single object type only:

```python
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # Only gets objects of one type
    for {obj_type}_obj in {obj_type}_objs: # Only iterates over one object type
```

**Impact:** Rules cannot reference multiple object types or express cross-object-type spatial relationships.

#### 4. **Interaction Representation** ✅ **CONFIRMED**

**Location:** `external/poe-world/classes/helper.py` (Lines 309-330, 549-600)

**Finding:** Interactions only capture binary "touching" relationships through collision detection. No distance, range, or other spatial information is included:

```python
class Interaction:
    def __init__(self, obj1: "Obj", obj2: "Obj") -> None:
        self.obj1 = obj1
        self.obj2 = obj2
    
    def str_w_id(self):
        return f"Interaction -- {self.obj1.str_w_id()} is touching {self.obj2.str_w_id()}"

# touches() method only checks for collision/overlap
def touches(self, other, touch_side=-1, touch_percent=0):
    return pygame.Rect(self.x - 1, self.y - 1, self.w + 2, self.h + 2).colliderect(
        pygame.Rect(other.x, other.y, other.w, other.h))
```

**Impact:** The system cannot represent spatial relationships like "within attack range" or "at distance X" that are essential for realistic game mechanics.

#### 5. **Synthesizer Types** ✅ **CONFIRMED**

**Location:** `external/poe-world/learners/synthesizer.py` (Lines 161-1717)

**Finding:** All available synthesizer types are designed for single object type processing. Even `PlayerInteractionSynthesizer` only handles player-specific interactions, not general spatial relationships:

```python
# Available synthesizer types (all single-object-type focused):
- ActionSynthesizer
- MultiTimestepActionSynthesizer  
- MultiTimestepMomentumSynthesizer
- MultiTimestepSizeChangeSynthesizer
- MultiTimestepStatusChangeSynthesizer
- PassiveMovementSynthesizer
- PassiveCreationSynthesizer
- VelocitySynthesizer
- PlayerInteractionSynthesizer  # Only handles player interactions
- SnappingSynthesizer
- ConstraintsSynthesizer
- RestartSynthesizer
```

**Impact:** No synthesizer exists that can handle cross-object-type spatial relationships or generate rules that depend on spatial context between different object types.

#### 6. **Causal Relationship Processing** ✅ **CONFIRMED**

**Location:** `external/poe-world/learners/synthesizer.py` (Lines 105-140)

**Finding:** The `_get_natural_language_effects` method only considers changes to objects of the target type, with no attribution of causality:

```python
def _get_natural_language_effects(self, x):
    input_target_obj_list = self.objects_selector(x.input_state)  # Only target type objects
    output_target_obj_list = self.objects_selector(x.output_state)  # Only target type objects
    
    effects = []
    for o in output_target_obj_list:
        if o.deleted == 1:
            effects.append(f'The {o.str_w_id()} is deleted')
        elif o.id not in input_ids:
            effects.append(f'A new {o.obj_type} object is created at (x={o.x},y={o.y})')
        else:
            effects.append(f'The {o.str_w_id()} sets x-axis velocity to {"%+d" % (o.velocity_x)}')
            effects.append(f'The {o.str_w_id()} sets y-axis velocity to {"%+d" % (o.velocity_y)}')
```

**Impact:** Synthesizers cannot distinguish between self-inflicted and externally-caused changes, leading to incorrect rule generation where effects are attributed to the wrong causal agent.

### Additional Findings

#### Available but Unused Spatial Information

The codebase **does contain** spatial information that could be utilized but is not provided to synthesizers:

```python
# Lines 450-500: Obj class properties
@property
def center_x(self): return self.x + self.w // 2
@property  
def center_y(self): return self.y + self.h // 2

# Lines 520-530: Distance calculation
def distance_to(self, other):
    return np.sqrt((self.center_x - other.center_x)**2 + (self.center_y - other.center_y)**2)
```

This spatial information exists but is **not integrated into the synthesis process**.

### Implications

#### 1. **Fundamental Architectural Limitation**

The spatial relationship limitation is not a bug but a **fundamental design choice** in PoE-World's architecture. The system prioritizes:
- **Modularity**: Each object type has its own synthesizer
- **Interpretability**: Rules are object-type-specific and easy to understand
- **Scalability**: Adding new object types doesn't affect existing synthesizers

However, this comes at the cost of **expressiveness** for complex spatial interactions.

#### 2. **Impact on Game Mechanic Modeling**

This limitation severely restricts PoE-World's ability to model realistic game mechanics:

- **Targeted Attacks**: Cannot express "attack affects only nearby enemies"
- **Area-of-Effect Abilities**: Cannot express "ability affects all objects within radius R"
- **Line-of-Sight Effects**: Cannot express "projectile travels until hitting obstacle"
- **Conditional Interactions**: Cannot express "interact only with closest object"
- **Spatial Constraints**: Cannot express "cannot move through walls"
- **Causal Relationships**: Cannot express "zombie's attack damages player" or "player's action affects zombie"
- **External Effects**: Cannot distinguish between self-inflicted and externally-caused changes

#### 3. **Scope of Required Changes**

Addressing this limitation would require **significant architectural changes**:

1. **Enhanced Interaction System**: Modify `Interaction` class to include distance, range, and spatial relationship information
2. **Cross-Object-Type Synthesizers**: Create new synthesizer types that can handle multiple object types
3. **Modified Prompt Templates**: Update prompts to include spatial context and relationship information
4. **Rule Generation Framework**: Extend the rule generation system to support multi-object-type rules
5. **Spatial Context Integration**: Integrate existing spatial information (center_x, center_y, distance_to) into the synthesis process

#### 4. **Trade-offs and Risks**

Any solution would need to balance:

- **Expressiveness vs. Interpretability**: More complex rules may be harder to understand
- **Modularity vs. Integration**: Cross-object-type rules may reduce the modular design
- **Performance vs. Functionality**: Additional spatial calculations may impact performance
- **Backward Compatibility**: Changes should not break existing functionality

### Conclusion

The investigation confirms that PoE-World has a **fundamental architectural limitation** in expressing complex spatial relationships between different object types. This limitation is by design and reflects the system's prioritization of modularity and interpretability over expressiveness for complex spatial interactions.

**Status:** ✅ **CONFIRMED** - All six investigation questions answered affirmatively.

**Recommendation:** This limitation should be addressed through architectural enhancements that maintain the system's core strengths while adding support for cross-object-type spatial relationships.

**Priority:** High - This affects the system's ability to model realistic game environments.

**Effort:** High - Would require significant architectural changes across multiple components.

**Risk:** Medium - Changes could affect existing functionality and model interpretability.

---

## Addendum: Why These Limitations Don't Affect Atari Environments

### Context: PoE-World's Original Design Target

PoE-World was originally designed for and tested on **Atari environments** like Pong and Montezuma's Revenge. The spatial and causal relationship limitations described in this issue are **not significant problems** for these simpler game environments, which explains why they weren't discovered during initial development.

### Atari Game Characteristics

#### 1. **Simple Object Types and Interactions**

**Pong Example:**
- **Objects:** Ball, Paddle (left), Paddle (right), Walls
- **Interactions:** Ball bounces off paddles and walls
- **Actions:** Move paddle up/down

**Montezuma's Revenge Example:**
- **Objects:** Player, Platforms, Ladders, Keys, Doors, Enemies
- **Interactions:** Player touches platforms/ladders, player collects keys, player opens doors
- **Actions:** Move left/right, climb up/down, jump

#### 2. **Binary Interaction Model is Sufficient**

In Atari games, the `touches()` interaction model (Lines 549-600 in `external/poe-world/classes/helper.py`) works perfectly:

```python
# Pong: Ball touching paddle = bounce
if ball.touches(paddle):
    ball.velocity_x = -ball.velocity_x  # Reverse direction

# Montezuma: Player touching ladder = can climb
if player.touches(ladder):
    player.can_climb = True
```

The simple collision detection provided by `touches()` is sufficient for these environments.

#### 3. **Limited Spatial Complexity**

**Pong:**
- 2D space with simple rectangular objects
- No "attack ranges" or "area effects"
- Ball either touches paddle or doesn't

**Montezuma's Revenge:**
- Platform-based movement
- Simple proximity detection (touching)
- No complex spatial relationships like "within attack range"

#### 4. **Simple Causal Relationships**

In Atari games, causality is often straightforward and self-contained:

**Pong:**
- Ball movement → Ball touches paddle → Ball bounces
- Each object type can understand its own role in the interaction

**Montezuma's Revenge:**
- Player touches key → Key disappears → Player can open door
- Player touches enemy → Player loses health

The causal chain is simple enough that each object type can understand its own effects without needing to see other object types.

### Why Object-Type Isolation Works for Atari

#### 1. **Self-Contained Object Behavior**

In Pong, each paddle only needs to know:
- How to move up/down
- How to bounce the ball when touched

It doesn't need to know about the other paddle or the scoring system.

#### 2. **Simple State Changes**

In Montezuma's Revenge, when a player touches a key:
- The key disappears (key's own behavior)
- The player gets a key (player's own behavior)

Each object type can handle its own state changes independently.

#### 3. **No Complex Spatial Reasoning Required**

Atari games typically don't require:
- "Attack affects only nearby enemies"
- "Ability affects all objects within radius R"
- "Move toward closest target"

### Why Crafter Reveals the Limitations

#### 1. **Complex Object Types and Relationships**

**Crafter Example:**
- **Objects:** Player, Zombies, Cows, Trees, Rocks, Tools, Weapons, etc.
- **Complex Interactions:** 
  - Player attacks zombie (spatial range matters)
  - Zombie chases player (distance-based behavior)
  - Player mines rock with tool (tool type matters)
  - Player plants seeds (location matters)

#### 2. **Spatial Relationships Matter**

```python
# Crafter: This type of rule cannot be expressed in PoE-World
def zombie_behavior(obj_list: ObjList, action: str) -> ObjList:
    player = obj_list.get_objs_by_obj_type('player')[0]
    zombies = obj_list.get_objs_by_obj_type('zombie')
    
    for zombie in zombies:
        distance = zombie.distance(player.position)
        if distance <= 8:  # Chase range
            zombie.move_toward(player)
        if distance <= 1:  # Attack range
            zombie.attack(player)
```

#### 3. **Complex Causal Chains**

**Crafter Example:**
- Player uses sword → Sword affects zombies within range → Zombies take damage
- Zombie attacks player → Player takes damage → Player health decreases

This requires understanding that:
1. One object's action affects another object type
2. The effect depends on spatial proximity
3. The causal agent is different from the affected object

### The Architectural Mismatch

PoE-World's architecture was designed for:
- **Simple collision-based interactions** ✅ (works for Atari)
- **Object-type isolation** ✅ (works for Atari)
- **Self-contained object behavior** ✅ (works for Atari)

But modern game environments like Crafter require:
- **Complex spatial relationships** ❌ (not supported)
- **Cross-object-type causality** ❌ (not supported)
- **Distance-based interactions** ❌ (not supported)

### Code Evidence from PoE-World

The codebase shows it was designed for simpler interactions:

```python
# Lines 549-600: external/poe-world/classes/helper.py
def touches(self, other, touch_side=-1, touch_percent=0):
    # Only checks collision/overlap - perfect for Atari games
    return pygame.Rect(self.x - 1, self.y - 1, self.w + 2, self.h + 2).colliderect(
        pygame.Rect(other.x, other.y, other.w, other.h))

# Lines 44-50: external/poe-world/learners/synthesizer.py
self.objects_selector = ObjTypeObjSelector(self.obj_type)  # Only sees own type
self.interactions_selector = ObjTypeInteractionSelector(self.obj_type)  # Only own interactions
```

This isolation works fine when interactions are simple and self-contained.

### Conclusion

The spatial and causal relationship limitations in PoE-World are **not significant for Atari environments** because:

1. **Atari games use simple collision-based interactions** that work well with the `touches()` model
2. **Object behavior is self-contained** - each object type can understand its own role without seeing other types
3. **Spatial relationships are binary** - objects either touch or don't touch
4. **Causal chains are simple** - effects are typically direct and don't require cross-object-type understanding

However, these same limitations become **critical problems** in more complex environments like Crafter, where:
- Spatial relationships matter (attack ranges, chase distances)
- Causal chains are complex (one object's action affects another)
- Object behavior depends on relationships with other object types

This explains why PoE-World worked well for its original Atari use cases but reveals fundamental limitations when applied to environments that require more complex spatial and causal reasoning. The limitations were always present in the architecture but only became apparent when the system was applied to environments that require more complex spatial and causal reasoning.
