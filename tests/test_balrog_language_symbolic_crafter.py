from distant_sunburn.balrog_components import CrafterEnvironmentConfig
from distant_sunburn.balrog_language_symbolic_crafter import (
    LanguageSymbolicCrafter,
)
from distant_sunburn.crafter_symbolic import SymbolicObservation


def test_language_symbolic_crafter():
    # Create a basic config
    crafter_config = CrafterEnvironmentConfig(
        area=(64, 64),
        view=(9, 9),
        size=(256, 256),
        reward=True,
        seed=42,  # Fixed seed for reproducibility
    )

    # Create the environment
    env = LanguageSymbolicCrafter(crafter_config)

    # Test reset
    reset_exp = env.reset()
    assert isinstance(reset_exp.info, SymbolicObservation)

    # Test step
    action = "Move West"
    exp = env.step(action)
    assert isinstance(exp.info, SymbolicObservation)

    # Test action validity check
    valid_action = env.check_action_validity("Move West")
    assert valid_action == "Move West"
