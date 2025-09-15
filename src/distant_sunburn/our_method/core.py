from torch.utils import data
from distant_sunburn.poe_world.core import ExpertFunction
from typing import Protocol, TypeVar, Any
from pathlib import Path
import cloudpickle
from typing_extensions import Self, Generic
from ..typing_utils import implements
import inspect
from typing import Type
from dataclasses import dataclass

SymbolicStateT = TypeVar("SymbolicStateT")
SymbolicStateT_contra = TypeVar("SymbolicStateT_contra", contravariant=True)


class LawProtocol(Protocol[SymbolicStateT_contra]):
    def precondition(
        self, current_state: SymbolicStateT_contra, action: Any
    ) -> bool: ...

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None: ...

    @property
    def __source_code__(self) -> str: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "LawProtocol[SymbolicStateT_contra]": ...

    @property
    def __name__(self) -> str: ...


class BaseLawProtocol(Protocol[SymbolicStateT_contra]):
    def precondition(
        self, current_state: SymbolicStateT_contra, action: Any
    ) -> bool: ...

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None: ...


class LawFunctionWrapper(Generic[SymbolicStateT_contra]):
    def __init__(self, law: BaseLawProtocol[SymbolicStateT_contra], source_code: str):
        self.law = law
        self.source_code = source_code

    def precondition(self, current_state: SymbolicStateT_contra, action: Any) -> bool:
        return self.law.precondition(current_state, action)

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None:
        self.law.effect(current_state, action)

    def save(self, path: str | Path) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LawFunctionWrapper[SymbolicStateT_contra]":
        """
        Deserializes an ExpertFunctionWrapper instance from a file using cloudpickle.

        Args:
            file_path: The path of the file to load.

        Returns:
            A new instance of ExpertFunctionWrapper.

        Raises:
            TypeError: If the unpickled object is not an instance of this class.
        """
        if not isinstance(path, Path):
            path = Path(path)
        with path.open("rb") as f:
            instance = cloudpickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(
                f"File '{path}' did not contain an instance of "
                f"{cls.__name__}, but of {type(instance).__name__}."
            )

        return instance

    @classmethod
    def from_non_runtime_created(
        cls, law: BaseLawProtocol[SymbolicStateT_contra]
    ) -> "LawFunctionWrapper[SymbolicStateT_contra]":
        return cls(law, inspect.getsource(law.__class__))

    @property
    def __name__(self) -> str:
        return self.law.__class__.__name__

    @property
    def __source_code__(self) -> str:
        return self.source_code


implements(LawProtocol)(LawFunctionWrapper)


@dataclass
class WeightedLaw(Generic[SymbolicStateT_contra]):
    law: LawProtocol[SymbolicStateT_contra]
    weight: float
    is_fitted: bool = False


SymbolicStateT = TypeVar("SymbolicStateT")


class WorldModelProtocol(Protocol[SymbolicStateT]):
    def sample_next_state(
        self, current_state: SymbolicStateT, action: Any
    ) -> SymbolicStateT: ...
    def evaluate_log_probability(
        self, state: SymbolicStateT, action: Any, next_state: SymbolicStateT
    ) -> float: ...
    def with_new_laws(
        self, new_laws: list[WeightedLaw[SymbolicStateT]]
    ) -> "WorldModelProtocol[SymbolicStateT]": ...
    @property
    def laws(self) -> list[WeightedLaw[SymbolicStateT]]: ...


@dataclass
class SymbolicTransition(Generic[SymbolicStateT]):
    prev_state: SymbolicStateT
    action: Any
    next_state: SymbolicStateT


class LawOptimizerProtocol(Protocol[SymbolicStateT]):
    def fit(
        self,
        laws: list[LawProtocol[SymbolicStateT]],
        transitions: list[SymbolicTransition[SymbolicStateT]],
    ) -> list[WeightedLaw[SymbolicStateT]]: ...
