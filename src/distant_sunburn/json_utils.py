import json
from typing import Any, NamedTuple
import jsonpatch


def flatten_json_to_pathmap(data: dict | list) -> dict[str, Any]:
    """
    Flattens a nested dictionary or list into a dictionary of JSON-Patch style paths
    using an iterative, stack-based approach.
    """
    flat_paths: dict[str, Any] = {}

    # The stack will hold tuples of (path, object_to_process)
    # Start with the root object and an empty path string.
    stack = [("", data)]

    while stack:
        path, obj = stack.pop()

        if isinstance(obj, dict):
            if not obj:
                # We will record an empty dict as a terminal value
                flat_paths[path] = obj
                continue
            # Add child items to the stack to be processed later.
            # We iterate in reverse to maintain a more natural (or consistent)
            # processing order since a stack is LIFO.
            for key, value in reversed(list(obj.items())):
                stack.append((f"{path}/{key}", value))

        elif isinstance(obj, list):
            if not obj:
                # We will record an empty list as a terminal value
                flat_paths[path] = obj
                continue
            # Add child items to the stack in reverse order.
            for i, value in reversed(list(enumerate(obj))):
                stack.append((f"{path}/{i}", value))

        else:
            # This is a terminal value, so we record its path.
            # The path will be non-empty because we start with "" for the root.
            flat_paths[path] = obj

    return flat_paths


class PatchOperation(NamedTuple):
    """Represents a JSON Patch operation for equality comparison.

    This named tuple captures the essential information needed to compare
    patch operations for intersection over union calculations.
    """

    path: str
    operation: str
    value: Any


def compute_patch_intersection_over_union(
    patch1: jsonpatch.JsonPatch, patch2: jsonpatch.JsonPatch, source_json: dict | list
) -> float:
    """
    Compute the intersection over union (IoU) of two JSON patches.

    This function compares two JsonPatch objects by converting their operations
    into comparable PatchOperation objects and computing the IoU metric.

    Args:
        patch1: First JsonPatch object
        patch2: Second JsonPatch object
        source_json: The original JSON structure that both patches were applied to.
                    This is used to get the total number of attributes for normalization.
                    Note: The JSON should terminate in primitive values (strings, numbers,
                    booleans, null) at all leaf nodes, which should always be the case
                    for valid JSON structures.

    Returns:
        float: IoU value between 0.0 and 1.0, where:
               - 1.0 means the patches are identical
               - 0.0 means the patches have no operations in common
               - Values in between indicate partial overlap

    Example:
        >>> source = {"a": 1, "b": 2}
        >>> patch1 = jsonpatch.make_patch(source, {"a": 3, "b": 2})
        >>> patch2 = jsonpatch.make_patch(source, {"a": 3, "b": 4})
        >>> iou = compute_patch_intersection_over_union(patch1, patch2, source)
        >>> print(f"IoU: {iou:.2f}")  # Should be 0.5 (1 common operation out of 2 total)
    """
    # Convert patches to sets of PatchOperation objects for comparison
    operations1 = set()
    for op in patch1.patch:
        operations1.add(
            PatchOperation(
                path=op["path"], operation=op["op"], value=op.get("value", None)
            )
        )

    operations2 = set()
    for op in patch2.patch:
        operations2.add(
            PatchOperation(
                path=op["path"], operation=op["op"], value=op.get("value", None)
            )
        )

    # Compute intersection and union
    intersection = operations1 & operations2
    union = operations1 | operations2

    # Calculate IoU
    if len(union) == 0:
        return 1.0  # Both patches are empty, so they're identical

    iou = len(intersection) / len(union)
    return iou
