import json
from typing import Any


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
