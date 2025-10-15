from onelife.json_utils import (
    flatten_json_to_pathmap,
    compute_patch_intersection_over_union,
)
import jsonpatch


def test_complex_manual_nested_structure():
    """Test with a manually constructed complex nested JSON structure."""
    # Manually construct a complex nested structure
    data = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "profile": {
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "coordinates": [40.7128, -74.0060],
                },
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                    "languages": ["en", "es", "fr"],
                },
            },
        },
        "orders": [
            {
                "id": "order-1",
                "items": [
                    {"name": "laptop", "price": 999.99},
                    {"name": "mouse", "price": 29.99},
                ],
                "status": "shipped",
            },
            {"id": "order-2", "items": [], "status": "pending"},
        ],
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "version": 1.0,
            "active": True,
        },
    }

    result = flatten_json_to_pathmap(data)

    # Manually assert the expected flattened structure
    expected = {
        "/user/id": 123,
        "/user/name": "John Doe",
        "/user/profile/age": 30,
        "/user/profile/address/street": "123 Main St",
        "/user/profile/address/city": "New York",
        "/user/profile/address/coordinates/0": 40.7128,
        "/user/profile/address/coordinates/1": -74.0060,
        "/user/profile/preferences/theme": "dark",
        "/user/profile/preferences/notifications": True,
        "/user/profile/preferences/languages/0": "en",
        "/user/profile/preferences/languages/1": "es",
        "/user/profile/preferences/languages/2": "fr",
        "/orders/0/id": "order-1",
        "/orders/0/items/0/name": "laptop",
        "/orders/0/items/0/price": 999.99,
        "/orders/0/items/1/name": "mouse",
        "/orders/0/items/1/price": 29.99,
        "/orders/0/status": "shipped",
        "/orders/1/id": "order-2",
        "/orders/1/items": [],
        "/orders/1/status": "pending",
        "/metadata/created_at": "2024-01-01T00:00:00Z",
        "/metadata/version": 1.0,
        "/metadata/active": True,
    }

    assert result == expected


def test_empty_structures():
    """Test edge cases with empty dictionaries and lists."""
    # Test with empty dict - gets recorded as terminal value with empty path
    data1 = {}
    result1 = flatten_json_to_pathmap(data1)
    assert result1 == {"": {}}

    # Test with empty list - gets recorded as terminal value with empty path
    data2 = []
    result2 = flatten_json_to_pathmap(data2)
    assert result2 == {"": []}

    # Test with dict containing empty dict
    data3 = {"empty_dict": {}}
    result3 = flatten_json_to_pathmap(data3)
    assert result3 == {"/empty_dict": {}}

    # Test with dict containing empty list
    data4 = {"empty_list": []}
    result4 = flatten_json_to_pathmap(data4)
    assert result4 == {"/empty_list": []}

    # Test with list containing empty dict
    data5 = [{}]
    result5 = flatten_json_to_pathmap(data5)
    assert result5 == {"/0": {}}

    # Test with list containing empty list
    data6 = [[]]
    result6 = flatten_json_to_pathmap(data6)
    assert result6 == {"/0": []}


def test_primitive_values():
    """Test with primitive values (strings, numbers, booleans, null)."""
    data = {
        "string": "hello world",
        "integer": 42,
        "float": 3.14159,
        "boolean_true": True,
        "boolean_false": False,
        "null_value": None,
    }

    result = flatten_json_to_pathmap(data)

    expected = {
        "/string": "hello world",
        "/integer": 42,
        "/float": 3.14159,
        "/boolean_true": True,
        "/boolean_false": False,
        "/null_value": None,
    }

    assert result == expected


def test_mixed_types_and_edge_cases():
    """Test with mixed data types and various edge cases."""
    data = {
        "nested": {
            "list_with_mixed_types": [1, "string", True, None, {"nested": "value"}],
            "dict_with_numeric_keys": {"0": "zero", "1": "one", "2": "two"},
        },
        "list_of_dicts": [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
        "special_chars": {
            "key/with/slashes": "value",
            "key.with.dots": "another_value",
        },
    }

    result = flatten_json_to_pathmap(data)

    expected = {
        "/nested/list_with_mixed_types/0": 1,
        "/nested/list_with_mixed_types/1": "string",
        "/nested/list_with_mixed_types/2": True,
        "/nested/list_with_mixed_types/3": None,
        "/nested/list_with_mixed_types/4/nested": "value",
        "/nested/dict_with_numeric_keys/0": "zero",
        "/nested/dict_with_numeric_keys/1": "one",
        "/nested/dict_with_numeric_keys/2": "two",
        "/list_of_dicts/0/a": 1,
        "/list_of_dicts/0/b": 2,
        "/list_of_dicts/1/c": 3,
        "/list_of_dicts/1/d": 4,
        "/special_chars/key/with/slashes": "value",
        "/special_chars/key.with.dots": "another_value",
    }

    assert result == expected


def test_single_level_structures():
    """Test with single-level structures (no nesting)."""
    # Single dict
    data1 = {"a": 1, "b": 2, "c": 3}
    result1 = flatten_json_to_pathmap(data1)
    expected1 = {"/a": 1, "/b": 2, "/c": 3}
    assert result1 == expected1

    # Single list
    data2 = ["x", "y", "z"]
    result2 = flatten_json_to_pathmap(data2)
    expected2 = {"/0": "x", "/1": "y", "/2": "z"}
    assert result2 == expected2


def test_edit_distance_normalization():
    """Test that edit distance normalization works correctly with JsonPatch."""
    # Create a complex JSON structure with known number of attributes
    source_json = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",
            "profile": {
                "age": 30,
                "city": "New York",
                "country": "USA",
                "preferences": {"theme": "dark", "language": "en"},
            },
        },
        "settings": {"notifications": True, "privacy": "public"},
    }

    # Create a modified version with 3 specific changes
    destination_json = {
        "user": {
            "id": 456,  # Changed: 123 -> 456
            "name": "John Doe",
            "email": "john.doe@example.com",  # Changed: john@example.com -> john.doe@example.com
            "profile": {
                "age": 30,
                "city": "San Francisco",  # Changed: New York -> San Francisco
                "country": "USA",
                "preferences": {"theme": "dark", "language": "en"},
            },
        },
        "settings": {"notifications": True, "privacy": "public"},
    }

    # Flatten both JSON structures to get all attributes
    source_flat = flatten_json_to_pathmap(source_json)
    destination_flat = flatten_json_to_pathmap(destination_json)

    # Calculate total number of attributes (should be the same for both)
    total_attributes = len(source_flat)
    assert total_attributes == len(
        destination_flat
    ), "Both JSONs should have same number of attributes"

    # Create JsonPatch to find the differences
    patch = jsonpatch.make_patch(source_json, destination_json)
    num_changes = len(patch.patch)

    # Calculate normalized edit distance
    normalized_distance = num_changes / total_attributes

    # Verify our expectations
    # We made 3 changes: id, email, and city
    expected_changes = 3
    expected_normalized_distance = expected_changes / total_attributes

    assert (
        num_changes == expected_changes
    ), f"Expected {expected_changes} changes, got {num_changes}"
    assert (
        normalized_distance == expected_normalized_distance
    ), f"Expected normalized distance {expected_normalized_distance}, got {normalized_distance}"

    # Verify the total number of attributes is what we expect
    # Manual count: /user/id, /user/name, /user/email, /user/profile/age, /user/profile/city,
    # /user/profile/country, /user/profile/preferences/theme, /user/profile/preferences/language,
    # /settings/notifications, /settings/privacy = 10 attributes
    expected_total_attributes = 10
    assert (
        total_attributes == expected_total_attributes
    ), f"Expected {expected_total_attributes} total attributes, got {total_attributes}"

    # Verify the normalized distance is 30% (3 changes out of 10 attributes)
    expected_percentage = 0.3
    assert (
        abs(normalized_distance - expected_percentage) < 0.001
    ), f"Expected ~30% normalized distance, got {normalized_distance:.1%}"

    # Print the patch for verification
    print(f"Total attributes: {total_attributes}")
    print(f"Number of changes: {num_changes}")
    print(f"Normalized distance: {normalized_distance:.1%}")
    print("JsonPatch operations:")
    for op in patch.patch:
        print(f"  {op['op']} {op['path']}: {op.get('value', 'N/A')}")


def test_path_consistency_with_jsonpatch():
    """Test that flatten_json_to_pathmap generates paths consistent with JsonPatch."""
    # Create a complex JSON structure
    json_data = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "profile": {
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "coordinates": [40.7128, -74.0060],
                },
                "preferences": {"theme": "dark", "languages": ["en", "es", "fr"]},
            },
        },
        "orders": [
            {
                "id": "order-1",
                "items": [
                    {"name": "laptop", "price": 999.99},
                    {"name": "mouse", "price": 29.99},
                ],
            },
            {"id": "order-2", "items": []},
        ],
    }

    # Flatten the JSON to get all paths
    flattened = flatten_json_to_pathmap(json_data)
    flattened_paths = set(flattened.keys())

    # Create a modified version to generate JsonPatch operations
    modified_json = {
        "user": {
            "id": 456,  # Changed
            "name": "John Doe",
            "profile": {
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "San Francisco",  # Changed
                    "coordinates": [40.7128, -74.0060],
                },
                "preferences": {
                    "theme": "light",  # Changed
                    "languages": ["en", "es", "fr"],
                },
            },
        },
        "orders": [
            {
                "id": "order-1",
                "items": [
                    {"name": "laptop", "price": 1099.99},  # Changed
                    {"name": "mouse", "price": 29.99},
                ],
            },
            {"id": "order-2", "items": []},
        ],
    }

    # Generate JsonPatch to see what paths it uses
    patch = jsonpatch.make_patch(json_data, modified_json)
    jsonpatch_paths = set()

    # Extract all paths from JsonPatch operations
    for operation in patch.patch:
        jsonpatch_paths.add(operation["path"])

    # Also test with a patch that adds/removes elements to see more path patterns
    json_with_additions = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",  # Added
            "profile": {
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "coordinates": [40.7128, -74.0060],
                },
                "preferences": {"theme": "dark", "languages": ["en", "es", "fr"]},
            },
        },
        "orders": [
            {
                "id": "order-1",
                "items": [
                    {"name": "laptop", "price": 999.99},
                    {"name": "mouse", "price": 29.99},
                ],
            }
            # Removed order-2
        ],
    }

    patch_with_additions = jsonpatch.make_patch(json_data, json_with_additions)
    for operation in patch_with_additions.patch:
        jsonpatch_paths.add(operation["path"])

    # Verify that all JsonPatch paths for existing elements exist in our flattened paths
    # Note: JsonPatch may reference paths that don't exist in the original structure
    # (e.g., when adding new elements or removing existing ones)
    print(f"JsonPatch operations:")
    for operation in patch.patch:
        print(f"  {operation['op']} {operation['path']}")
    for operation in patch_with_additions.patch:
        print(f"  {operation['op']} {operation['path']}")

    # For replace operations, the paths should exist in our flattened structure
    replace_paths = set()
    for operation in patch.patch + patch_with_additions.patch:
        if operation["op"] == "replace":
            replace_paths.add(operation["path"])

    missing_replace_paths = replace_paths - flattened_paths
    assert (
        not missing_replace_paths
    ), f"JsonPatch replace operations used paths not found in flattened structure: {missing_replace_paths}"

    # Verify that our flattened paths are valid JsonPatch paths
    # (This is a weaker test since we might have paths that JsonPatch doesn't use)
    print(f"Flattened paths count: {len(flattened_paths)}")
    print(f"JsonPatch paths count: {len(jsonpatch_paths)}")
    print(f"JsonPatch paths: {sorted(jsonpatch_paths)}")

    # Test specific path patterns that JsonPatch uses for existing elements
    expected_path_patterns = [
        "/user/id",
        "/user/profile/address/city",
        "/user/profile/preferences/theme",
        "/orders/0/items/0/price",
    ]

    for expected_path in expected_path_patterns:
        assert (
            expected_path in flattened_paths
        ), f"Expected path {expected_path} not found in flattened paths"

    # Test that array indices are handled correctly
    array_paths = [
        path
        for path in flattened_paths
        if "/" in path and any(part.isdigit() for part in path.split("/"))
    ]
    assert len(array_paths) > 0, "Should have array index paths"

    # Verify specific array paths exist
    expected_array_paths = [
        "/user/profile/address/coordinates/0",
        "/user/profile/address/coordinates/1",
        "/user/profile/preferences/languages/0",
        "/user/profile/preferences/languages/1",
        "/user/profile/preferences/languages/2",
        "/orders/0/items/0/name",
        "/orders/0/items/0/price",
        "/orders/0/items/1/name",
        "/orders/0/items/1/price",
        "/orders/1/items",
    ]

    for expected_array_path in expected_array_paths:
        assert (
            expected_array_path in flattened_paths
        ), f"Expected array path {expected_array_path} not found"

    print("✓ All JsonPatch paths are consistent with flattened paths")
    print("✓ Array indexing follows JsonPatch conventions")
    print("✓ Nested object paths follow JsonPatch conventions")


def test_patch_iou_identical_patches():
    """Test that identical patches have IoU = 1.0."""
    original = {"name": "Alice", "age": 30}

    # Create two identical modifications
    modified1 = {"name": "Alice", "age": 31}
    modified2 = {"name": "Alice", "age": 31}

    patch1 = jsonpatch.make_patch(original, modified1)
    patch2 = jsonpatch.make_patch(original, modified2)

    iou = compute_patch_intersection_over_union(patch1, patch2)
    assert iou == 1.0, f"Identical patches should have IoU=1.0, got {iou}"


def test_patch_iou_completely_different_patches():
    """Test that completely different patches have IoU = 0.0."""
    original = {"name": "Alice", "age": 30, "email": "alice@example.com"}

    # Create patches that modify different fields
    modified1 = {
        "name": "Alice",
        "age": 31,
        "email": "alice@example.com",
    }  # Changes age
    modified2 = {
        "name": "Alice",
        "age": 30,
        "email": "alice.new@example.com",
    }  # Changes email

    patch1 = jsonpatch.make_patch(original, modified1)
    patch2 = jsonpatch.make_patch(original, modified2)

    iou = compute_patch_intersection_over_union(patch1, patch2)
    assert iou == 0.0, f"Completely different patches should have IoU=0.0, got {iou}"


def test_patch_iou_partial_overlap():
    """Test that partially overlapping patches have IoU between 0 and 1."""
    original = {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "preferences": {"theme": "dark", "notifications": True},
    }

    # Patch 1: Changes age and theme
    modified1 = {
        "name": "Alice",
        "age": 31,  # Changed
        "email": "alice@example.com",
        "preferences": {"theme": "light", "notifications": True},  # Changed
    }

    # Patch 2: Changes age and notifications (overlaps on age)
    modified2 = {
        "name": "Alice",
        "age": 31,  # Same change as patch1
        "email": "alice@example.com",
        "preferences": {"theme": "dark", "notifications": False},  # Different change
    }

    patch1 = jsonpatch.make_patch(original, modified1)
    patch2 = jsonpatch.make_patch(original, modified2)

    iou = compute_patch_intersection_over_union(patch1, patch2)
    # Should have 1 common operation (age change) out of 3 total unique operations
    expected_iou = 1.0 / 3.0
    assert (
        abs(iou - expected_iou) < 0.001
    ), f"Expected IoU≈{expected_iou:.3f}, got {iou}"


def test_patch_iou_with_empty_patches():
    """Test IoU calculation with empty patches."""
    original = {"name": "Alice", "age": 30}
    modified = {"name": "Alice", "age": 31}

    patch_with_changes = jsonpatch.make_patch(original, modified)
    empty_patch = jsonpatch.make_patch(original, original)  # No changes

    # One patch empty, one with changes
    iou_with_empty = compute_patch_intersection_over_union(
        patch_with_changes,
        empty_patch,
    )
    assert (
        iou_with_empty == 0.0
    ), f"Patch with empty patch should have IoU=0.0, got {iou_with_empty}"

    # Both patches empty
    iou_both_empty = compute_patch_intersection_over_union(empty_patch, empty_patch)
    assert (
        iou_both_empty == 1.0
    ), f"Both empty patches should have IoU=1.0, got {iou_both_empty}"


def test_patch_iou_with_object_value_triggers_unhashable_dict_error():
    """Reproduce bug: IOU raises TypeError when operation value is a dict.

    The JsonPatch diff can legally include object (dict) values for add/replace
    operations. Our current IOU implementation attempts to put these values into
    a set via a NamedTuple, which triggers `TypeError: unhashable type: 'dict'`.
    """
    original = {"user": {"profile": {}}}
    modified1 = {"user": {"profile": {"settings": {"theme": "dark"}}}}
    modified2 = {"user": {"profile": {"settings": {"theme": "light"}}}}

    patch1 = jsonpatch.make_patch(original, modified1)
    patch2 = jsonpatch.make_patch(original, modified2)

    # This call currently raises: TypeError: unhashable type: 'dict'
    # The test is meant to reproduce the bug; it will fail until the bug is fixed.
    compute_patch_intersection_over_union(patch1, patch2)
