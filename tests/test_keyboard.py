import numpy as np
from train_futo_model.util.keyboard.futo import (
    get_key_by_position,
    _KEY_WIDTH,
    _KEY_HEIGHT,
)

# FIXME make clear that tests are futo specific


def test_exact_key_centers():
    """Test that exact key centers return the correct key."""
    # Test some keys from each row
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 0.5, _KEY_HEIGHT * 3.5]))
        == "Q"
    )
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 2.5, _KEY_HEIGHT * 3.5]))
        == "E"
    )
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 1.0, _KEY_HEIGHT * 2.5]))
        == "A"
    )
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 7.0, _KEY_HEIGHT * 2.5]))
        == "J"
    )
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 4.0, _KEY_HEIGHT * 1.5]))
        == "C"
    )
    assert (
        get_key_by_position(np.array([_KEY_WIDTH * 8.0, _KEY_HEIGHT * 1.5]))
        == "M"
    )


def test_points_between_keys():
    """Test points that are between keys return the closest key."""
    # Between Q and W (should be closer to Q)
    pos = np.array([_KEY_WIDTH * 0.9, _KEY_HEIGHT * 3.5])
    assert get_key_by_position(pos) == "Q"

    # Between Q and A (should be closer to Q)
    pos = np.array([_KEY_WIDTH * 0.5, _KEY_HEIGHT * 3.0])
    assert get_key_by_position(pos) == "Q"

    # Between J and K (should be closer to K)
    pos = np.array([_KEY_WIDTH * 7.6, _KEY_HEIGHT * 2.5])
    assert get_key_by_position(pos) == "K"


def test_edge_cases():
    """Test points at the edges and corners of the keyboard."""
    # Top-left corner (Q)
    assert get_key_by_position(np.array([0, _KEY_HEIGHT * 4.0])) == "Q"
    # Top-right corner (P)
    assert get_key_by_position(np.array([1.0, _KEY_HEIGHT * 4.0])) == "P"
    # Bottom-left corner (Z)
    assert get_key_by_position(np.array([_KEY_WIDTH * 2.0, 0])) == "Z"
    # Bottom-right corner (M)
    assert get_key_by_position(np.array([1.0, 0])) == "M"


def test_points_between_equidistant_keys():
    """Test points that are exactly between two keys."""
    # Exactly between Q and W
    pos = np.array([_KEY_WIDTH * 1.0, _KEY_HEIGHT * 3.5])
    # Should return the key that comes first in the min() comparison (implementation dependent)
    result = get_key_by_position(pos)
    assert result in ["Q", "W"]

    # Exactly between A and Z (diagonally)
    pos = np.array([_KEY_WIDTH * 1.0, _KEY_HEIGHT * 2.0])
    result = get_key_by_position(pos)
    assert result in ["A", "Z"]


def test_points_outside_keyboard():
    """Test points outside the keyboard area."""
    # Far above the keyboard
    assert get_key_by_position(np.array([0.5, 2.0])) == "Q"
    # Far below the keyboard
    assert get_key_by_position(np.array([0.5, -1.0])) == "Z"
    # Far left of the keyboard
    assert get_key_by_position(np.array([-1.0, _KEY_HEIGHT * 2.5])) == "A"
    # Far right of the keyboard
    assert get_key_by_position(np.array([2.0, _KEY_HEIGHT * 2.5])) == "L"
