"""Pytest configuration file with shared fixtures and utilities."""

import os
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Return the absolute path to the project root directory.
    
    Returns:
        Path: Path object pointing to the project root
    """
    return project_root


@pytest.fixture(scope="session")
def config_dir(project_root_path: Path) -> Path:
    """Return the path to the config directory.
    
    Args:
        project_root_path: Path to the project root
        
    Returns:
        Path: Path to the config directory
    """
    return project_root_path / "configs"
