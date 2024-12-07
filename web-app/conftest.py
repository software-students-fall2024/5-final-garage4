"""
This module contains shared fixtures for pytest, such as the test client for the Flask app.
"""

import pytest
from app import app


@pytest.fixture
def test_client():
    """
    Creates and returns a test client for the Flask app to be used in tests.
    """
    with app.test_client() as client:
        yield client
