# Auto-generated tests for /mnt/data/generators.py
import pytest
import data.generators

def test_func_generate_1d_convex_callable():
    assert callable(getattr(data.generators, 'generate_1d_convex', None))

def test_func_generate_1d_non_convex_callable():
    assert callable(getattr(data.generators, 'generate_1d_non_convex', None))

def test_func_generate_2d_convex_callable():
    assert callable(getattr(data.generators, 'generate_2d_convex', None))

def test_func_generate_2d_non_convex_callable():
    assert callable(getattr(data.generators, 'generate_2d_non_convex', None))
