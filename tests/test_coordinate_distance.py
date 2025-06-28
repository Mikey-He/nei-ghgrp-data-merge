import pytest
import os
import sys
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math

# Stub pandas
if 'pandas' not in sys.modules:
    pandas_stub = types.ModuleType('pandas')
    pandas_stub.isna = lambda x: x != x
    sys.modules['pandas'] = pandas_stub

# Stub numpy with math functions
if 'numpy' not in sys.modules:
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.radians = math.radians
    numpy_stub.sin = math.sin
    numpy_stub.cos = math.cos
    numpy_stub.arcsin = math.asin
    numpy_stub.sqrt = math.sqrt
    numpy_stub.isscalar = lambda x: isinstance(x, (int, float, complex))
    numpy_stub.bool_ = bool
    sys.modules['numpy'] = numpy_stub

# Stub fuzzywuzzy.fuzz
if 'fuzzywuzzy' not in sys.modules:
    fuzzywuzzy_stub = types.ModuleType('fuzzywuzzy')
    fuzz_module = types.ModuleType('fuzzywuzzy.fuzz')
    fuzzywuzzy_stub.fuzz = fuzz_module
    sys.modules['fuzzywuzzy'] = fuzzywuzzy_stub
    sys.modules['fuzzywuzzy.fuzz'] = fuzz_module

# Stub tqdm.tqdm
if 'tqdm' not in sys.modules:
    tqdm_stub = types.ModuleType('tqdm')
    def dummy_tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
    tqdm_stub.tqdm = dummy_tqdm
    sys.modules['tqdm'] = tqdm_stub

import importlib
merge = importlib.import_module('merge_nei_ghgrp')
NEIGHGRPMerger = merge.NEIGHGRPMerger


def test_identical_coordinates_zero_distance():
    merger = NEIGHGRPMerger()
    dist = merger.calculate_coordinate_distance(10.0, 20.0, 10.0, 20.0)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_one_degree_latitude_distance():
    merger = NEIGHGRPMerger()
    dist = merger.calculate_coordinate_distance(0.0, 0.0, 1.0, 0.0)
    # Distance between latitudes differs by 1 degree is about 111.2 km
    assert dist == pytest.approx(111.19, rel=0.01)
