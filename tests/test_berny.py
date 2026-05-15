import numpy as np

from berny import Berny, BernyParams, Geometry


def water():
    return Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )


def test_berny_params_defaults():
    p = BernyParams()
    assert p.gradientmax == 0.45e-3
    assert p.trust == 0.3
    assert p.dihedral is True


def test_berny_param_override():
    b = Berny(water(), trust=0.5, gradientrms=1e-4)
    assert b.trust == 0.5
    assert b._state.params.gradientrms == 1e-4
    # Untouched defaults still come from BernyParams.
    assert b._state.params.gradientmax == 0.45e-3


def test_berny_unknown_param_rejected():
    # BernyParams has fixed fields; typos no longer silently end up in a
    # params dict that nobody reads.
    try:
        Berny(water(), trust=0.5, gradeintrms=1e-4)
    except TypeError as e:
        assert 'gradeintrms' in str(e)
    else:
        raise AssertionError('expected TypeError for unknown param')


def test_berny_debug_restart_roundtrip():
    geom = water()
    b = Berny(geom, debug=True)
    next(b)
    state = b.send((0.0, np.zeros((3, 3))))
    assert isinstance(state, dict)
    assert 'geom' in state and 'params' in state
    b2 = Berny(geom, restart=state)
    assert b2.trust == b._state.trust
