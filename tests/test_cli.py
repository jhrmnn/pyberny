import json
import pickle
import sys
from argparse import Namespace
from io import StringIO

import pytest

from berny import Berny
from berny.cli import berny_unpickled, driver, get_berny, handler, init, main
from berny.geomlib import Geometry

WATER_XYZ = '3\n\nO 0 0 0\nH 0.96 0 0\nH 0 0.96 0\n'


def test_handler_converges_returns_none():
    # Until 2026, handler called berny.send(energy, gradients) — two
    # positional args to a method that takes a single tuple — so the CLI
    # crashed on every step. It also let StopIteration propagate on
    # convergence instead of returning None to the driver.
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )
    berny = Berny(geom)
    next(berny)
    f = StringIO('0.0\n0 0 0\n0 0 0\n0 0 0\n')
    result = handler(berny, f)
    assert result is None
    assert berny.converged


def test_handler_returns_geometry_when_not_converged(monkeypatch):
    # If the optimizer isn't done, handler should hand back the next
    # geometry — drive a single non-converging step.
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    )
    berny = Berny(geom)
    next(berny)
    f = StringIO('0.0\n1 0 0\n-1 0 0\n0 1 0\n')
    result = handler(berny, f)
    assert isinstance(result, Geometry)
    assert not berny.converged


def test_get_berny_reads_stdin_xyz(monkeypatch):
    monkeypatch.setattr(sys, 'stdin', StringIO(WATER_XYZ))
    args = Namespace(format='xyz', paramfile=None)
    berny = get_berny(args)
    # The function primes the generator with one `next` call already.
    assert berny._n == 1


def test_get_berny_reads_paramfile(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'stdin', StringIO(WATER_XYZ))
    paramfile = tmp_path / 'params.json'
    paramfile.write_text(json.dumps({'maxsteps': 5}))
    args = Namespace(format='xyz', paramfile=str(paramfile))
    berny = get_berny(args)
    assert berny._maxsteps == 5


def test_init_writes_pickle_in_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'stdin', StringIO(WATER_XYZ))
    args = Namespace(format='xyz', paramfile=None)
    init(args)
    pkl = tmp_path / 'berny.pickle'
    assert pkl.exists()
    with open(pkl, 'rb') as f:
        restored = pickle.load(f)
    assert isinstance(restored, Berny)
    assert restored.geom_format == 'xyz'


def test_berny_unpickled_round_trips_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Prime a Berny and pickle it manually so we can verify the context
    # manager loads it, lets us mutate it, and re-pickles the mutation.
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )
    berny = Berny(geom)
    next(berny)
    with open('berny.pickle', 'wb') as f:
        pickle.dump(berny, f)
    with berny_unpickled() as b:
        b.geom_format = 'aims'  # arbitrary mutation
    with open('berny.pickle', 'rb') as f:
        restored = pickle.load(f)
    assert restored.geom_format == 'aims'


def test_driver_missing_pickle_exits_with_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'stdin', StringIO('0.0\n0 0 0\n0 0 0\n0 0 0\n'))
    with pytest.raises(SystemExit) as exc:
        driver()
    assert exc.value.code == 1
    assert 'No pickled berny' in capsys.readouterr().err


def test_driver_unconverged_writes_geom_and_exits_10(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    )
    berny = Berny(geom)
    berny.geom_format = 'xyz'
    next(berny)
    with open('berny.pickle', 'wb') as f:
        pickle.dump(berny, f)
    monkeypatch.setattr(sys, 'stdin', StringIO('0.0\n1 0 0\n-1 0 0\n0 1 0\n'))
    with pytest.raises(SystemExit) as exc:
        driver()
    assert exc.value.code == 10
    out = capsys.readouterr().out
    assert out.splitlines()[0].strip().isdigit()  # XYZ atom count line


def test_driver_converged_exits_zero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )
    berny = Berny(geom)
    berny.geom_format = 'xyz'
    next(berny)
    with open('berny.pickle', 'wb') as f:
        pickle.dump(berny, f)
    monkeypatch.setattr(sys, 'stdin', StringIO('0.0\n0 0 0\n0 0 0\n0 0 0\n'))
    with pytest.raises(SystemExit) as exc:
        driver()
    assert exc.value.code == 0


def test_main_init_dispatches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['berny', '--init'])
    monkeypatch.setattr(sys, 'stdin', StringIO(WATER_XYZ))
    main()
    assert (tmp_path / 'berny.pickle').exists()


def test_main_driver_dispatches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )
    berny = Berny(geom)
    berny.geom_format = 'xyz'
    next(berny)
    with open('berny.pickle', 'wb') as f:
        pickle.dump(berny, f)
    monkeypatch.setattr(sys, 'argv', ['berny'])
    monkeypatch.setattr(sys, 'stdin', StringIO('0.0\n0 0 0\n0 0 0\n0 0 0\n'))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
