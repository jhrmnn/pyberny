import pytest

from berny.solvers import _mopac_keyword_line


def test_mopac_neutral_singlet():
    assert _mopac_keyword_line('PM7', 0, 1) == 'PM7 1SCF GRADIENTS'


def test_mopac_cation():
    assert _mopac_keyword_line('PM7', 1, 1) == 'PM7 1SCF GRADIENTS CHARGE=1'


def test_mopac_dianion():
    assert _mopac_keyword_line('PM7', -2, 1) == 'PM7 1SCF GRADIENTS CHARGE=-2'


def test_mopac_doublet_cation():
    assert _mopac_keyword_line('PM7', 1, 2) == 'PM7 1SCF GRADIENTS CHARGE=1 DOUBLET UHF'


def test_mopac_unsupported_multiplicity():
    with pytest.raises(ValueError, match='unsupported MOPAC multiplicity'):
        _mopac_keyword_line('PM7', 0, 99)
