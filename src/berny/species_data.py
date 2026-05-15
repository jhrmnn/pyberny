# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
import csv
from importlib.resources import files

__all__ = ()

# Names PySCF / ASE / NWChem use for basis-function-only centers ("ghost atoms").
# They have no covalent or vdW radius and zero mass, so InternalCoords never
# bonds to them and they don't contribute to angles or dihedrals.
_GHOST_ALIASES = {'ghost', 'x', 'bq'}


def _is_ghost(symbol):
    if not isinstance(symbol, str):
        return False
    name = symbol.lstrip('-').lower()
    return name in _GHOST_ALIASES


_GHOST_ROW = {
    'number': 0.0,
    'name': 'ghost',
    'symbol': 'Ghost',
    'covalent_radius': 0.0,
    'vdw_radius': 0.0,
    'mass': 0.0,
}


def get_property(idx, name):
    if isinstance(idx, str):
        if _is_ghost(idx):
            return _GHOST_ROW[name]
        try:
            value = species_data[idx][name]
        except KeyError:
            raise KeyError(f'No species with symbol {idx!r}') from None
    else:
        try:
            value = next(
                row[name] for row in species_data.values() if row['number'] == idx
            )
        except StopIteration:
            raise KeyError(f'No species with number {idx!r}') from None
    if value == '':
        raise KeyError(f'No {name!r} data for species {idx!r}')
    return value


def _get_species_data():
    csv_text = files(__package__).joinpath('species-data.csv').read_text()
    reader = csv.DictReader(csv_text.splitlines(), quoting=csv.QUOTE_NONNUMERIC)
    species_data = {row['symbol']: row for row in reader}
    return species_data


species_data = _get_species_data()
