# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
import csv
from importlib.resources import files

__all__ = ()

#: A single value from the species table — either a number (``QUOTE_NONNUMERIC``
#: in the CSV reader) or a string for textual columns such as ``name``.
SpeciesValue = float | str

# Names PySCF / ASE / NWChem use for basis-function-only centers ("ghost atoms").
# They have no covalent or vdW radius and zero mass, so InternalCoords never
# bonds to them and they don't contribute to angles or dihedrals.
_GHOST_ALIASES = {'ghost', 'x', 'bq'}


def _is_ghost(symbol: str) -> bool:
    return symbol.lstrip('-').lower() in _GHOST_ALIASES


_GHOST_ROW: dict[str, SpeciesValue] = {
    'number': 0.0,
    'name': 'ghost',
    'symbol': 'Ghost',
    'covalent_radius': 0.0,
    'vdw_radius': 0.0,
    'mass': 0.0,
}


def get_property(idx: str | float, name: str) -> SpeciesValue:
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
    # '' is the missing-data sentinel; a numeric 0 is valid data, so the
    # `not value` simplification PLC1901 suggests would be wrong here.
    if value == '':  # noqa: PLC1901
        raise KeyError(f'No {name!r} data for species {idx!r}')
    return value


def _get_species_data() -> dict[str, dict[str, SpeciesValue]]:
    csv_text = files(__package__).joinpath('species-data.csv').read_text()
    reader = csv.DictReader(csv_text.splitlines(), quoting=csv.QUOTE_NONNUMERIC)
    return {str(row['symbol']): dict(row) for row in reader}


species_data: dict[str, dict[str, SpeciesValue]] = _get_species_data()
