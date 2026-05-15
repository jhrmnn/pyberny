# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
import csv
from importlib.resources import files

__all__ = ()


def get_property(idx, name):
    if isinstance(idx, str):
        try:
            value = species_data[idx][name]
        except KeyError:
            raise KeyError(f'No species with symbol "{idx}"') from None
    else:
        try:
            value = next(
                row[name] for row in species_data.values() if row['number'] == idx
            )
        except StopIteration:
            raise KeyError(f'No species with number "{idx}"') from None
    if value == '':
        raise KeyError(f'No "{name}" data for species "{idx}"')
    return value


def _get_species_data():
    csv_text = files(__package__).joinpath('species-data.csv').read_text()
    reader = csv.DictReader(csv_text.splitlines(), quoting=csv.QUOTE_NONNUMERIC)
    species_data = {row['symbol']: row for row in reader}
    return species_data


species_data = _get_species_data()
