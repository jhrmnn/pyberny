# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
import sys
import csv

from pkg_resources import resource_string


def get_property(idx, name):
    if isinstance(idx, str):
        return species_data[idx][name]
    try:
        return next(row[name] for row in species_data if row['number'] == idx)
    except StopIteration:
        raise KeyError('No species with number "{}"'.format(idx))


def _get_species_data():
    csv_lines = resource_string(__name__, 'species-data.csv').split(b'\n')
    if sys.version_info[0] > 2:
        csv_lines = [l.decode() for l in csv_lines]
    reader = csv.DictReader(csv_lines, quoting=csv.QUOTE_NONNUMERIC)
    species_data = {row['symbol']: row for row in reader}
    return species_data


species_data = _get_species_data()
