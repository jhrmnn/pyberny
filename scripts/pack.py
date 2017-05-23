#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import hashlib
import io
from glob import glob
from itertools import takewhile
import tarfile
from base64 import b64encode


filename = 'berny'


def strip():
    with open(filename) as f:
        lines = takewhile(lambda l: l != '# ==>\n', f.readlines())
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)


def pack():
    strip()
    h = hashlib.new('md5')
    with io.BytesIO() as ftar:
        archive = tarfile.open(mode='w|gz', fileobj=ftar)
        for path in sorted(glob('bernylib/*.py')):
            archive.add(path)
            with open(path, 'rb') as f:
                h.update(f.read())
        archive.close()
        archive = ftar.getvalue()
    version = h.hexdigest()
    with open(filename, 'a') as f:
        f.write('# ==>\n')
        f.write('# version: {}\n'.format(version))
        f.write('# archive: {}\n'.format(b64encode(archive).decode()))
        f.write('# <==\n')


if __name__ == '__main__':
    pack()
