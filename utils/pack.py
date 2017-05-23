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


def unpack():
    from base64 import b64decode
    import tarfile
    import io
    with open(__file__) as f:
        for line in f:
            if line == '# ==>\n':
                break
        else:
            raise RuntimeError('No packed lib')
        version = next(f)[:-1].split(None, 2)[-1]
        libpath = '.berny-{}'.format(version)
        if not os.path.exists(libpath):
            archive = b64decode(next(f)[:-1].split(None, 2)[-1])
            with io.BytesIO(archive) as ftar:
                tar = tarfile.open(mode='r|gz', fileobj=ftar)
                tar.extractall(libpath)
    return libpath


if __name__ == '__main__':
    pack()
