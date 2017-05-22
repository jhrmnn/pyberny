# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import sys


class Logger(object):
    berny = None

    def __call__(self, *items):
        sys.stderr.write('{} {}\n'.format(Logger.berny.nsteps, ' '.join(map(str, items))))

    def register(self, berny):
        Logger.berny = berny


info = Logger()
error = Logger()
