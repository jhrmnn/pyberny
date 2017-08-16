# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import print_function


class Logger(object):
    def __init__(self, verbosity=0, out=None):
        self.verbosity = verbosity
        self.out = out
        self.n = 0

    def __call__(self, msg, **kwargs):
        level = kwargs.get('level', 0)
        if level < -self.verbosity:
            return
        print(self.n, msg, file=self.out)
