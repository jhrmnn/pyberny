# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import print_function
import re


class Logger(object):
    def __init__(self, verbosity=0, out=None, regex=None):
        self.verbosity = verbosity
        self.out = out
        self.n = 0
        self.regex = regex

    def __call__(self, msg, **kwargs):
        level = kwargs.get('level', 0)
        if level < -self.verbosity:
            return
        if self.regex is not None and not re.search(self.regex, msg):
            return
        print(self.n, msg, file=self.out)
