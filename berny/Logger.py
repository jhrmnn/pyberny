# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import print_function
import re


class Logger(object):
    """
    Manages logging of information.

    :param int verbosity: verbosity level
    :param file out: argument passed to ``print(..., file)``
    :param str regex: log only messages that match regex
    """

    def __init__(self, verbosity=None, out=None, regex=None):
        self.verbosity = verbosity or 0
        self.out = out
        self.n = 0
        self.regex = regex

    def __call__(self, msg, level=0):
        """
        Log a message.

        :param str msg: the message
        :param int level: this is compared against the verbosity level
        """
        if level < -self.verbosity:
            return
        if self.regex is not None and not re.search(self.regex, msg):
            return
        print(self.n, msg, file=self.out)
