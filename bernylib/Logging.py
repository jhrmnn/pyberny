import sys


class Logger(object):
    berny = None

    def __call__(self, *lines):
        for line in lines:
            sys.stderr.write('{} {}\n'.format(Logger.berny.nsteps, line))

    def register(self, berny):
        Logger.berny = berny


info = Logger()
error = Logger()
