import sys


class Logger(object):
    berny = None

    def __call__(self, *items):
        sys.stderr.write('{} {}\n'.format(Logger.berny.nsteps, ' '.join(map(str, items))))

    def register(self, berny):
        Logger.berny = berny


info = Logger()
error = Logger()
