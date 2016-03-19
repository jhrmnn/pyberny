class Logger:
    def __init__(self):
        self.berny = None

    def __call__(self, *lines):
        for line in lines:
            print('{} {}'.format(self.berny.nsteps, line))

    def register(self, berny):
        self.berny = berny


info = Logger()
