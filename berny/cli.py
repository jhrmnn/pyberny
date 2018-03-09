# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import sys
import json
import pickle
from argparse import ArgumentParser
from socket import socket, AF_INET, SOCK_STREAM
from contextlib import contextmanager

from berny import Berny, geomlib


@contextmanager
def berny_unpickled(berny=None):
    picklefile = 'berny.pickle'
    if not berny:
        with open(picklefile, 'rb') as f:
            berny = pickle.load(f)
    try:
        yield berny
    except:
        raise
    with open(picklefile, 'wb') as f:
        pickle.dump(berny, f)


def handler(berny, f):
    energy = float(next(f))
    gradients = [[float(x) for x in l.split()] for l in f if l.strip()]
    berny.send(energy, gradients)
    return next(berny)


def get_berny(args):
    geom = geomlib.load(sys.stdin, args.format)
    if args.paramfile:
        with open(args.paramfile) as f:
            params = json.load(f)
    else:
        params = None
    berny = Berny(geom, **params)
    next(berny)
    return berny


def init(args):
    berny = get_berny(args)
    berny.geom_format = args.format
    with berny_unpickled(berny) as berny:
        pass


def server(args):
    berny = get_berny(args)
    host, port = args.socket
    server = socket(AF_INET, SOCK_STREAM)
    server.bind((host, int(port)))
    server.listen(0)
    while True:
        sock, addr = server.accept()
        f = sock.makefile('r+')
        geom = handler(berny, f)
        if geom:
            f.write(geom.dumps(args.format))
            f.flush()
        f.close()
        sock.close()
        if not geom:
            break


def driver():
    try:
        with berny_unpickled() as berny:
            geom = handler(berny, sys.stdin)
    except FileNotFoundError:
        sys.stderr.write('error: No pickled berny, run with --init first?\n')
        sys.exit(1)
    if not geom:
        sys.exit(0)
    geom.dump(sys.stdout, berny.geom_format)
    sys.exit(10)


def main():
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('--init', action='store_true', help='Initialize Berny optimizer.')
    arg('-f', '--format', choices=['xyz', 'aims'], default='xyz', help='Format of geometry')
    arg('-s', '--socket', nargs=2, metavar=('host', 'port'), help='Listen on given address')
    arg('paramfile', nargs='?', help='Optional optimization parameters as JSON')
    args = parser.parse_args()
    if args.init:
        init(args)
    elif args.socket:
        server(args)
    else:
        driver()
