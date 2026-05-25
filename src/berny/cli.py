# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import argparse
import json
import pickle
import sys
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from socket import AF_INET, SOCK_STREAM, socket
from typing import IO, Optional

from berny import Berny, geomlib

__all__ = ()


@contextmanager
def berny_unpickled(berny: Optional[Berny] = None) -> Iterator[Berny]:
    picklefile = 'berny.pickle'
    if not berny:
        with open(picklefile, 'rb') as f:
            berny = pickle.load(f)
    try:
        yield berny
    except Exception:
        raise
    with open(picklefile, 'wb') as f:
        pickle.dump(berny, f)


def handler(berny: Berny, f: IO[str]) -> Optional[geomlib.Geometry]:
    energy = float(next(f))
    gradients = [[float(x) for x in l.split()] for l in f if l.strip()]
    import numpy as np

    berny.send((energy, np.array(gradients)))
    try:
        return next(berny)
    except StopIteration:
        return None


def get_berny(args: argparse.Namespace) -> Berny:
    geom = geomlib.load(sys.stdin, args.format)
    if args.paramfile:
        with open(args.paramfile) as f:
            params = json.load(f)
    else:
        params = {}
    berny = Berny(geom, **params)
    next(berny)
    return berny


def init(args: argparse.Namespace) -> None:
    berny = get_berny(args)
    # ``geom_format`` is a runtime-only attribute used by ``driver`` to know how
    # to re-emit geometries; it isn't part of ``Berny``'s static interface.
    berny.geom_format = args.format  # type: ignore[attr-defined]
    with berny_unpickled(berny):
        pass


def server(args: argparse.Namespace) -> None:
    berny = get_berny(args)
    host, port = args.socket
    server = socket(AF_INET, SOCK_STREAM)
    server.bind((host, int(port)))
    server.listen(0)
    while True:
        sock, _addr = server.accept()
        # ``socket.makefile`` only accepts read-only or write-only text modes
        # ("r"/"w"); use separate handles for the two directions.
        reader = sock.makefile('r')
        writer = sock.makefile('w')
        geom = handler(berny, reader)
        if geom:
            writer.write(geom.dumps(args.format))
            writer.flush()
        reader.close()
        writer.close()
        sock.close()
        if not geom:
            break


def driver() -> None:
    try:
        with berny_unpickled() as berny:
            geom = handler(berny, sys.stdin)
    except FileNotFoundError:
        sys.stderr.write('error: No pickled berny, run with --init first?\n')
        sys.exit(1)
    if not geom:
        sys.exit(0)
    geom.dump(sys.stdout, berny.geom_format)  # type: ignore[attr-defined]
    sys.exit(10)


def main() -> None:
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('--init', action='store_true', help='Initialize Berny optimizer.')
    arg(
        '-f',
        '--format',
        choices=['xyz', 'aims'],
        default='xyz',
        help='Format of geometry',
    )
    arg(
        '-s',
        '--socket',
        nargs=2,
        metavar=('host', 'port'),
        help='Listen on given address',
    )
    arg('paramfile', nargs='?', help='Optional optimization parameters as JSON')
    args = parser.parse_args()
    if args.init:
        init(args)
    elif args.socket:
        server(args)
    else:
        driver()
