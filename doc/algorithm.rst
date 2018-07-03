Algorithm
=========

Redundant internal coordinates
------------------------------

1. All bonds shorter than 1.3 times the sum of covalent radii are
   created.
2. If there are unconnected fragments, all bonds between unconnected
   fragments shorter than the sum of van der Waals radii plus *d* are
   created, with *d* starting at 0 and increasing by 1 angstrom, until
   all fragments are connected.
3. All angles greater than 45° are created.
4. All dihedrals with 1–2–3, 2–3–4 angles both greater than 45° are
   created. If one of the angles is zero, so that three atoms lie on a
   line, they are used as a new base for a dihedral. This process is
   recursively repeated.
5. In the case of a crystal, just the internal coordinate closest to the
   original unit cell is retained from all periodic images.

Generalized inverse
-------------------

The Wilson matrix :math:`\mathbf B`, which relates differences in the internal
redundant coordinates to differences in the Cartesian coordinates, is in
general non-square and non-invertible. A generalized inverse of a matrix
is obtained by taking its singular value decomposition and inverting
only the nonzero singular values. For invertible matrices, this is
equivalent to an ordinary inverse. In practice, the zero values are in
fact nonzero but several orders of magnitude smaller than the true
nonzero values.
