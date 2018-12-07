Algorithm
=========

The optimization algorithm implemented in Berny loosely follows the
"standard method" (SM) described in the appendix of [BirkholzTCA16]_. What
follows is a summary of that method, more detailed specification when
necessary, and description of any deviations.

.. todo:: Make the algorithm `fully conform <https://github.com/azag0/pyberny/issues/29>`_ to the SM.

Sketch of the algorithm
-----------------------

1. Form `redundant internal coordinates`_.
2. Form a diagonal Hessian guess [SwartIJQC06]_.
3. Obtain energy and Cartesian gradients for the current geometry.
4. Form the Wilson **B** matrix and its `generalized inverse`_.
5. Update the Hessian using the BFGS scheme.
6. Update trust region (Eq. 5.1.6 of [Fletcher00]_).
7. Perform linear search (Gaussian `website <http://gaussian.com/opt/>`__,
   section Examples, "If a minimum is sought..."). (Not in the SM.)
8. Project to a nonredundant subspace [PengJCC96]_.
9. Perform a quadratic RFO step [BanerjeeJPC85]_.
10. Transform back to Cartesian coordinates [PengJCC96]_.
11. If convergence is not reached (criteria from the SM), go to 3.

Redundant internal coordinates
------------------------------

1. All bonds shorter than 1.3 times the sum of covalent radii are created
   [PengJCC96]_.
2. If there are unconnected fragments, all bonds between unconnected fragments
   shorter than the sum of van der Waals radii plus *d* are created, with *d*
   starting at 0 and increasing by 1 angstrom, until all fragments are
   connected (custom scheme by JH).
3. All angles greater than 45° are created.
4. All dihedrals with 1–2–3, 2–3–4 angles both greater than 45° are created. If
   one of the angles is zero, so that three atoms lie on a line, they are used
   as a new base for a dihedral. This process is recursively repeated
   [PengJCC96]_.
5. In the case of a crystal, just the internal coordinate closest to the
   original unit cell is retained from all its periodic images.

.. todo:: Implement `linear bends <https://github.com/azag0/pyberny/issues/30>`_.

Generalized inverse
-------------------

The Wilson **B** matrix, which relates differences in the internal redundant
coordinates to differences in the Cartesian coordinates, is in general
non-square and non-invertible. Its generalized inverse is obtained from the
pseudoinverse of :math:`\mathbf B\mathbf B^\mathrm T` (singular), which is in
turn obtained via singular value decomposition and inversion of only the
nonzero singular values. For invertible matrices, this procedure is equivalent
to an ordinary inverse. In practice, the zero values are in fact nonzero but
several orders of magnitude smaller than the true nonzero values.

References
----------

.. [BirkholzTCA16] Birkholz, A. B. & Schlegel, H. B. Exploration of some
   refinements to geometry optimization methods. Theor. Chem. Acc. 135, (2016).
   DOI: `10.1007/s00214-016-1847-3
   <http://dx.doi.org/10.1007/s00214-016-1847-3>`_
.. [PengJCC96] Peng, C., Ayala, P. Y., Schlegel, H. B. & Frisch, M. J. Using
   redundant internal coordinates to optimize equilibrium geometries and
   transition states. J. Comput. Chem. 17, 49–56 (1996). DOI:
   `10.1002/(SICI)1096-987X(19960115)17:1\<49::AID-JCC5\>3.0.CO;2-0
   <https://doi.org/10.1002/(SICI)1096-987X(19960115)17:1\<49::AID-JCC5\>3.0.CO;2-0>`_
.. [SwartIJQC06] Swart, M. & Bickelhaupt, F. M. Optimization of Strong and Weak
   Coordinates. Int. J. Quantum Chem. 106, 2536–2544 (2006). DOI:
   `10.1002/qua.21049 <https://doi.org/10.1002/qua.21049>`_
.. [Fletcher00] Fletcher, R. Practical Methods of Optimization. (Wiley, 2000).
   URL:
   https://www.wiley.com/en-us/Practical+Methods+of+Optimization%2C+2nd+Edition-p-9780471494638
.. [BanerjeeJPC85] Banerjee, A., Adams, N. & Simons, J. Search for Stationary
   Points on Surfaces. J. Phys. Chem. 57, 52–57 (1985). DOI:
   `10.1021/j100247a015 <https://doi.org/10.1021/j100247a015>`_
