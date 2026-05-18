Standard method — full reference
================================

.. role:: sm
.. role:: pb

This page documents the *standard method* (SM) for geometry optimization in
sufficient detail to reimplement it from scratch. The SM was defined in the
appendix of [BirkholzTCA16]_ as a reference algorithm against which various
refinements are compared. The bulk of the formulas below are taken verbatim
from that appendix; numbered equations match the original paper so that they
can be cross-checked. Sections that the SM defers to other papers (RFO
derivation, B-matrix derivatives, trust-region theory, Bofill update,
recursive dihedrals) are filled in here with explicit formulas and cited
sources.

How to read this page
---------------------

* Every block of formulas is self-contained; together they specify the
  algorithm completely.
* :sm:`Pale-yellow callouts` flag places where the SM, or a paper it cites,
  leaves something unspecified.
* :pb:`Pale-blue callouts` flag where PyBerny deviates from, or has not yet
  implemented, what the SM prescribes. The relevant source location is
  given when useful.

.. note::

   This page is a specification of the SM. For a higher-level sketch of
   what PyBerny actually does (with links from each step into the code),
   see :doc:`algorithm`.


Notation
--------

* :math:`N` — number of atoms; :math:`n_\text{crt}=3N` Cartesian degrees of
  freedom.
* :math:`x\in\mathbb R^{n_\text{crt}}` — Cartesian coordinates;
  :math:`x_k\in\mathbb R^3` the position of atom :math:`k`.
* :math:`q\in\mathbb R^{n_\text{ric}}` — redundant internal coordinates;
  :math:`q_i` an individual stretch, bend, or torsion.
* :math:`n_\text{act}=n_\text{crt}-6` (5 for linear molecules) — number of
  non-redundant internal degrees of freedom.
* :math:`g_x = \partial E/\partial x`, :math:`g_q = \partial E/\partial q`
  — gradients in the two coordinate systems.
* :math:`H_x`, :math:`H_q` — corresponding Hessian matrices.
* For Hessian-update formulas, the shorthand of the SM is used:

  .. math::

     s = \Delta q,\qquad y = \Delta g_q,\qquad z = y - H\,s,

  where :math:`\Delta q = q^{(n+1)}-q^{(n)}` and analogously for the gradient.


Overview of one optimization step
---------------------------------

Given the current Cartesian geometry :math:`x^{(n)}`, energy :math:`E^{(n)}`,
and Cartesian gradient :math:`g_x^{(n)}`:

1. Build the Wilson B-matrix at :math:`x^{(n)}` (§\ `B-matrix`_).
2. Compute the generalised inverse :math:`B^-` (§\ `Generalised inverse`_).
3. Transform the gradient: :math:`g_q = (B^-)^\mathrm T g_x` (Eq. 20).
4. Update the internal-coordinate Hessian :math:`H` from
   :math:`(s,y) = (q^{(n)}-q^{(n-1)}, g_q^{(n)}-g_q^{(n-1)})`
   (§\ `Hessian update`_). On the first step, take a diagonal guess
   (§\ `Initial Hessian`_).
5. Update the trust radius :math:`\tau` from the previous predicted vs.
   actual energy change (§\ `Trust region`_).
6. Project gradient and Hessian into the non-redundant active space
   (§\ `Projection`_).
7. Solve the rational-function (RFO) sub-problem for a step
   :math:`\Delta q^\mathrm{r}` (§\ `RFO step`_); enforce
   :math:`\|\Delta q^\mathrm{r}\|\le\tau` by a sphere-restricted
   minimisation if necessary.
8. Back-transform :math:`\Delta q^\mathrm{r}` to Cartesians iteratively
   (§\ `Back-transformation`_).
9. Test convergence (§\ `Convergence`_). If not converged, set
   :math:`n\leftarrow n+1` and repeat.


Coordinate definitions
----------------------

Bond stretch
^^^^^^^^^^^^

For atoms :math:`i,j` with positions :math:`x_i,x_j\in\mathbb R^3`:

.. math::

   r_{ij} \;=\; \|x_i - x_j\|.

B-matrix rows (gradients of :math:`r_{ij}` w.r.t. :math:`x_i,x_j`):

.. math::

   \frac{\partial r_{ij}}{\partial x_i} = \frac{x_i-x_j}{r_{ij}},\qquad
   \frac{\partial r_{ij}}{\partial x_j} = -\frac{x_i-x_j}{r_{ij}}.

Bond angle
^^^^^^^^^^

For atoms :math:`i,j,k` with central atom :math:`j`, define
:math:`v_1 = x_i-x_j`, :math:`v_2 = x_k-x_j`:

.. math::

   \cos\theta_{ijk} \;=\; \frac{v_1\cdot v_2}{\|v_1\|\,\|v_2\|},\qquad
   0 \le \theta_{ijk} \le \pi.

B-matrix rows (Wilson [Schlegel1984]_; with
:math:`\hat e_1=v_1/\|v_1\|`, :math:`\hat e_2=v_2/\|v_2\|`):

.. math::

   \frac{\partial\theta}{\partial x_i} &=
       \frac{\cos\theta\,\hat e_1 - \hat e_2}{\|v_1\|\sin\theta}, \\
   \frac{\partial\theta}{\partial x_k} &=
       \frac{\cos\theta\,\hat e_2 - \hat e_1}{\|v_2\|\sin\theta}, \\
   \frac{\partial\theta}{\partial x_j} &= -\frac{\partial\theta}{\partial x_i}
       - \frac{\partial\theta}{\partial x_k}.

:sm:`Note (unspecified in [BirkholzTCA16]_):` the formulas above become
singular at :math:`\theta=0` or :math:`\pi`. The SM handles
:math:`\theta\to\pi` by switching to the linear-bend coordinate system
(see `Linear bends`_); for :math:`\theta\to 0` no prescription is given.

Torsion (dihedral)
^^^^^^^^^^^^^^^^^^

For atoms :math:`i,j,k,l` with central bond :math:`j\!-\!k`, write
:math:`v_1=x_i-x_j`, :math:`v_2=x_l-x_k`, :math:`w=x_k-x_j`,
:math:`\hat e_w=w/\|w\|`. The components of :math:`v_1,v_2` perpendicular
to :math:`w` are :math:`a_1 = v_1-(v_1\!\cdot\hat e_w)\hat e_w`,
:math:`a_2 = v_2-(v_2\!\cdot\hat e_w)\hat e_w`. Then

.. math::

   \cos\varphi_{ijkl} = \frac{a_1\cdot a_2}{\|a_1\|\,\|a_2\|},\qquad
   \mathrm{sgn}\,\varphi = \mathrm{sgn}\,\det[v_2,v_1,w],\qquad
   -\pi < \varphi_{ijkl} \le \pi.

Compact B-matrix rows, with
:math:`A=(v_1\!\cdot\hat e_w)/\|w\|`,
:math:`B=(v_2\!\cdot\hat e_w)/\|w\|`:

.. math::

   \frac{\partial\varphi}{\partial x_i} &=
       \cot\varphi\,\frac{a_1}{\|a_1\|^2} - \frac{a_2}{\|a_1\|\|a_2\|\sin\varphi}, \\
   \frac{\partial\varphi}{\partial x_l} &=
       \cot\varphi\,\frac{a_2}{\|a_2\|^2} - \frac{a_1}{\|a_1\|\|a_2\|\sin\varphi}, \\
   \frac{\partial\varphi}{\partial x_j} &=
       \frac{(1-A)\,a_2 - B\,a_1}{\|a_1\|\|a_2\|\sin\varphi}
       - \cot\varphi\left[(1-A)\frac{a_1}{\|a_1\|^2} - B\frac{a_2}{\|a_2\|^2}\right], \\
   \frac{\partial\varphi}{\partial x_k} &=
       \frac{(1+B)\,a_1 + A\,a_2}{\|a_1\|\|a_2\|\sin\varphi}
       - \cot\varphi\left[(1+B)\frac{a_2}{\|a_2\|^2} + A\frac{a_1}{\|a_1\|^2}\right].

These formulas are due to Wilson and tabulated in [Schlegel1984]_; PyBerny
implements them in :func:`berny.coords.Dihedral.eval` and includes the
:math:`\varphi\to 0,\pm\pi` limiting forms there.

:sm:`Note:` the SM appendix simply states that :math:`\varphi` lies in
:math:`(-\pi,\pi]` and does not say how to handle the
:math:`\pm\pi`-wraparound during back-transformation, only that "some
care must be taken with dihedral angles to avoid extraneous multiples of
360°" (paraphrasing [PengJCC96]_). PyBerny resolves this by comparing
each newly computed dihedral against a template and shifting by multiples
of :math:`\pi` as needed
(:meth:`berny.coords.InternalCoords.eval_geom`).

Out-of-plane bends
^^^^^^^^^^^^^^^^^^

The SM lists out-of-plane bends as an admissible coordinate but does
*not* use them, observing that for molecules with more than 4 atoms the
redundancy in the standard stretch/bend/torsion set already covers the
relevant motions. :pb:`PyBerny:` same — no out-of-plane coordinates are
generated.

Linear bends
^^^^^^^^^^^^

When an angle :math:`\theta_{123}` exceeds 165° and the central atom 2
is bonded to fewer than 5 atoms, the SM replaces it by a dummy-atom
construction:

* Place a dummy atom *d* in the plane spanned by :math:`r_{12}, r_{23}`
  (arbitrary plane if the angle is exactly :math:`\pi`) such that
  :math:`\theta_{12d}=\theta_{32d}` and :math:`r_{2d}` equals a fixed
  constant (:math:`2\,a_0` in [BirkholzTCA16]_; :sm:`the choice is
  explicitly described as "more or less arbitrary"`).
* Replace :math:`\theta_{123}` by the equivalent sum
  :math:`\theta_{12d}+\theta_{32d}`.
* Add the dihedrals :math:`\varphi_{n12d}` and :math:`\varphi_{d23m}`
  (for every neighbour :math:`n` of atom 1 and :math:`m` of atom 3) and
  the improper :math:`\varphi_{12d3}`.
* Constrain three coordinates per linear bend: :math:`r_{2d}`,
  :math:`\theta_{12d}-\theta_{32d}`, and one dihedral chosen so that
  every dummy atom appears in at least one constraint (when only three
  atoms are involved, the improper :math:`\varphi_{12d3}` is used as the
  arbitrary dihedral).

The constrained transformation uses a projected B-matrix

.. math::

   B_\text{prj} \;=\; B_\text{opt} - B_\text{opt}\,B_\text{constr}^-\,B_\text{constr},

assuming that gradient and Hessian elements on dummy atoms are zero. If
analytic B-matrix derivatives are used (e.g. for Hessian transformation)
they are projected analogously:

.. math::

   [\partial B_\text{prj}]_i \;=\;
   (I - B_\text{cns}^-\,B_\text{cns})\,[\partial B_\text{opt}]_i\,(I - B_\text{cns}^-\,B_\text{cns}).

The back-transformation is done unconstrained with goal values for the
constrained coordinates set to their current values; a second iterative
back-transformation (moving only the dummy atom) re-imposes the dihedral
constraint if needed.

:pb:`Not implemented in PyBerny.` Linear-bend handling via dummy atoms is
open issue
`#30 <https://github.com/jhrmnn/pyberny/issues/30>`_. PyBerny currently
just skips dihedrals through nearly-linear angles via a recursive
"chain through the linear atom" rule (see `Construction of the
coordinate set`_), which works for most cases but is not equivalent to
the SM treatment.


Construction of the coordinate set
----------------------------------

:sm:`Important caveat:` the SM explicitly does **not** specify a
connectivity-detection rule — in [BirkholzTCA16]_ connectivity was
provided manually for every test molecule. The remainder of this
sub-section follows the canonical Peng/Ayala/Schlegel/Frisch construction
([PengJCC96]_), which is what PyBerny implements.

#. **Bonds.** A pair :math:`(i,j)` is bonded iff
   :math:`r_{ij} < 1.3\,(R_i^\text{cov}+R_j^\text{cov})`. :pb:`PyBerny`:
   additional pseudo-bonds are added between disconnected fragments
   (van-der-Waals radii, gradually inflated) until the molecular graph
   is connected — this part is a PyBerny choice not present in either
   the SM or [PengJCC96]_.
#. **Angles.** For each atom :math:`j`, every pair :math:`(i,k)` of
   atoms bonded to :math:`j` defines a candidate angle. PyBerny keeps
   only angles with :math:`\theta_{ijk}>\pi/4`.
#. **Dihedrals.** For each pair of angles sharing a bond, the
   corresponding dihedral is added if both 1-2-3 and 2-3-4 angles
   exceed :math:`\pi/4`. If one of those angles is (close to) zero,
   so that three atoms lie on a line, those three are used as a new
   base and the search recurses through the linear atom
   ([PengJCC96]_).
#. **Crystal cells.** :pb:`PyBerny-only:` for periodic systems, only
   the periodic image of each internal coordinate closest to the
   original unit cell is kept.


B-matrix
--------

Definition (Eq. 14):

.. math::

   B_{ia} \;=\; \frac{\partial q_i}{\partial x_a}\;\Longrightarrow\;
   \delta q_i = \sum_a B_{ia}\,\delta x_a.

:math:`B` is :math:`n_\text{ric}\times n_\text{crt}`. Rows are populated
by the per-coordinate gradients given in `Coordinate definitions`_.


Generalised inverse
-------------------

:math:`B` is rectangular and rank-deficient (rank
:math:`n_\text{act}=n_\text{crt}-6`). Two equivalent ways are given in
the SM.

Penalty form (Eqs. 15–18)
^^^^^^^^^^^^^^^^^^^^^^^^^

Add a projector onto the 6-dimensional translation-rotation space of
:math:`B^\mathrm T B`:

.. math::

   P_\text{TR} \;=\; \sum_{i=1}^{3}\Bigl(t_i t_i^\mathrm T + r_i r_i^\mathrm T\Bigr)
   \qquad\text{(Eq. 15)}

with, for atom :math:`k`,

.. math::

   t_{i,k} = e_i,\qquad
   r_{i,k} = x_k\times e_i
   \qquad\text{(Eqs. 16-17)}

where :math:`e_i` is the unit vector along Cartesian axis :math:`i`.
These vectors span (but are not orthonormal in) the rigid-body subspace.
The generalised inverse is

.. math::

   B^- \;=\; \bigl(B^\mathrm T B + P_\text{TR}\bigr)^{-1} B^\mathrm T.
   \qquad\text{(Eq. 18)}

The benefit of this form is that
:math:`B^\mathrm T B + P_\text{TR}` is invertible and amenable to
iterative inversion for large systems.

SVD form (Eq. 19)
^^^^^^^^^^^^^^^^^

Let :math:`B = U\,\Sigma\,V^\mathrm T` be the SVD, with :math:`U` of
size :math:`n_\text{ric}`, :math:`V` of size :math:`n_\text{crt}`, and
:math:`\Sigma` zero everywhere except for the first
:math:`n_\text{act}` diagonal elements. Then

.. math::

   B^- \;=\; V\,\Sigma^-\,U^\mathrm T,
   \qquad\text{(Eq. 19)}

where :math:`\Sigma^-_{ii}=1/\Sigma_{ii}` for non-zero singular values
and :math:`0` otherwise. A side benefit is that the **active space** is
read off as the first :math:`n_\text{act}` rows of :math:`U`.

:pb:`PyBerny variant:` instead of either Eq. 18 or Eq. 19 directly,
PyBerny computes
:math:`B^- = B^\mathrm T\bigl(B B^\mathrm T\bigr)^+` where
:math:`(B B^\mathrm T)^+` is the Moore-Penrose pseudoinverse via SVD with
a gap-based threshold (:func:`berny.Math.pinv`,
:file:`berny.py:162`). This is equivalent at the level of the gradient
transformation that follows, but it does *not* compute the SM active
space; the projector :math:`P = B B^-` is used instead, see
`Projection`_.

Gradient and Hessian transformation (Eqs. 20-21)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the chain rule :math:`g_x = B^\mathrm T g_q`:

.. math::

   g_q \;=\; (B^-)^\mathrm T g_x.
   \qquad\text{(Eq. 20)}

For a Cartesian Hessian:

.. math::

   H_q \;=\; (B^-)^\mathrm T\!
   \left(H_x - g_q^\mathrm T\,\frac{\partial B}{\partial x}\right)\!B^-.
   \qquad\text{(Eq. 21)}

:pb:`PyBerny note:` PyBerny never computes :math:`H_x` (it only consumes
Cartesian gradients) and so Eq. 21 is never invoked. The Hessian is
maintained in internal coordinates from step 1, starting from the
diagonal guess of `Initial Hessian`_ and updated by BFGS thereafter
(see `Hessian update`_).


Projection
----------

With the active space :math:`U_\text{act}` (top :math:`n_\text{act}`
rows of :math:`U` from the SVD), the SM projects gradient and Hessian
into the non-redundant subspace:

.. math::

   g_\text{r} = U_\text{act}\,g_q,\qquad
   H_\text{r} = U_\text{act}\,H_q\,U_\text{act}^\mathrm T.

The step :math:`\Delta q^\mathrm{r}` is computed in this reduced space
and lifted back via

.. math::

   \Delta q_0 \;=\; U_\text{act}^\mathrm T\,\Delta q^\mathrm{r}.

:pb:`PyBerny variant — penalty projector from` [PengJCC96]_. PyBerny
does not compute :math:`U_\text{act}`. Instead it forms the projector

.. math::

   P \;=\; B\,B^- = (B^-)^\mathrm T B^\mathrm T,

and uses the augmented Hessian

.. math::

   H_\text{proj} \;=\; P\,H\,P + \alpha\,(I - P),\qquad \alpha=1000\;\text{a.u.}

(Peng et al., Eq. 9 with :math:`\alpha=1000` au). The :math:`\alpha(I-P)`
penalty drives the step out of the redundant subspace and is
mathematically equivalent to projecting into the active space when
:math:`\alpha` is large enough relative to the eigenvalues of
:math:`H` ([PengJCC96]_). The hard-coded constant 1000 a.u. is at
:file:`berny.py:191`.


RFO step
--------

The SM uses rational-function optimisation [BanerjeeJPC85]_ to compute
the step. The PES is approximated locally as

.. math::

   E_\text{RF}(q_0+\Delta q) \;=\; E_0 +
   \frac{\Delta q^\mathrm T g_0 + \tfrac12 \Delta q^\mathrm T H_0\,\Delta q}
        {1 + \Delta q^\mathrm T S\,\Delta q},
   \qquad\text{(Eq. 24)}

with :math:`S = I` taken for convenience. The :math:`n+1` stationary
points of Eq. 24 are exactly the eigenvectors of the augmented Hessian

.. math::

   H_\text{aug} \;=\; \begin{pmatrix} H & g \\ g^\mathrm T & 0 \end{pmatrix}.
   \qquad\text{(Eq. 25)}

If :math:`(v_m,\lambda_m)` is the :math:`m`-th eigenpair of
:math:`H_\text{aug}`, the step is

.. math::

   \Delta q_{\text{RFO},m} \;=\;
   \frac{1}{v_{m,n+1}}\bigl(v_{m,1},v_{m,2},\dots,v_{m,n}\bigr)^\mathrm T,
   \qquad\text{(Eq. 26)}

or, equivalently,

.. math::

   \Delta q_{\text{RFO},m} \;=\; -(H - \lambda_m I)^{-1} g
   \qquad\text{(Eq. 27)},

with

.. math::

   \lambda_m \;=\; \Delta q_{\text{RFO},m}^\mathrm T\,g.
   \qquad\text{(Eq. 28)}

For a minimum the most negative eigenvalue of :math:`H_\text{aug}` is
selected; the resulting :math:`\lambda_m` is negative and shifts
:math:`H` to a positive-definite operator, so Eq. 27 yields a
descent step even when :math:`H` itself has negative eigenvalues.

Trust-region restriction
^^^^^^^^^^^^^^^^^^^^^^^^

If the pure RFO step satisfies :math:`\|\Delta q\|\le\tau`, accept it as
is. Otherwise solve the **sphere-restricted minimisation** of Eq. 27,
i.e. find the largest :math:`\lambda<\lambda_\text{min}(H)` for which

.. math::

   \bigl\|(\lambda I - H)^{-1} g\bigr\| \;=\; \tau,

and use :math:`\Delta q = (\lambda I - H)^{-1} g`. PyBerny solves this
1-D equation with a Newton iteration
(:func:`berny.Math.findroot`).

Transition states (partitioned RFO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For TS optimisation the SM uses partitioned RFO (pRFO,
[BanerjeeJPC85]_, [Baker1996]_): the Hessian is split into an
:math:`(n-1)`-dimensional minimisation subspace and a 1-D maximisation
subspace (usually the eigenvector to be followed uphill), each handled
by its own augmented-Hessian sub-problem.

:sm:`Unspecified in [BirkholzTCA16]_:` which eigenvector to follow at
each step. The SM notes only "usually selected to be eigenvectors of
the Hessian". The original references give detailed eigenvector-mode
following rules.

:pb:`Not implemented in PyBerny.` Only minimisation is supported. TS
support is part of open issue
`#29 <https://github.com/jhrmnn/pyberny/issues/29>`_.


Trust region
------------

Initial value
^^^^^^^^^^^^^

SM: :math:`\tau_0 = 0.5` bohr or rad. :pb:`PyBerny default:`
:math:`\tau_0 = 0.3` (:class:`berny.BernyParams`).

Adaptive update
^^^^^^^^^^^^^^^

Let

.. math::

   \Delta E_\text{quad} \;=\; \Delta q^\mathrm T g + \tfrac12\,\Delta q^\mathrm T H\,\Delta q

be the quadratic energy change predicted at the *previous* step, and
:math:`\Delta E_\text{act}` the actual change observed at the current
step. Define the ratio

.. math::

   r \;=\; \frac{\Delta E_\text{act}}{\Delta E_\text{quad}}.

Update :math:`\tau` according to:

* if :math:`r > 0.75` *and* :math:`\|\Delta q\| \ge 0.8\,\tau`,
  :math:`\tau \leftarrow 2\tau`;
* if :math:`r < 0.25`,
  :math:`\tau \leftarrow \tfrac14\,\|\Delta q\|`;
* :sm:`otherwise (implicit in [BirkholzTCA16]_):` :math:`\tau` is left
  unchanged. See [Fletcher00]_ §5.1 and [DennisSchnabel83]_ ch. 6 for
  the theoretical background of these thresholds.

:pb:`PyBerny note:` PyBerny implements this rule in
:func:`berny.berny.update_trust` exactly as written above. The
boundary check :math:`\|\Delta q\|\ge 0.8\tau` is, however, applied as a
strict-equality test in code (``abs(norm(dq) - trust) < 1e-10``), which
in practice is satisfied only when the previous step hit the trust
sphere — equivalent to the SM rule in floating-point arithmetic.


Hessian update
--------------

Define :math:`s = \Delta q`, :math:`y = \Delta g_q`, :math:`z = y - H s`.

BFGS (Eq. 29)
^^^^^^^^^^^^^

.. math::

   \Delta H_\text{BFGS}
   \;=\;
   \frac{y\,y^\mathrm T}{y^\mathrm T s}
   - \frac{H\,s\,s^\mathrm T H}{s^\mathrm T H s}.

Standard choice for minimisation. The update is reliable as long as
:math:`s^\mathrm T y > 0` and :math:`H` remains positive definite.

SR1 / Murtagh-Sargent (Eq. 30)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \Delta H_\text{SR1} \;=\; \frac{z\,z^\mathrm T}{z^\mathrm T s}.

Produces good updates when the change in :math:`H` is large; unstable
when :math:`s^\mathrm T z` is small relative to :math:`z^\mathrm T z`.

PSB (Eq. 31)
^^^^^^^^^^^^

.. math::

   \Delta H_\text{PSB}
   \;=\;
   \frac{s\,z^\mathrm T + z\,s^\mathrm T}{s^\mathrm T s}
   - \frac{(s^\mathrm T z)\,s\,s^\mathrm T}{(s^\mathrm T s)^2}.

Very stable, but updates tend to be small/poor for large Hessian changes.

Bofill / MSP (Eqs. 32-33)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \Delta H_\text{MSP}
   \;=\;
   \phi_\text{MSP}\,\Delta H_\text{SR1}
   + (1-\phi_\text{MSP})\,\Delta H_\text{PSB},
   \qquad
   \phi_\text{MSP}
   \;=\;
   \frac{(s^\mathrm T z)^2}{(z^\mathrm T z)\,(s^\mathrm T s)}.

This is the [Bofill1994]_ φ-mixed combination, recommended by the SM for
transition-state work.

:pb:`PyBerny:` only BFGS is implemented
(:func:`berny.berny.update_hessian`,
:file:`berny.py:223`). SR1, PSB, and MSP are not available. The SM
itself uses BFGS for minimisations, so for minima PyBerny is in line
with the SM; for transition states the lack of MSP is one of several
missing pieces (see `RFO step`_).


Back-transformation
-------------------

Given a step :math:`\Delta q_0` in internal coordinates at Cartesian
geometry :math:`x_0`, the corresponding Cartesian displacement is
defined as the minimiser of the functional

.. math::

   F(x_0 + \Delta x) \;=\;
   \tfrac12\bigl\|q_0 + \Delta q_0 - q(x_0+\Delta x)\bigr\|^2.
   \qquad\text{(Eq. 22)}

This is solved by the fixed-point iteration

.. math::

   x_{i+1} \;=\; x_i + B_i^-\,\Delta q_i,\qquad
   \Delta q_{i+1} \;=\; \Delta q_i - \bigl(q(x_{i+1}) - q(x_i)\bigr).
   \qquad\text{(Eq. 23)}

The SM converges when RMS\ :math:`(\Delta x_i)<10^{-6}\;a_0`.

:pb:`PyBerny:` capped at 20 iterations; if it does not converge,
PyBerny falls back to the first-iteration estimate
:math:`x_0 + B_0^- \Delta q_0`
(:meth:`berny.coords.InternalCoords.update_geom`). This fallback is the
same as that suggested by [PengJCC96]_ ("in the rare cases in which the
iteration does not converge").


Initial Hessian
---------------

:sm:`The SM appendix is silent on the initial-Hessian model.` In the
test calculations of [BirkholzTCA16]_ no extrapolation or
multi-step-history updating was used; the Hessian was updated using
"only the information at the current point and the most recent point",
but the *starting* Hessian model is not specified.

PyBerny fills this gap with a Lindh-style diagonal guess
([SwartIJQC06]_), expressed in terms of the screening function

.. math::

   \rho_{ij} \;=\;
   \exp\!\left(1 - \frac{r_{ij}}{R_i^\text{cov} + R_j^\text{cov}}\right),

so that :math:`\rho_{ij}` is :math:`e\approx 2.72` at zero distance,
:math:`1` at covalent contact, and decays for weak/non-bonded pairs.
Diagonal entries are

.. math::

   k_\text{bond} &= 0.45\,\rho_{ij}, \\
   k_\text{angle} &= 0.15\,\rho_{ij}\,\rho_{jk}, \\
   k_\text{dihedral} &= 0.005\,\rho_{ij}\,\rho_{jk}\,\rho_{kl}.

The prefactors 0.45 / 0.15 / 0.005 are the original Lindh values quoted
in [SwartIJQC06]_; PyBerny implements these in
:meth:`berny.coords.Bond.hessian` and siblings.


Convergence
-----------

The SM uses the same four-criterion test as Gaussian 09:

================================== ===================
RMS internal-coord. step           :math:`< 1.2\times 10^{-3}` bohr/rad
max\ :math:`|\Delta q|`            :math:`< 1.8\times 10^{-3}` bohr/rad
RMS gradient (active space)        :math:`< 1.5\times 10^{-4}` Eh/bohr or Eh/rad
max gradient (active space)        :math:`< 4.5\times 10^{-4}` Eh/bohr or Eh/rad
================================== ===================

All four must be satisfied simultaneously. The defaults in
:class:`berny.BernyParams` reproduce these thresholds exactly. When the
sphere-restricted minimisation was triggered on a step, the step-based
criteria are by construction not satisfied; PyBerny then skips the
step-based criteria and demands the gradient-based criteria only
(:func:`berny.berny.is_converged`).


Coordinate weighting
--------------------

:sm:`Not in the SM.` PyBerny computes per-coordinate weights derived
from the same :math:`\rho_{ij}` screening function ([SwartIJQC06]_;
see :meth:`berny.coords.Angle.weight` etc.) and threads them through
to the step-computation routine. However, they are currently *not*
consumed by the RFO / sphere-restricted-minimisation step
(:func:`berny.berny.quadratic_step` receives but ignores them). So in
practice PyBerny's behaviour matches the SM here; this is a latent
extension point.


Linear search
-------------

:sm:`Not in the SM.` The SM appendix explicitly notes that "no
extrapolation methods were employed" for the test calculations.

:pb:`PyBerny extension` (:func:`berny.berny.linear_search`): between
the current point and the best point so far, PyBerny fits a constrained
quartic polynomial to :math:`(E,g_\parallel)` at both endpoints. If the
quartic has a usable minimum on :math:`t\in[-1,2]`, that
:math:`t`-value is used to interpolate energy and gradient; otherwise
a cubic fit is tried on :math:`t\in[0,1]`; if both fail, the
better-energy endpoint is kept. The interpolated point is then used as
the expansion origin for the RFO step. See the Gaussian
``opt`` documentation, "If a minimum is sought…", for the original
rationale.


PyBerny vs. SM at a glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Aspect
     - SM ([BirkholzTCA16]_)
     - PyBerny
   * - Coordinate set
     - stretches / bends / torsions
     - stretches / bends / torsions
   * - Out-of-plane bends
     - not used
     - not used
   * - Linear bends
     - dummy-atom construction
     - **not implemented** (issue `#30`_)
   * - Bond detection
     - manual
     - :math:`r_{ij}<1.3\,(R_i^\text{cov}+R_j^\text{cov})` plus a vdW
       shell that grows by 1 Å until cluster fragments are connected
   * - Initial Hessian
     - **unspecified**
     - Lindh diagonal with :math:`\rho_{ij}` screening
   * - B-matrix
     - Eq. 14
     - same
   * - :math:`B^-`
     - Eq. 18 or Eq. 19
     - :math:`B^\mathrm T (B B^\mathrm T)^+` via SVD
   * - Active space
     - :math:`U_\text{act}` from SVD
     - :math:`P = B B^-` projector + 1000 a.u. penalty
   * - Hessian transformation (Cart→int)
     - Eq. 21
     - never computed (internal-only Hessian)
   * - RFO step
     - Eqs. 24-28
     - same
   * - Trust region
     - adaptive (75/25 rule), :math:`\tau_0=0.5`
     - same rule, :math:`\tau_0=0.3`
   * - Sphere-restricted step
     - Lagrange-multiplier shift
     - same (Newton 1-D root find)
   * - TS optimisation (pRFO)
     - specified
     - **not implemented** (issue `#29`_)
   * - Hessian update (minima)
     - BFGS
     - BFGS
   * - Hessian update (TS)
     - BFGS / SR1 / PSB / MSP
     - only BFGS available
   * - Back-transformation
     - Eqs. 22-23, RMS Δx :math:`<10^{-6}`
     - 20-iteration cap, Peng fallback to first estimate
   * - Convergence criteria
     - 4 thresholds
     - same
   * - Line search
     - not used
     - quartic-then-cubic between best and current

.. _#29: https://github.com/jhrmnn/pyberny/issues/29
.. _#30: https://github.com/jhrmnn/pyberny/issues/30


Additional references
---------------------

In addition to the references defined in :doc:`algorithm`, this page
cites:

.. [Schlegel1984] Schlegel, H. B. Estimating the Hessian for gradient-type
   geometry optimizations. *Theor. Chim. Acta* 66, 333 (1984). Source of
   the analytic B-matrix derivatives for stretch/bend/torsion
   coordinates.
.. [Bofill1994] Bofill, J. M. Updated Hessian matrix and the restricted
   step method for locating transition structures. *J. Comput. Chem.*
   15(1):1 (1994). Source of the MSP / "Bofill" Hessian update.
.. [Baker1996] Baker, J. & Chan, F. The location of transition states:
   a comparison of cartesian, Z-matrix, and natural internal
   coordinates. *J. Comput. Chem.* 17(7):888 (1996). Eigenvector-following
   for TS optimisation.
.. [DennisSchnabel83] Dennis, J. E. & Schnabel, R. B. *Numerical Methods
   for Unconstrained Optimization and Nonlinear Equations.* Prentice-Hall
   (1983). Chapter 6 covers adaptive trust-region theory.
