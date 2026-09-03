from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._statistics import rank_correlation, spread


@dataclass(frozen=True)
class GramRead:
    """What the kernel matrix says about the kernel, before any conditioning.

    The fields are measurements, not verdicts: the failure modes below have names but no
    thresholds that separate them, and a threshold depends on the space, since eleven terms of one
    chain and four hundred convolutional architectures do not share one.  A caller states the
    bound its space earns and compares.

    Attributes:
        size (int): How many terms the matrix covers.
        symmetry_error (float): The largest asymmetry of the normalized matrix, ``max |K - K^T|``.
            A kernel is symmetric by definition, so anything above rounding error is a defect in
            the kernel rather than a property of the terms.
        minimum_eigenvalue (float): The smallest eigenvalue of the normalized matrix, symmetrized.
            A kernel is positive semidefinite, so this is at least zero in exact arithmetic.
            Eigenvalues below zero at machine precision call for the noise term the surrogate adds
            to the diagonal, and a value below zero by more than that is a defect.  Measured after
            normalization because normalization is a congruence, which leaves the signs of the
            eigenvalues alone while putting their scale at one.  Symmetrized because an asymmetric
            matrix has complex eigenvalues and no smallest one, and its asymmetry is reported in
            its own field rather than through an unreadable spectrum.
        condition_number (float): The condition number of the same symmetrized matrix.
            Near-duplicate rows drive it up.  That is the ill-conditioning the noise term on the
            diagonal guards against, and the reason the loop conditions its Gaussian process on
            distinct pairs only.
        diagonal_spread (float): ``max k(t,t) / min k(t,t)`` on the **raw** matrix.  The hazard
            specific to counting kernels: self-similarity grows with term size, so large terms
            dominate the scale.  One means no spread, and the subtree kernel on a chain of eleven
            nested lists reaches one hundred eleven.  This is the one field the normalization
            would erase, which is why it is read before it.
        off_diagonal_mean (float): The mean of the off-diagonal entries of the normalized matrix.
            Near one is the near-constant matrix that discriminates nothing, since all terms count
            as alike.  Near zero is the near-diagonal matrix that transfers nothing, since every
            term resembles only itself and observations reach nowhere.
        off_diagonal_spread (float): The range of those entries.  Near zero says the matrix has no
            structure at all, whichever of the two extremes it sits at.  A healthy matrix shows
            structure between them, the blocks of term families an ordering by similarity brings
            out.
        seriation (tuple[int, ...]): The indices of the terms ordered along the first kernel
            principal component.  A landscape picture over a search space with no order of its own
            needs an axis, and this ordering supplies one under which those blocks become visible.
            **Read it together with the next field**: it is not always the order the terms have,
            since under a stationary kernel on a chain it folds the ends, though this repository's
            counting kernels recover it, and :attr:`seriation_neighbour_similarity` is what says
            which case a matrix is in.
        seriation_neighbour_similarity (float): The mean normalized kernel value between terms
            that the seriation puts next to each other.  A seriation places similar terms side by
            side, so this should stand well above :attr:`off_diagonal_mean`, the similarity of two
            terms picked at random.  Where it does not, the axis is not the one that was asked for
            and the landscape drawn along it is misleading rather than approximate.
        coordinate (tuple[float, ...]): That component itself, one value per term in the **input**
            order.  The sign of an eigenvector is arbitrary.  It is fixed here so that two runs on
            the same matrix seriate the same way, but it carries no meaning.
        coordinate_spread (float): Its range.  Zero says the centered matrix is degenerate and
            there is no axis at all, which the seriation cannot say on its own: with every
            coordinate equal, ``argsort`` returns the caller's input order, and a picture drawn
            along it shows whatever order the terms were passed in.  A constant kernel matrix, the
            one that discriminates nothing, is exactly that case.
        objective_alignment (float | None): The absolute rank correlation between the coordinate
            and the objective values, where those were supplied.  A matrix is healthy when its
            blocks track the objective.  Absolute because of the sign above.  ``None`` where the
            coordinate or the objective is constant and the ranks therefore do not exist.
        normalized_matrix (np.ndarray): The normalized matrix, for plotting.  Seriate it with
            ``matrix[np.ix_(read.seriation, read.seriation)]``.
        diagonal (tuple[float, ...]): The raw self-similarities, for plotting the scale spread.
    """

    size: int
    symmetry_error: float
    minimum_eigenvalue: float
    condition_number: float
    diagonal_spread: float
    off_diagonal_mean: float
    off_diagonal_spread: float
    seriation: tuple[int, ...]
    seriation_neighbour_similarity: float
    coordinate: tuple[float, ...]
    coordinate_spread: float
    objective_alignment: float | None
    normalized_matrix: np.ndarray
    diagonal: tuple[float, ...]


def read_gram(matrix: np.ndarray, *, objective: Sequence[float] | None = None) -> GramRead:
    """Measure a kernel matrix for symmetry, conditioning, scale spread, and structure.

    This is the first of the four acceptance checks, and the only one that needs no data: the
    matrix betrays a broken kernel before any Gaussian process is conditioned.  A kernel that
    cannot tell the terms apart cannot be repaired by conditioning it on more of them.

    The argument is the matrix rather than a kernel and a list of terms, because an sklearn kernel
    *is* the callable that produces it::

        read = read_gram(kernel(terms), objective=[q(t) for t in terms])

    and a caller who computed the matrix once for something else does not compute it again.

    Every structural measurement is taken after the normalization ``K_ij / sqrt(K_ii K_jj)``, so
    that the counting kernels' growing self-similarity does not disguise the structure.  The scale
    spread itself is measured before it, since normalizing is what removes it.

    Args:
        matrix (np.ndarray): The kernel matrix on a set of terms, square and at least ``2 x 2``.
        objective (Sequence[float] | None): The objective value of each term, in the order the
            matrix covers them.  Supplied, it answers whether the seriation's blocks track the
            objective.  Omitted, that field stays ``None``.

    Returns:
        GramRead: The measurements.

    Raises:
        ValueError: If the matrix is not square, holds fewer than two terms, is not finite, or has
            a self-similarity that is not strictly positive.  The last is not a badly fitting
            kernel but a broken one, since ``k(t,t) = <phi(t), phi(t)>`` is a squared norm, and it
            is also the point at which the normalization would divide by zero.  Also if the
            objective does not match the matrix in length.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = f"a kernel matrix is square; this one has shape {array.shape}"
        raise ValueError(msg)
    size = int(array.shape[0])
    if size < 2:
        msg = (
            f"reading a kernel matrix means reading what it says *between* terms, and {size} "
            f"term has no off-diagonal entry to say it in"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "the kernel matrix holds values that are not finite"
        raise ValueError(msg)

    diagonal = np.diag(array).copy()
    if np.any(diagonal <= 0.0):
        smallest = float(np.min(diagonal))
        msg = (
            f"the smallest self-similarity of this matrix is {smallest}, and a kernel value "
            f"k(t,t) is a squared norm in the feature space, so it is positive for every term.  "
            f"A non-positive one is a defect in the kernel, and it is also the number that "
            f"the normalization divides by."
        )
        raise ValueError(msg)

    scale = np.sqrt(np.outer(diagonal, diagonal))
    normalized = array / scale

    off_diagonal = normalized[~np.eye(size, dtype=bool)]

    symmetry_error = float(np.max(np.abs(normalized - normalized.T)))
    symmetric = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    coordinate = _first_kernel_component(symmetric)

    alignment: float | None = None
    if objective is not None:
        values = np.asarray(objective, dtype=float).reshape(-1)
        if values.size != size:
            msg = (
                f"the matrix covers {size} terms and the objective {values.size} values; the "
                f"alignment pairs them by position"
            )
            raise ValueError(msg)
        correlation = rank_correlation(coordinate, values)
        alignment = None if correlation is None else abs(correlation)

    order = np.argsort(coordinate, kind="stable")
    neighbours = normalized[order[:-1], order[1:]]

    return GramRead(
        size=size,
        symmetry_error=symmetry_error,
        minimum_eigenvalue=float(eigenvalues[0]),
        condition_number=float(np.linalg.cond(symmetric)),
        diagonal_spread=float(np.max(diagonal) / np.min(diagonal)),
        off_diagonal_mean=float(np.mean(off_diagonal)),
        off_diagonal_spread=spread(off_diagonal),
        seriation=tuple(int(index) for index in order),
        seriation_neighbour_similarity=float(np.mean(neighbours)),
        coordinate=tuple(float(value) for value in coordinate),
        coordinate_spread=spread(coordinate),
        objective_alignment=alignment,
        normalized_matrix=normalized,
        diagonal=tuple(float(value) for value in diagonal),
    )


def _first_kernel_component(symmetric: np.ndarray) -> np.ndarray:
    """Return the first kernel principal component of a normalized kernel matrix.

    The axis a landscape picture needs: center the matrix in the feature space and project onto
    the leading eigenvector, which places similar terms side by side.  It is kernel PCA with one
    component, written out rather than imported, because the two lines that matter here, the
    centering and the sign convention, are the two a caller of a general decomposition would have
    to check anyway.

    ``eigh`` returns an eigenvector up to sign, so a convention is needed or the seriation of one
    matrix comes out forwards or backwards depending on the LAPACK build, and two pictures of the
    same kernel stop being comparable.  The convention here is the **first entry that is clearly
    away from zero**, made positive.

    The obvious convention, the entry of largest magnitude, which is what ``svd_flip`` uses, is
    the one this had first, and it does not work on the case it was written for.  On a chain of
    nested lists the component is exactly antisymmetric, so its two largest magnitudes are equal:
    measured, they differ by 2.2e-16, and the winner is decided by rounding.  Two spellings of the
    same squared exponential that agree to 1.1e-16, ``exp(-d**2/(2*l**2))`` and sklearn's own
    ``RBF(l)``, then produce opposite seriations.  Comparing the first sufficiently large entry
    against a threshold has no such tie: it turns on one number being big, not on two being
    ordered.

    What no convention can supply is meaning.  A seriation is determined up to direction, and on a
    symmetric landscape nothing in the kernel prefers one end.  This fixes the direction so that
    repeated reads agree, and that is all it does.

    **Measured against a chain of nested lists: this component need not recover the order of a
    chain, and where it fails it fails at the ends.**  On the eleven single-element lists under
    the squared-exponential covariance the coordinate runs ``0.513, 0.654, 0.679, 0.559, 0.315,
    0, -0.315, ...``, monotone through the middle and folded back over the outermost three
    positions at each end, so the seriation puts the lists of lengths zero and four side by side,
    whose normalized similarity is 0.085 against the 0.857 of true neighbors.  A chain is the case
    where the kernel geometry and the term size agree, so seriating along this component is at its
    most favorable there.

    It is not a numerical accident: the leading eigenvector of a centered stationary kernel on a
    grid is its lowest oscillating mode, and that mode turns at the boundary.  It **is**
    conditional on the lengthscale, and the earlier claim here that it was not has been withdrawn:
    measured over the eleven lists, the fold is there up to about ``l = 3.65`` and gone above it,
    where the covariance is flat enough over the whole grid that the leading mode is monotone.  A
    counting kernel is not stationary at all, and the normalized subtree kernel on the same terms
    recovers the size order outright.

    Nothing is done about any of it here.  The first kernel principal component is what the
    construction calls for, and it is what this computes.  What this package adds is the
    measurement that exposes the deviation, :attr:`GramRead.seriation_neighbour_similarity`.

    Args:
        symmetric (np.ndarray): The normalized, symmetrized kernel matrix.

    Returns:
        np.ndarray: One coordinate per term.
    """
    size = symmetric.shape[0]
    ones = np.full((size, size), 1.0 / size)
    centred = symmetric - ones @ symmetric - symmetric @ ones + ones @ symmetric @ ones
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (centred + centred.T))

    leading = eigenvectors[:, -1]
    # A negative eigenvalue at this point is numerical noise around zero: the centered matrix is
    # positive semidefinite whenever the kernel matrix is, and the leading one is the largest.
    # Clipping keeps the square root real without inventing a scale.
    coordinate = leading * float(np.sqrt(max(eigenvalues[-1], 0.0)))

    magnitudes = np.abs(coordinate)
    largest = float(np.max(magnitudes))
    if largest == 0.0:
        # Every entry is zero: the centered matrix is the zero matrix, which happens exactly when
        # the kernel says every pair of terms is equally similar.  There is no axis to orient.
        return coordinate
    first_substantial = int(np.argmax(magnitudes > 1e-8 * largest))
    if coordinate[first_substantial] < 0.0:
        coordinate = -coordinate
    return coordinate
