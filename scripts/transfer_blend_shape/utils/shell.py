"""Detached shell (mesh element) support for the deformation transfer.

Background
----------
The deformation transfer solves for vertex positions from per-triangle
deformation gradients. That operator is *translation invariant* -- each
triangle's row block sums to zero -- so it determines positions only up to one
free translation **per connected component**. The solve removes that freedom
using the zero-delta ``static`` vertices, whose contribution is moved to the
right hand side of the normal equations.

A head mesh is not one component. It is the face skin plus a set of detached
eyebrow cards and eyelash strips. Any of those shells in which *every* vertex
moves therefore has no anchor, its block of the normal matrix is singular, and
the solve loses its position entirely -- the shell keeps its shape but lands at
(or near) the object origin.

This module fixes that by taking the detached shells out of the linear solve
altogether and instead *deriving* their motion from the already-solved skin.
That also sidesteps a second problem: eyelash geometry is sliver-heavy, and
sliver triangles are exactly what makes the per-triangle QR inverse and the
area-ratio smoothing weights ill-conditioned.

Follow model
------------
Each detached shell is bound to the nearest patch of skin at rest. For a
deformed skin we build a smooth field of local rigid transforms (one Kabsch fit
per skin vertex over its 1-ring, which overlap and so vary smoothly), blend them
per follower vertex with normalised Gaussian weights, and optionally collapse
the result to a single rigid transform for the whole shell.

Two properties hold *exactly*, by construction, and are covered by tests:

* rest pose in -> rest pose out, to machine precision;
* a rigid motion of the skin carries the shell rigidly.

``stiffness`` selects the behaviour an artist wants per shell:

* ``1.0`` -- the shell is moved by a single rigid transform. Its shape is
  preserved exactly; only its position and orientation track the skin. This is
  what eyebrow cards need.
* ``0.0`` -- the shell deforms with the skin via the smooth blended field. This
  is what eyelash strips need so they bend along the lid.
* anything between -- linear blend of the two.
"""
import logging

import numpy
import scipy.spatial


__all__ = [
    "EPS",
    "connected_components",
    "kabsch",
    "is_degenerate",
    "project_to_rotation",
    "build_binding",
    "evaluate_binding",
    "transplant_source_offset",
]

log = logging.getLogger(__name__)

EPS = 1e-12

# A source shell with almost no extent cannot establish a scale for an authored
# displacement, and the target/source ratio then explodes. Grooms differ between
# characters, but nothing like this much, so a ratio beyond it means the source
# shell is degenerate rather than genuinely small.
MAX_SIZE_RATIO = 10.0


# ----------------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------------
def connected_components(connectivity):
    """Label the connected components (mesh elements/shells) of a mesh.

    Iterative flood fill -- recursion would overflow on production head meshes.

    :param list[list[int]] connectivity: Per-vertex connected vertex indices.
    :return: Per-vertex component label, and the labels ordered by descending
        vertex count.
    :rtype: tuple[numpy.ndarray, list[int]]
    """
    num = len(connectivity)
    labels = numpy.full(num, -1, dtype=numpy.int64)
    order = []

    label = 0
    for seed in range(num):
        if labels[seed] != -1:
            continue

        stack = [seed]
        labels[seed] = label
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            for other in connectivity[index]:
                if labels[other] == -1:
                    labels[other] = label
                    stack.append(other)

        order.append((size, label))
        label += 1

    order.sort(key=lambda item: (-item[0], item[1]))
    return labels, [item[1] for item in order]


# ----------------------------------------------------------------------------
# rigid maths
# ----------------------------------------------------------------------------
def kabsch(a, b):
    """Best-fit rigid transform mapping point set ``a`` onto point set ``b``.

    Reflections are excluded, so the result is always a proper rotation.

    :param numpy.ndarray a: (K, 3) source points.
    :param numpy.ndarray b: (K, 3) destination points.
    :return: Rotation (3, 3) and translation (3,) with ``b ~= a @ r.T + t``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)

    centre_a = a.mean(axis=0)
    centre_b = b.mean(axis=0)
    h = (a - centre_a).T.dot(b - centre_b)

    u, _, vt = numpy.linalg.svd(h)
    flip = numpy.array([1.0, 1.0, numpy.sign(numpy.linalg.det(vt.T.dot(u.T))) or 1.0])
    r = (vt.T * flip).dot(u.T)
    return r, centre_b - r.dot(centre_a)


def is_degenerate(points, ratio=1e-4):
    """Whether a point cloud is effectively one dimensional (near collinear).

    A rigid fit needs the cloud to span at least a plane. A collinear cloud
    leaves the rotation about its own axis unconstrained.

    :param numpy.ndarray points: (K, 3) points.
    :param float ratio: Second/first singular value ratio below which the cloud
        counts as degenerate.
    :rtype: bool
    """
    points = numpy.asarray(points, dtype=float)
    if len(points) < 3:
        return True

    singular = numpy.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] < EPS:
        return True

    return bool(singular[1] / singular[0] < ratio)


def project_to_rotation(matrices):
    """Project each matrix onto the closest proper rotation.

    Polar decomposition via SVD. A matrix that is already a rotation is
    returned unchanged, which is what keeps the rigid-motion guarantee exact.

    :param numpy.ndarray matrices: (..., 3, 3) matrices.
    :return: (..., 3, 3) proper rotations.
    :rtype: numpy.ndarray
    """
    u, _, vt = numpy.linalg.svd(matrices)
    determinant = numpy.linalg.det(numpy.matmul(u, vt))
    flip = numpy.ones(u.shape[:-1])
    flip[..., 2] = numpy.where(determinant < 0.0, -1.0, 1.0)
    return numpy.matmul(u * flip[..., None, :], vt)


# ----------------------------------------------------------------------------
# binding
# ----------------------------------------------------------------------------
def build_binding(follower_points, skin_points, skin_connectivity, num_neighbours=8):
    """Bind follower vertices to a skin patch at rest.

    Everything that depends only on the rest pose is precomputed here so a
    blend shape set costs one bind and N cheap evaluations.

    :param numpy.ndarray follower_points: (M, 3) follower rest positions.
    :param numpy.ndarray skin_points: (P, 3) skin rest positions.
    :param list[list[int]] skin_connectivity: Per-skin-vertex 1-ring, indexed
        in the same local space as ``skin_points``.
    :param int num_neighbours: Skin vertices blended per follower vertex. More
        gives a smoother, broader field.
    :return: Binding table for :func:`evaluate_binding`.
    :rtype: dict
    """
    follower_points = numpy.asarray(follower_points, dtype=float)
    skin_points = numpy.asarray(skin_points, dtype=float)

    # padded 1-ring table, each row is [self, neighbours..., padding]
    width = max(len(ring) for ring in skin_connectivity) + 1
    ring_index = numpy.zeros((len(skin_points), width), dtype=numpy.int64)
    ring_mask = numpy.zeros((len(skin_points), width), dtype=float)
    for i, ring in enumerate(skin_connectivity):
        indices = [i] + list(ring)
        ring_index[i, :len(indices)] = indices
        ring_mask[i, :len(indices)] = 1.0

    # a ring that is collinear, or a valence one vertex sitting on top of its
    # only neighbour, leaves the rotation about its own axis undetermined. Rank
    # is a rest pose property so it is resolved once, here, and those vertices
    # are then excluded from the blend rather than allowed to contribute a
    # rotation that is only partly defined.
    ring_valid = _ring_rank_is_sufficient(skin_points, ring_index, ring_mask)
    if not numpy.all(ring_valid):
        log.debug("%d of %d skin rings are rank deficient and will not contribute "
                  "a rotation.", int(numpy.count_nonzero(~ring_valid)), len(skin_points))

    # nearest skin vertices per follower vertex, with a smooth adaptive kernel
    k = int(min(num_neighbours, len(skin_points)))
    tree = scipy.spatial.cKDTree(skin_points)
    distances, neighbour_index = tree.query(follower_points, k=k)

    # cKDTree drops the neighbour axis entirely when k is 1, and atleast_2d
    # would prepend it instead of appending, transposing the whole table
    distances = numpy.asarray(distances, dtype=float).reshape(len(follower_points), -1)
    neighbour_index = numpy.asarray(neighbour_index).reshape(len(follower_points), -1)

    bandwidth = distances[:, -1:].copy()
    bandwidth[bandwidth < EPS] = 1.0
    weights = numpy.exp(-3.0 * (distances / bandwidth) ** 2)
    total = weights.sum(axis=1, keepdims=True)
    degenerate = (total < EPS).ravel()
    if numpy.any(degenerate):
        weights[degenerate] = 1.0
        total[degenerate] = float(k)
    weights /= total

    # weights used for orientation only, dropping the unusable rings. Where a
    # follower has no usable ring at all the blend falls back to the identity,
    # which leaves it translating with the skin rather than guessing a rotation.
    rotation_weights = weights * ring_valid[neighbour_index]
    rotation_total = rotation_weights.sum(axis=1, keepdims=True)
    usable = (rotation_total > EPS).ravel()
    rotation_weights[usable] /= rotation_total[usable]
    rotation_weights[~usable] = 0.0

    return {
        "ring_index": ring_index,
        "ring_mask": ring_mask,
        "ring_valid": ring_valid,
        "skin_rest": skin_points,
        "follower_rest": follower_points,
        "neighbour_index": neighbour_index,
        "neighbour_weight": weights,
        "rotation_weight": rotation_weights,
    }


def _ring_rank_is_sufficient(points, ring_index, ring_mask, ratio=1e-4):
    """Whether each padded 1-ring spans at least a plane, vectorised.

    The squared singular values of the centred ring are the eigenvalues of its
    covariance, so this avoids one SVD per vertex on a production head.

    :param numpy.ndarray points: (P, 3) rest positions.
    :param numpy.ndarray ring_index: (P, R) padded ring table.
    :param numpy.ndarray ring_mask: (P, R) padding mask.
    :param float ratio: Minimum second/first singular value ratio.
    :return: (P,) boolean, True where the ring determines a rotation.
    :rtype: numpy.ndarray
    """
    mask = ring_mask[..., None]
    counts = ring_mask.sum(axis=1, keepdims=True)
    ring = points[ring_index]
    centred = (ring - ((ring * mask).sum(axis=1) / counts)[:, None, :]) * mask

    eigenvalues = numpy.linalg.eigvalsh(numpy.einsum("prk,prl->pkl", centred, centred))
    largest = eigenvalues[:, 2]
    middle = numpy.clip(eigenvalues[:, 1], 0.0, None)

    valid = largest > EPS
    result = numpy.zeros(len(points), dtype=float)
    result[valid] = numpy.sqrt(middle[valid] / largest[valid])
    return result >= ratio


def _local_rotations(binding, skin_deformed):
    """Per-skin-vertex rotation, fitted over each 1-ring.

    The 1-rings overlap, so the resulting field varies smoothly across the
    surface. That smoothness is the whole point: a per-vertex nearest-triangle
    rivet is discontinuous wherever adjacent vertices pick different triangles,
    which is what puts stair-stepping artefacts into thin strips like eyelashes.

    :param dict binding:
    :param numpy.ndarray skin_deformed: (P, 3)
    :return: Rotations (P, 3, 3).
    :rtype: numpy.ndarray
    """
    ring_index = binding["ring_index"]
    ring_mask = binding["ring_mask"][..., None]
    rest = binding["skin_rest"]

    counts = binding["ring_mask"].sum(axis=1, keepdims=True)
    rest_ring = rest[ring_index]
    deformed_ring = skin_deformed[ring_index]

    centre_rest = (rest_ring * ring_mask).sum(axis=1) / counts
    centre_deformed = (deformed_ring * ring_mask).sum(axis=1) / counts

    a = (rest_ring - centre_rest[:, None, :]) * ring_mask
    b = (deformed_ring - centre_deformed[:, None, :]) * ring_mask

    covariance = numpy.einsum("prk,prl->pkl", a, b)
    u, _, vt = numpy.linalg.svd(covariance)
    determinant = numpy.linalg.det(numpy.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
    flip = numpy.ones((len(covariance), 3))
    flip[:, 2] = numpy.where(determinant < 0.0, -1.0, 1.0)
    return numpy.matmul(vt.transpose(0, 2, 1) * flip[:, None, :], u.transpose(0, 2, 1))


def evaluate_binding(binding, skin_deformed, stiffness=1.0, shell_labels=None):
    """Derive follower positions from a deformed skin.

    :param dict binding: From :func:`build_binding`.
    :param numpy.ndarray skin_deformed: (P, 3) deformed skin positions, in the
        same local index space used to build the binding.
    :param stiffness: ``1.0`` keeps each shell perfectly rigid, ``0.0`` lets it
        deform smoothly with the skin, in between blends. Either a single value
        for every shell or a ``{shell label: value}`` mapping.
    :type stiffness: float | dict[int, float]
    :param shell_labels: (M,) per-follower shell id. When omitted the followers
        are treated as a single shell.
    :return: (M, 3) follower positions.
    :rtype: numpy.ndarray
    """
    skin_deformed = numpy.asarray(skin_deformed, dtype=float)
    follower_rest = binding["follower_rest"]
    neighbour_index = binding["neighbour_index"]
    weights = binding["neighbour_weight"][..., None]

    rotations = _local_rotations(binding, skin_deformed)

    # blend the neighbouring local frames, then re-orthogonalise. Blending
    # identical rotations returns that rotation untouched, which is what makes
    # the rest pose and rigid motion cases exact. Rings that cannot define a
    # rotation are already weighted out, and a follower with no usable ring at
    # all falls back to the identity.
    rotation_weight = binding.get("rotation_weight")
    if rotation_weight is None:
        rotation_weight = binding["neighbour_weight"]

    blended = (rotations[neighbour_index] * rotation_weight[..., None, None]).sum(axis=1)
    unusable = rotation_weight.sum(axis=1) <= EPS
    if numpy.any(unusable):
        blended[unusable] = numpy.eye(3)

    blended = project_to_rotation(blended)

    # the rest and deformed anchors must be the same weighted combination of the
    # same skin vertices, otherwise the rest pose would not round trip exactly
    anchor_rest = (binding["skin_rest"][neighbour_index] * weights).sum(axis=1)
    anchor_deformed = (skin_deformed[neighbour_index] * weights).sum(axis=1)

    smooth = numpy.einsum(
        "mij,mj->mi", blended, follower_rest - anchor_rest) + anchor_deformed

    shell_labels = (numpy.zeros(len(smooth), dtype=numpy.int64)
                    if shell_labels is None else numpy.asarray(shell_labels))

    out = smooth.copy()
    for label in numpy.unique(shell_labels):
        amount = (float(stiffness.get(int(label), 1.0))
                  if isinstance(stiffness, dict) else float(stiffness))
        amount = min(max(amount, 0.0), 1.0)
        if amount <= 0.0:
            continue

        indices = numpy.nonzero(shell_labels == label)[0]
        rest = follower_rest[indices]
        if len(indices) >= 3 and not is_degenerate(rest):
            r, t = kabsch(rest, smooth[indices])
            rigid = rest.dot(r.T) + t
        else:
            # A single row strip, two vertices, or a perfectly collinear shell
            # cannot be fitted by Kabsch, its own points leave the rotation about
            # their shared axis undetermined. The skin underneath does define
            # that rotation though, so take it from the blended field rather
            # than dropping it, which would leave a lash strip facing its rest
            # direction while the head turns.
            log.debug("Shell %s cannot be fitted from its own points, taking the "
                      "rotation from the skin.", int(label))
            r = project_to_rotation(blended[indices].mean(axis=0))
            centre = rest.mean(axis=0)
            rigid = (rest - centre).dot(r.T) + smooth[indices].mean(axis=0)

        out[indices] = (rigid if amount >= 1.0
                        else amount * rigid + (1.0 - amount) * smooth[indices])

    return out


# ----------------------------------------------------------------------------
# authored offset
# ----------------------------------------------------------------------------
def transplant_source_offset(followed, predicted, authored, rest, shell_labels):
    """Re-apply the source's authored motion that pure following cannot explain.

    Pure following gives a shell the motion implied by the skin underneath it.
    A modeller may deliberately have moved a brow card further than the skin, and
    that intent is lost by following alone. This measures the leftover rigid
    motion on the source and transplants it onto the target result, rotating the
    shell about its own centre so no deformation is introduced. Translation is
    scaled by the shell's target/source size ratio so it respects proportions.

    :param numpy.ndarray followed: (M, 3) target positions from following.
    :param numpy.ndarray predicted: (M, 3) source positions from following.
    :param numpy.ndarray authored: (M, 3) source positions as actually modelled.
    :param numpy.ndarray rest: (M, 3) source follower rest positions.
    :param numpy.ndarray shell_labels: (M,) per-follower shell id.
    :return: (M, 3) corrected target positions.
    :rtype: numpy.ndarray
    """
    out = numpy.array(followed, dtype=float, copy=True)
    shell_labels = numpy.asarray(shell_labels)

    for label in numpy.unique(shell_labels):
        indices = numpy.nonzero(shell_labels == label)[0]
        shell_rest = rest[indices]
        if len(indices) < 3 or is_degenerate(shell_rest):
            out[indices] += (authored[indices] - predicted[indices]).mean(axis=0)
            continue

        r_predicted, _ = kabsch(shell_rest, predicted[indices])
        r_authored, _ = kabsch(shell_rest, authored[indices])

        # The leftover motion is the authored motion with the followed motion
        # divided out. Splitting it into a rotation and a centroid displacement
        # keeps it independent of where either head sits in space, so it can be
        # replayed on a target of different proportions.
        rotation = r_authored.dot(r_predicted.T)
        displacement = authored[indices].mean(axis=0) - predicted[indices].mean(axis=0)

        # a displacement authored on the source is scaled into the target's
        # proportions using the shell's own size
        source_size = numpy.linalg.norm(shell_rest - shell_rest.mean(axis=0), axis=1).mean()
        target_size = numpy.linalg.norm(
            followed[indices] - followed[indices].mean(axis=0), axis=1).mean()
        # EPS is absolute, so a shell that was small but not collapsed passed
        # straight through it and scaled the displacement by the full ratio: a
        # source card a thousandth of the target's size multiplied the authored
        # motion by a thousand. An exactly collapsed shell took the degenerate
        # branch above and came out right, a nearly collapsed one did not.
        if source_size > EPS and target_size > EPS:
            ratio = target_size / source_size
            if 1.0 / MAX_SIZE_RATIO <= ratio <= MAX_SIZE_RATIO:
                displacement = displacement * ratio
            else:
                log.warning("Shell %s has a target/source size ratio of %.3g, which is "
                            "outside the plausible range. Transplanting its authored "
                            "displacement unscaled.", label, ratio)

        centre = followed[indices].mean(axis=0)
        out[indices] = ((followed[indices] - centre).dot(rotation.T)
                        + centre + displacement)

    return out
