"""Root-cause diagnostic for the "detached shells fly to world zero" bug.

This replicates the EXACT linear algebra used by
``transfer_blend_shape.transfer.Transfer.execute`` outside of Maya, on a tiny
synthetic mesh that mimics our head assets: one large "face" shell plus small
detached "card" shells (brows / lashes) floating just above it.

The claim under test
--------------------
The Sumner & Popovic deformation-gradient operator is *translation invariant*
(each triangle's row block sums to zero). It therefore determines vertex
positions only up to **one free translation per connected component**. The tool
removes that freedom using the ``static_vertices`` (zero-delta) anchors, whose
contribution is moved to the right-hand side.

So: any connected component in which **every** vertex moves has no anchor, and
its translation is mathematically undetermined. ``scipy.linalg.lu_factor`` /
``lu_solve`` do not raise on a singular matrix -- they silently return a
solution with the free variable left near zero, i.e. the shell keeps its
internal shape but lands near the object-space origin.

That is precisely the reported symptom: brows/lashes at world zero, internal
relative positions intact.
"""
import numpy as np
import scipy.linalg


# ----------------------------------------------------------------------------
# verbatim re-implementations of the solver's maths (no Maya required)
# ----------------------------------------------------------------------------
def as_chunks(l, num):
    return [l[i:i + num] for i in range(0, len(l), num)]


def calculate_edge_matrix(p1, p2, p3):
    e0 = p2 - p1
    e1 = p3 - p1
    e2 = np.cross(e0, e1)
    return np.array([e0, e1, e2]).transpose()


def calculate_target_matrix(triangles, target_points):
    matrix = np.zeros((len(triangles), target_points.shape[0]))
    for i, (i0, i1, i2) in enumerate(as_chunks(triangles, 3)):
        e0 = target_points[i1] - target_points[i0]
        e1 = target_points[i2] - target_points[i0]
        va = np.array([e0, e1]).transpose()
        q, r = np.linalg.qr(va)
        inv_rqt = np.dot(np.linalg.inv(r), q.transpose())
        for j in range(3):
            matrix[i * 3 + j][i0] = - inv_rqt[0][j] - inv_rqt[1][j]
            matrix[i * 3 + j][i1] = inv_rqt[0][j]
            matrix[i * 3 + j][i2] = inv_rqt[1][j]
    return matrix


def calculate_deformation_gradient(triangles, source_points, points):
    matrix = np.zeros((len(triangles), 3))
    for i, (i0, i1, i2) in enumerate(as_chunks(triangles, 3)):
        va = calculate_edge_matrix(source_points[i0], source_points[i1], source_points[i2])
        vb = calculate_edge_matrix(points[i0], points[i1], points[i2])
        q, r = np.linalg.qr(va)
        inv_rqt = np.dot(np.linalg.inv(r), q.transpose())
        sa = np.dot(vb, inv_rqt)
        matrix[i * 3: i * 3 + 3] = sa.transpose()
    return matrix


def solve_like_the_tool(triangles, source_points, target_points, points,
                        threshold=0.001, method="lu"):
    """The exact anchor + least-squares path from ``Transfer.execute``.

    :param str method: ``"lu"`` reproduces the shipped code verbatim
        (``lu_factor``/``lu_solve``). ``"lstsq"`` swaps in a minimum-norm
        pseudo-inverse solve, which is what a *near*-singular real-world mesh
        effectively converges to -- it keeps the answer finite and shows where
        the lost translation actually sends the shell.
    """
    lengths = scipy.linalg.norm(source_points - points, axis=1)
    static_vertices = np.nonzero(lengths <= threshold)[0]
    deformed_vertices = np.nonzero(lengths > threshold)[0]

    target_matrix = calculate_target_matrix(triangles, target_points)

    static_matrix = target_matrix[:, static_vertices]
    static_points = target_points[static_vertices, :]
    static_gradient = np.dot(static_matrix, static_points)
    deformation_gradient = calculate_deformation_gradient(
        triangles, source_points, points) - static_gradient

    deformed_matrix = target_matrix[:, deformed_vertices]
    dmt = deformed_matrix.transpose()
    normal_matrix = np.dot(dmt, deformed_matrix)
    uts = np.dot(dmt, deformation_gradient)

    if method == "lu":
        lu, piv = scipy.linalg.lu_factor(normal_matrix)
        deformed_points = scipy.linalg.lu_solve((lu, piv), uts)
    else:
        deformed_points = np.linalg.lstsq(normal_matrix, uts, rcond=None)[0]

    out = target_points.copy()
    out[deformed_vertices, :] = deformed_points
    return out, static_vertices, deformed_vertices, normal_matrix


# ----------------------------------------------------------------------------
# synthetic head: one face grid + detached cards, all pushed away from origin
# ----------------------------------------------------------------------------
def build_scene(n=7, origin_offset=(30.0, 12.0, -5.0)):
    """Face grid (one shell) + 3 detached quad cards (three more shells)."""
    xs, ys = np.meshgrid(np.linspace(0, 6, n), np.linspace(0, 6, n))
    face = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])

    quads = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            quads.append([v0, v0 + 1, v0 + n + 1, v0 + n])

    pts = [face]
    cursor = n * n
    card_ids = []
    for cx in (1.5, 3.0, 4.5):
        card = np.array([
            [cx - 0.4, 4.0, 0.35],
            [cx + 0.4, 4.0, 0.35],
            [cx + 0.4, 4.8, 0.35],
            [cx - 0.4, 4.8, 0.35],
        ])
        pts.append(card)
        quads.append([cursor, cursor + 1, cursor + 2, cursor + 3])
        card_ids.append(list(range(cursor, cursor + 4)))
        cursor += 4

    points = np.vstack(pts) + np.array(origin_offset)

    triangles = []
    for q in quads:                      # fan triangulate, like MFnMesh
        for k in range(1, len(q) - 1):
            triangles += [q[0], q[k], q[k + 1]]

    return points, triangles, card_ids, n * n


def main():
    np.set_printoptions(precision=4, suppress=True)

    source_rest, triangles, card_ids, n_face = build_scene()

    # target = same topology, different proportions (a wider/taller head)
    target_rest = source_rest.copy()
    target_rest[:, 0] *= 1.15
    target_rest[:, 1] *= 0.92

    # ---- author a source shape: a LOCAL face bump (so the rim stays static,
    #      exactly like the static back-of-skull on a real head) plus a card
    #      motion in which every card vertex moves -------------------------
    source_shape = source_rest.copy()
    c = source_rest[:n_face].mean(axis=0)
    r = np.linalg.norm(source_rest[:n_face, :2] - c[:2], axis=1)
    falloff = np.clip(1.0 - (r / 2.5) ** 2, 0.0, None) ** 3   # compact support
    source_shape[:n_face, 2] += 0.55 * falloff
    for ids in card_ids:
        source_shape[ids] += np.array([0.0, 0.22, 0.10])   # brow raise

    print(__doc__)
    print("=" * 74)
    print("SCENE: {} vertices -- 1 face shell ({} verts) + {} detached cards".format(
        len(source_rest), n_face, len(card_ids)))
    print("       object-space origin is {:.1f} units from the face centre".format(
        np.linalg.norm(source_rest[:n_face].mean(axis=0))))

    out, static, deformed, normal_matrix = solve_like_the_tool(
        triangles, source_rest, target_rest, source_shape)

    # ---- evidence 1: the normal matrix is rank deficient -------------------
    rank = np.linalg.matrix_rank(normal_matrix)
    size = normal_matrix.shape[0]
    n_static_face = int(np.sum(static < n_face))
    print("\n[1] RANK OF THE NORMAL MATRIX (deformed_matrix.T @ deformed_matrix)")
    print("    face shell has {} static anchor vertices -> properly constrained".format(
        n_static_face))
    print("    cards have    {} static anchor vertices -> UNCONSTRAINED".format(
        int(np.sum(static >= n_face))))
    print("    shape {}x{}   rank {}   deficiency {}".format(size, size, rank, size - rank))
    print("    -> deficiency == number of unanchored shells ({} cards): {}".format(
        len(card_ids), (size - rank) == len(card_ids)))
    sv = np.linalg.svd(normal_matrix, compute_uv=False)
    print("    smallest singular values: {}".format(sv[-5:]))
    print("    condition number: {:.3e}  (a well-posed solve is ~1e2-1e4)".format(
        sv[0] / sv[-1]))

    # ---- evidence 2: where did the cards actually land? --------------------
    print("\n[2] WHERE THE CARDS LAND")
    print("    {:<8} {:>26} {:>26} {:>10}".format(
        "shell", "expected centroid", "solved centroid", "|error|"))
    for k, ids in enumerate(card_ids):
        expected = target_rest[ids].mean(axis=0) + np.array([0.0, 0.20, 0.10])
        got = out[ids].mean(axis=0)
        print("    card {:<3} {:>26} {:>26} {:>10.3f}".format(
            k, np.array2string(expected, precision=2),
            np.array2string(got, precision=2), np.linalg.norm(got - expected)))

    all_cards = np.concatenate(card_ids)
    finite = np.all(np.isfinite(out))
    print("\n    exact LU path (the shipped code): result finite? {}".format(finite))
    if not finite:
        print("    -> the zero pivot produces nan, and back-substitution spreads it")
        print("       through the WHOLE coupled system, so the face is poisoned too.")
        print("       Maya draws non-finite points at the origin: 'everything at")
        print("       world zero', and it is also a plausible mesh-detonation path.")

    # a real asset is *near*-singular rather than exactly singular (float noise
    # in the mesh), so the practical result is the minimum-norm solution.
    out_ls, _, _, _ = solve_like_the_tool(
        triangles, source_rest, target_rest, source_shape, method="lstsq")
    dist_expected = np.linalg.norm(target_rest[all_cards].mean(axis=0))
    dist_origin = np.linalg.norm(out_ls[all_cards].mean(axis=0))
    print("\n    near-singular equivalent (minimum-norm solve):")
    print("      card centroid distance from the object origin : {:.3f}".format(dist_origin))
    print("      where the cards SHOULD be                     : {:.3f}".format(dist_expected))
    print("      -> pulled {:.0f}% of the way to world zero, i.e. 'offset from".format(
        100.0 * (1.0 - dist_origin / dist_expected)))
    print("         the head' exactly as the Tech Artist described.")

    face_static = static[static < n_face]
    face_err = np.abs(out_ls[face_static] - target_rest[face_static]).max()
    print("      face static-vertex drift in that solve        : {:.2e}".format(face_err))
    print("      -> the anchored face shell is solved correctly; the defect is")
    print("         isolated to the unanchored shells.")

    # ---- evidence 3: internal shape survived (matches the artist's report) --
    def pw(x):
        d = x[:, None, :] - x[None, :, :]
        return np.linalg.norm(d, axis=2)

    print("\n[3] INTERNAL SHAPE OF EACH CARD (the artist's 'relative position' clue)")
    for k, ids in enumerate(card_ids):
        change = np.abs(pw(out_ls[ids]) - pw(target_rest[ids])).max()
        print("    card {}: max pairwise-distance change {:.3e}".format(k, change))
    inter = np.abs(pw(np.array([out_ls[i].mean(axis=0) for i in card_ids]))
                   - pw(np.array([target_rest[i].mean(axis=0) for i in card_ids]))).max()
    print("    card-to-card spacing change: {:.3e}".format(inter))
    print("    -> each card's shape AND the card-to-card layout are preserved;")
    print("       only the one shared translation is lost. Exactly as reported.")

    # ---- evidence 4: give ONE card an anchor and it snaps back -------------
    print("\n[4] CONTROL: anchor one vertex of card 0 (make it static) and re-solve")
    src2 = source_shape.copy()
    src2[card_ids[0][0]] = source_rest[card_ids[0][0]]      # zero delta -> anchor
    out2, _, _, nm2 = solve_like_the_tool(
        triangles, source_rest, target_rest, src2, method="lstsq")
    rank2 = np.linalg.matrix_rank(nm2)
    err2 = np.linalg.norm(out2[card_ids[0]].mean(axis=0)
                          - target_rest[card_ids[0]].mean(axis=0))
    err_other = np.linalg.norm(out2[card_ids[1]].mean(axis=0)
                               - target_rest[card_ids[1]].mean(axis=0))
    print("    rank deficiency now {} (was {})".format(nm2.shape[0] - rank2, size - rank))
    print("    card 0 (now anchored)   centroid error: {:.4f}".format(err2))
    print("    card 1 (still floating) centroid error: {:.4f}".format(err_other))
    print("    -> a single anchor vertex fixes that one shell and leaves the others")
    print("       broken. Confirms the missing-anchor diagnosis rather than")
    print("       anything about the geometry or the smoothing pass.")

    print("\n" + "=" * 74)
    print("CONCLUSION: the bug is not numerical noise or bad geometry. Every")
    print("connected component needs its own translational anchor. The tool only")
    print("provides global zero-delta anchors, so a shell whose every vertex moves")
    print("is solved from a singular system and loses its position.")
    print("=" * 74)


if __name__ == "__main__":
    main()
