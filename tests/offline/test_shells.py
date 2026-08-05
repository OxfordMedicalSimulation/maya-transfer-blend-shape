"""Offline regression tests for detached shell support.

These drive the real ``transfer_blend_shape.transfer.Transfer`` numerical
pipeline through Maya stubs, so they cover the shipped code rather than a
reimplementation. Run with::

    python tests/offline/test_shells.py

The synthetic asset mimics a head: one large skin shell plus small planar
"eyebrow cards" and a curved multi-row "eyelash strip", all placed far from the
object origin so a lost translation is unmistakable.
"""
import sys
import unittest

import numpy

import mayastub

mayastub.install()

from transfer_blend_shape.utils import shell           # noqa: E402


ORIGIN_OFFSET = numpy.array([30.0, 12.0, -5.0])


# ----------------------------------------------------------------------------
# synthetic asset
# ----------------------------------------------------------------------------
def grid(rows, columns, width, height):
    xs, ys = numpy.meshgrid(numpy.linspace(0.0, width, columns),
                            numpy.linspace(0.0, height, rows))
    points = numpy.column_stack([xs.ravel(), ys.ravel(), numpy.zeros(rows * columns)])
    quads = []
    for j in range(rows - 1):
        for i in range(columns - 1):
            v0 = j * columns + i
            quads.append([v0, v0 + 1, v0 + columns, v0 + columns - columns + columns])
            quads[-1] = [v0, v0 + 1, v0 + columns + 1, v0 + columns]
    return points, quads


class Asset(object):
    """A head-like mesh: skin + N brow cards + one curved lash strip."""

    def __init__(self, num_cards=3, skin_resolution=9):
        points, quads = grid(skin_resolution, skin_resolution, 6.0, 6.0)
        self.skin = numpy.arange(len(points))
        all_points = [points]
        cursor = len(points)
        self.cards = []

        # planar brow cards, each its own shell, each only four vertices
        for index in range(num_cards):
            cx = 1.0 + index * 1.6
            card = numpy.array([
                [cx - 0.45, 4.6, 0.35],
                [cx + 0.45, 4.6, 0.35],
                [cx + 0.45, 5.2, 0.35],
                [cx - 0.45, 5.2, 0.35],
            ])
            all_points.append(card)
            quads.append([cursor, cursor + 1, cursor + 2, cursor + 3])
            self.cards.append(numpy.arange(cursor, cursor + 4))
            cursor += 4

        # a curved two row lash strip, deliberately sliver heavy
        columns = 12
        xs = numpy.linspace(0.8, 5.2, columns)
        curve = 0.25 * numpy.sin(numpy.linspace(0.0, numpy.pi, columns))
        row0 = numpy.column_stack([xs, 2.0 + curve, numpy.full(columns, 0.22)])
        row1 = numpy.column_stack([xs, 2.0 + curve, numpy.full(columns, 0.24)])
        lash = numpy.vstack([row0, row1])
        all_points.append(lash)
        for i in range(columns - 1):
            quads.append([cursor + i, cursor + i + 1,
                          cursor + columns + i + 1, cursor + columns + i])
        self.lash = numpy.arange(cursor, cursor + 2 * columns)
        cursor += 2 * columns

        self.points = numpy.vstack(all_points) + ORIGIN_OFFSET
        self.quads = quads

        self.triangles = []
        for quad in quads:
            for k in range(1, len(quad) - 1):
                self.triangles += [quad[0], quad[k], quad[k + 1]]

        connectivity = [set() for _ in range(len(self.points))]
        for quad in quads:
            for i, v in enumerate(quad):
                connectivity[v].add(quad[(i + 1) % len(quad)])
                connectivity[v].add(quad[(i - 1) % len(quad)])
        self.connectivity = [sorted(item) for item in connectivity]

    @property
    def followers(self):
        return numpy.sort(numpy.concatenate(self.cards + [self.lash]))

    def target(self):
        """Different proportions, same topology."""
        points = self.points.copy()
        points[:, 0] *= 1.18
        points[:, 1] *= 0.90
        return points

    def shape(self, card_delta=(0.0, 0.24, 0.10), lash_delta=(0.0, 0.10, 0.03)):
        """A brow-raise style shape where every card and lash vertex moves."""
        points = self.points.copy()
        skin = self.points[self.skin]
        centre = skin.mean(axis=0)
        r = numpy.linalg.norm(skin[:, :2] - centre[:2], axis=1)
        falloff = numpy.clip(1.0 - (r / 3.0) ** 2, 0.0, None) ** 3
        points[self.skin, 2] += 0.6 * falloff
        points[self.skin, 1] += 0.25 * falloff
        for ids in self.cards:
            points[ids] += numpy.array(card_delta)
        points[self.lash] += numpy.array(lash_delta)
        return points

    def transfer(self, **kwargs):
        return mayastub.OfflineTransfer(
            self.points, self.target(), self.connectivity, self.triangles, **kwargs)


def pairwise(points):
    delta = points[:, None, :] - points[None, :, :]
    return numpy.linalg.norm(delta, axis=2)


def roughness(displacement):
    """Second difference of a displacement field along a vertex run.

    Smooth deformation shows small values. An artefact, a vertex that jumps
    relative to its neighbours, shows up as a large one. Only comparable between
    fields sampled on the *same* vertices, since the measure scales with spacing.
    """
    return float(numpy.abs(numpy.diff(displacement, n=2, axis=0)).max())


def nearest_triangle_rivet(follower_rest, skin_rest, skin_deformed, triangles):
    """A per-vertex rivet, the naive approach this fix replaces.

    Each follower vertex is attached to its nearest skin triangle and carried by
    that triangle's local frame. Adjacent vertices can pick different triangles,
    and the frames of neighbouring triangles are not continuous, so the field
    gains steps that read as stair-stepping in a thin strip. Kept in the tests
    purely as the baseline the smooth field has to beat.
    """
    triangles = numpy.asarray(triangles).reshape(-1, 3)
    centroids = skin_rest[triangles].mean(axis=1)
    out = numpy.zeros_like(follower_rest)

    for i, point in enumerate(follower_rest):
        tri = triangles[numpy.argmin(numpy.linalg.norm(centroids - point, axis=1))]

        def frame(points):
            a, b, c = points
            normal = numpy.cross(b - a, c - a)
            normal = normal / max(numpy.linalg.norm(normal), 1e-12)
            e1 = b - a
            e1 = e1 / max(numpy.linalg.norm(e1), 1e-12)
            return a, e1, numpy.cross(normal, e1), normal

        a, e1, e2, normal = frame(skin_rest[tri])
        local = numpy.array([(point - a).dot(e1), (point - a).dot(e2), (point - a).dot(normal)])
        a2, f1, f2, normal2 = frame(skin_deformed[tri])
        out[i] = a2 + local[0] * f1 + local[1] * f2 + local[2] * normal2

    return out


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
class TestShellMaths(unittest.TestCase):
    """The geometric guarantees the follow model rests on."""

    @staticmethod
    def rotation(seed):
        rng = numpy.random.default_rng(seed)
        q, _ = numpy.linalg.qr(rng.standard_normal((3, 3)))
        if numpy.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        return q

    def test_connected_components(self):
        asset = Asset()
        labels, order = shell.connected_components(asset.connectivity)
        self.assertEqual(len(order), 1 + len(asset.cards) + 1)
        self.assertEqual(len(numpy.nonzero(labels == order[0])[0]), len(asset.skin))
        # the largest shell must be the skin
        for label in order[1:]:
            self.assertLess(len(numpy.nonzero(labels == label)[0]), len(asset.skin))

    def test_kabsch_recovers_rigid_transform(self):
        rng = numpy.random.default_rng(3)
        a = rng.standard_normal((40, 3))
        r = self.rotation(11)
        t = numpy.array([4.0, -2.0, 7.0])
        got_r, got_t = shell.kabsch(a, a.dot(r.T) + t)
        numpy.testing.assert_allclose(got_r, r, atol=1e-10)
        numpy.testing.assert_allclose(got_t, t, atol=1e-10)
        self.assertAlmostEqual(float(numpy.linalg.det(got_r)), 1.0, places=10)

    def test_kabsch_rejects_reflection(self):
        a = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        r, _ = shell.kabsch(a, a * numpy.array([1.0, 1.0, -1.0]))
        self.assertGreater(float(numpy.linalg.det(r)), 0.0)

    def test_degenerate_detection(self):
        line = numpy.column_stack([numpy.linspace(0, 1, 8), numpy.zeros(8), numpy.zeros(8)])
        self.assertTrue(shell.is_degenerate(line))
        plane = numpy.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertFalse(shell.is_degenerate(plane))

    def test_rest_pose_round_trips_exactly(self):
        asset = Asset()
        skin = asset.points[asset.skin]
        binding = shell.build_binding(
            asset.points[asset.followers], skin,
            [[j for j in asset.connectivity[i]] for i in asset.skin])
        for stiffness in (0.0, 0.5, 1.0):
            out = shell.evaluate_binding(binding, skin, stiffness=stiffness)
            numpy.testing.assert_allclose(
                out, asset.points[asset.followers], atol=1e-9,
                err_msg="rest pose must round trip at stiffness {}".format(stiffness))

    def test_rigid_skin_motion_is_followed_exactly(self):
        asset = Asset()
        skin = asset.points[asset.skin]
        binding = shell.build_binding(
            asset.points[asset.followers], skin,
            [[j for j in asset.connectivity[i]] for i in asset.skin])

        r = self.rotation(7)
        t = numpy.array([-3.0, 8.0, 1.5])
        expected = asset.points[asset.followers].dot(r.T) + t
        for stiffness in (0.0, 0.5, 1.0):
            out = shell.evaluate_binding(binding, skin.dot(r.T) + t, stiffness=stiffness)
            numpy.testing.assert_allclose(
                out, expected, atol=1e-7,
                err_msg="rigid motion must carry followers at stiffness {}".format(stiffness))


class TestWorldZeroRegression(unittest.TestCase):
    """The reported defect and the behaviour the Tech Artist asked for."""

    def setUp(self):
        self.asset = Asset()
        self.shape = self.asset.shape()

    def solve(self, **kwargs):
        t = self.asset.transfer(**kwargs)
        points, _, _ = t.calculate_points(self.shape, "browRaise")
        return t, points

    def test_shells_are_detected(self):
        t, _ = self.solve()
        self.assertEqual(len(t.get_shells()), 1 + len(self.asset.cards) + 1)
        numpy.testing.assert_array_equal(t.get_skin_vertices(), self.asset.skin)
        numpy.testing.assert_array_equal(t.get_follower_vertices(), self.asset.followers)

    def test_followers_stay_on_the_head(self):
        """The regression: shells used to lose their translation entirely."""
        _, points = self.solve()
        self.assertTrue(numpy.all(numpy.isfinite(points)))

        target = self.asset.target()
        for name, ids in [("card", self.asset.cards[0]), ("lash", self.asset.lash)]:
            moved = numpy.linalg.norm(points[ids].mean(axis=0) - target[ids].mean(axis=0))
            from_origin = numpy.linalg.norm(points[ids].mean(axis=0))
            self.assertLess(moved, 1.0, "{} moved {:.3f} units, expected a small "
                                        "follow".format(name, moved))
            self.assertGreater(from_origin, 20.0,
                               "{} collapsed towards the object origin".format(name))

    def test_legacy_path_still_reproduces_the_bug(self):
        """Guards the diagnosis: the old behaviour must still fail this way."""
        t = self.asset.transfer(solve_detached_shells=True)
        with self.assertRaises(RuntimeError) as context:
            t.calculate_points(self.shape, "browRaise")

        self.assertIn("singular", str(context.exception).lower())

    def test_brow_cards_stay_rigid(self):
        """Stiffness 1 preserves card shape exactly while it tracks the skin."""
        _, points = self.solve(shell_stiffness=1.0)
        target = self.asset.target()
        for index, ids in enumerate(self.asset.cards):
            error = numpy.abs(pairwise(points[ids]) - pairwise(target[ids])).max()
            self.assertLess(error, 1e-9, "card {} deformed by {:.3e}".format(index, error))
            # and it did actually move
            self.assertGreater(
                numpy.linalg.norm(points[ids].mean(axis=0) - target[ids].mean(axis=0)), 1e-4)

    def test_soft_shells_deform_but_do_not_explode(self):
        """Stiffness 0 lets a strip bend along the lid without tearing."""
        _, points = self.solve(shell_stiffness=0.0)
        target = self.asset.target()
        lash = self.asset.lash[:12]

        edge_rest = numpy.linalg.norm(numpy.diff(target[lash], axis=0), axis=1)
        edge_out = numpy.linalg.norm(numpy.diff(points[lash], axis=0), axis=1)
        stretch = numpy.abs(edge_out / edge_rest - 1.0).max()
        self.assertLess(stretch, 0.35, "lash edges stretched by {:.1%}".format(stretch))

    def test_smooth_follow_is_smoother_than_a_nearest_triangle_rivet(self):
        """The artefact fix, measured against the approach that causes them.

        Both fields are sampled on the identical vertices, so their roughness is
        directly comparable. An absolute threshold would be arbitrary; beating
        the naive method by a wide margin is the claim that actually matters.
        """
        _, points = self.solve(shell_stiffness=0.0)
        target = self.asset.target()
        transfer = self.asset.transfer()
        skin = transfer.get_skin_vertices()
        lash = self.asset.lash[:12]

        # skin triangles, remapped into the skin's local vertex order
        skin_set = set(skin.tolist())
        remap = {v: i for i, v in enumerate(skin)}
        local = []
        for k in range(0, len(self.asset.triangles), 3):
            tri = self.asset.triangles[k:k + 3]
            if all(v in skin_set for v in tri):
                local += [remap[v] for v in tri]

        rivet = nearest_triangle_rivet(target[lash], target[skin], points[skin], local)

        smooth_roughness = roughness(points[lash] - target[lash])
        rivet_roughness = roughness(rivet - target[lash])

        self.assertGreater(
            rivet_roughness, smooth_roughness * 3.0,
            "smooth follow ({:.4f}) should be far smoother than a per-vertex rivet "
            "({:.4f})".format(smooth_roughness, rivet_roughness))

        # and it must not add roughness beyond the skin it is following
        band = skin[numpy.argsort(
            numpy.abs(target[skin][:, 1] - target[lash].mean(axis=0)[1]))[:9]]
        band = band[numpy.argsort(target[band][:, 0])]
        self.assertLess(smooth_roughness, roughness(points[band] - target[band]),
                        "follower field is rougher than the skin driving it")

    def test_per_shell_stiffness_override(self):
        t = self.asset.transfer(shell_stiffness=1.0)
        lash_shell = None
        for index, vertices in enumerate(t.get_shells()):
            if len(vertices) == len(self.asset.lash):
                lash_shell = index

        self.assertIsNotNone(lash_shell)
        t.set_shell_stiffness(0.0, index=lash_shell)
        self.assertEqual(t.get_shell_stiffness(lash_shell), 0.0)

        # every other follower keeps the default. Shells are ordered by vertex
        # count, so the lash strip sorts above the four vertex cards.
        card_shells = [index for index, vertices in enumerate(t.get_shells())
                       if index and len(vertices) == 4]
        self.assertTrue(card_shells)
        for index in card_shells:
            self.assertEqual(t.get_shell_stiffness(index), 1.0)

        points, _, _ = t.calculate_points(self.shape, "browRaise")
        target = self.asset.target()
        # the cards remain rigid, the lash is allowed to deform
        card_error = numpy.abs(pairwise(points[self.asset.cards[0]])
                               - pairwise(target[self.asset.cards[0]])).max()
        self.assertLess(card_error, 1e-9)

    def test_static_shape_short_circuits(self):
        t = self.asset.transfer()
        points, weights, deformed = t.calculate_points(self.asset.points.copy(), "rest")
        self.assertIsNone(points)
        self.assertIsNone(weights)
        self.assertEqual(len(deformed), 0)

    def test_identity_transfer_reproduces_the_source(self):
        """Source and target identical, so the answer is known exactly."""
        asset = Asset()
        t = mayastub.OfflineTransfer(
            asset.points, asset.points.copy(), asset.connectivity, asset.triangles,
            iterations=0, preserve_source_offset=True, shell_stiffness=1.0)
        shape = asset.shape()
        points, _, _ = t.calculate_points(shape, "browRaise")

        # rigid card motion is fully explained, so it must be reproduced exactly
        for index, ids in enumerate(asset.cards):
            numpy.testing.assert_allclose(
                points[ids], shape[ids], atol=1e-7,
                err_msg="card {} not reproduced under identity transfer".format(index))

    def test_source_offset_is_off_by_default(self):
        t = self.asset.transfer()
        self.assertFalse(t.preserve_source_offset)

    def test_shell_description_drives_the_interface(self):
        t = self.asset.transfer()
        description = t.get_shell_description()

        self.assertEqual(len(description), len(t.get_shells()))
        self.assertEqual(description[0]["role"], "skin")
        self.assertIsNone(description[0]["stiffness"])
        self.assertEqual(description[0]["vertices"], len(self.asset.skin))

        for item in description[1:]:
            self.assertEqual(item["role"], "follower")
            self.assertEqual(item["stiffness"], 1.0)
            self.assertGreater(item["size"], 0.0)
            self.assertEqual(len(item["centre"]), 3)

        # shells are ordered largest first, which is what the skin lookup assumes
        counts = [item["vertices"] for item in description]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_setters_validate_their_input(self):
        t = self.asset.transfer()
        for value in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                t.set_shell_stiffness(value)
        for value in ("rigid", None, True):
            with self.assertRaises(TypeError):
                t.set_shell_stiffness(value)
        with self.assertRaises(ValueError):
            t.set_shell_neighbours(0)
        with self.assertRaises(TypeError):
            t.set_preserve_source_offset(1)

    def test_neighbour_count_changes_nothing_at_rest(self):
        """More neighbours widens the field but must not break exactness."""
        for num in (4, 8, 16):
            t = self.asset.transfer(shell_neighbours=num)
            points, _, _ = t.calculate_points(self.shape, "browRaise")
            self.assertTrue(numpy.all(numpy.isfinite(points)))
            for ids in self.asset.cards:
                error = numpy.abs(pairwise(points[ids])
                                  - pairwise(self.asset.target()[ids])).max()
                self.assertLess(error, 1e-9,
                                "cards must stay rigid at {} neighbours".format(num))


class TestNumericalRobustness(unittest.TestCase):
    """The sliver driven failure modes that produced the lash artefacts."""

    def test_degenerate_areas_do_not_poison_smoothing_weights(self):
        asset = Asset()
        # collapse a lash quad so its vertices have exactly zero area
        points = asset.points.copy()
        points[asset.lash[3]] = points[asset.lash[2]]
        points[asset.lash[15]] = points[asset.lash[14]]

        t = mayastub.OfflineTransfer(
            points, points * 1.05, asset.connectivity, asset.triangles)
        weights = t.calculate_laplacian_weights(asset.shape(), numpy.array([0]))

        self.assertTrue(numpy.all(numpy.isfinite(weights)),
                        "degenerate triangles produced non-finite smoothing weights")

    def test_solve_reports_singularity_instead_of_returning_nan(self):
        asset = Asset()
        t = asset.transfer(solve_detached_shells=True)
        with self.assertRaises(RuntimeError):
            t.calculate_points(asset.shape(), "browRaise")

    def test_no_static_skin_vertices_raises_clearly(self):
        asset = Asset()
        shape = asset.shape()
        # move every skin vertex so the skin itself has no anchor
        shape[asset.skin] += numpy.array([0.0, 0.0, 0.5])
        t = asset.transfer()
        with self.assertRaises(RuntimeError) as context:
            t.calculate_points(shape, "browRaise")

        self.assertIn("static", str(context.exception).lower())

    def test_single_shell_mesh_is_unaffected(self):
        """Backwards compatibility: a one piece mesh must behave as before."""
        points, quads = grid(9, 9, 6.0, 6.0)
        points = points + ORIGIN_OFFSET
        triangles = []
        for quad in quads:
            for k in range(1, len(quad) - 1):
                triangles += [quad[0], quad[k], quad[k + 1]]
        connectivity = [set() for _ in range(len(points))]
        for quad in quads:
            for i, v in enumerate(quad):
                connectivity[v].add(quad[(i + 1) % len(quad)])
                connectivity[v].add(quad[(i - 1) % len(quad)])
        connectivity = [sorted(item) for item in connectivity]

        target = points.copy()
        target[:, 0] *= 1.1
        t = mayastub.OfflineTransfer(points, target, connectivity, triangles)
        self.assertEqual(len(t.get_shells()), 1)
        self.assertEqual(len(t.get_follower_vertices()), 0)
        self.assertIsNone(t.get_target_shell_binding())

        shape = points.copy()
        centre = points.mean(axis=0)
        r = numpy.linalg.norm(points[:, :2] - centre[:2], axis=1)
        shape[:, 2] += 0.5 * numpy.clip(1.0 - (r / 3.0) ** 2, 0.0, None) ** 3
        out, _, _ = t.calculate_points(shape, "bump")
        self.assertTrue(numpy.all(numpy.isfinite(out)))


class TestMemoizeIdentity(unittest.TestCase):
    """The cache key must not be reusable across distinct instances."""

    def test_instances_with_equal_repr_do_not_share_cache(self):
        """Deterministic stand-in for an address collision.

        The old key was ``str(args)``, so it embedded the instance's memory
        address. CPython reuses addresses after a collection, meaning a new
        Transfer could inherit a discarded one's cached points and silently
        solve against the wrong mesh. Two instances that share a repr reproduce
        that collision without having to win a race with the allocator.
        """
        from transfer_blend_shape.utils import decorator

        class Thing(object):
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return "<thing>"

            @decorator.memoize
            def get(self):
                return self.value

        first, second = Thing("a"), Thing("b")
        self.assertEqual(repr(first), repr(second))
        self.assertEqual(first.get(), "a")
        self.assertEqual(second.get(), "b", "cache leaked between instances")

    def test_repeated_calls_are_cached(self):
        from transfer_blend_shape.utils import decorator

        calls = []

        class Thing(object):
            @decorator.memoize
            def get(self):
                calls.append(1)
                return len(calls)

        thing = Thing()
        self.assertEqual(thing.get(), 1)
        self.assertEqual(thing.get(), 1)
        self.assertEqual(len(calls), 1)

        thing.get.clear()
        self.assertEqual(thing.get(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2, argv=[sys.argv[0]])
