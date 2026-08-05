import time
import numpy
import logging
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from maya import cmds
from maya.api import OpenMaya

from transfer_blend_shape.utils import api
from transfer_blend_shape.utils import colour
from transfer_blend_shape.utils import conversion
from transfer_blend_shape.utils import decorator
from transfer_blend_shape.utils import naming
from transfer_blend_shape.utils import shell
from transfer_blend_shape.utils.deform import blend_shape

log = logging.getLogger(__name__)

EPS = 1e-12


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh. The transfer can be aided by a virtual
    mesh that creates additional triangles.

    A mesh made of several detached shells, for example a face with separate
    eyebrow cards and eyelash strips, cannot be solved as one system. The
    deformation gradient operator is translation invariant, so it fixes vertex
    positions only up to one free translation per connected component, and that
    freedom is removed using the static zero-delta vertices. A shell in which
    every vertex moves therefore has no anchor, making its block of the normal
    matrix singular and losing its position completely.

    Rather than solve those shells, their motion is derived from the solved
    skin, see :mod:`transfer_blend_shape.utils.shell`. That also keeps
    sliver-heavy geometry such as eyelashes out of the per-triangle solve and
    out of the area based smoothing, both of which are ill conditioned for
    slivers.
    """

    def __init__(
            self,
            source_mesh=None,
            target_mesh=None,
            virtual_mesh=None,
            iterations=3,
            threshold=0.001,
            create_colour_sets=False,
            shell_stiffness=1.0,
            shell_neighbours=8,
            preserve_source_offset=True,
            solve_detached_shells=False
    ):
        self._source_mesh = None
        self._target_mesh = None
        self._virtual_mesh = None
        self._threshold = 0.001
        self._iterations = 3
        self._create_colour_sets = False
        self._shell_stiffness = 1.0
        self._shell_stiffness_overrides = {}
        self._shell_neighbours = 8
        self._preserve_source_offset = True
        self._solve_detached_shells = False

        self.set_source_mesh(source_mesh)
        self.set_virtual_mesh(virtual_mesh)
        self.set_target_mesh(target_mesh)
        self.set_iterations(iterations)
        self.set_threshold(threshold)
        self.set_create_colour_sets(create_colour_sets)
        self.set_shell_stiffness(shell_stiffness)
        self.set_shell_neighbours(shell_neighbours)
        self.set_preserve_source_offset(preserve_source_offset)
        self.set_solve_detached_shells(solve_detached_shells)

    # ------------------------------------------------------------------------

    @property
    def source_mesh(self):
        """
        :return: Source mesh
        :rtype: str
        """
        return self._source_mesh

    @decorator.memoize
    def get_source_points(self):
        """
        :return: Source points
        :rtype: numpy.Array
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query points.")

        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)
        return numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]

    @decorator.memoize
    def get_source_triangles(self):
        """
        :return: Source triangles
        :rtype: list[int]
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query triangle indices.")

        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)
        _, triangles = mesh_fn.getTriangles()
        return list(triangles)

    @decorator.memoize
    def get_source_area(self):
        """
        :return: Source triangle area
        :rtype: numpy.Array
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query area.")

        source_points = self.get_source_points()
        return self.calculate_area(source_points)

    def set_source_mesh(self, source_mesh):
        """
        :param str source_mesh:
        """
        self._source_mesh = source_mesh
        self.get_source_points.clear()
        self.get_source_triangles.clear()
        self.get_source_area.clear()
        self.get_virtual_triangles.clear()
        self.get_source_shell_binding.clear()

    # ------------------------------------------------------------------------

    @property
    def target_mesh(self):
        """
        :return: Target mesh
        :rtype: str
        """
        return self._target_mesh

    @decorator.memoize
    def get_target_points(self):
        """
        :return: Target points
        :rtype: numpy.Array
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query points.")

        mesh_fn = api.conversion.get_mesh_fn(self.target_mesh)
        return numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]

    @decorator.memoize
    def get_target_connectivity(self):
        """
        :return: Target connectivity
        :rtype: list[list[int]]
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query connectivity.")

        connectivity = []
        mesh_dag = api.conversion.get_dag(self.target_mesh)
        mesh_iter = OpenMaya.MItMeshVertex(mesh_dag)

        while not mesh_iter.isDone():
            indices = list(mesh_iter.getConnectedVertices())
            connectivity.append(indices)
            mesh_iter.next()

        return connectivity

    @decorator.memoize
    def get_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query matrix.")

        return self.calculate_target_matrix()

    def set_target_mesh(self, target_mesh):
        """
        :param str target_mesh:
        """
        self._target_mesh = target_mesh
        self.get_target_points.clear()
        self.get_target_connectivity.clear()
        self.get_target_matrix.clear()
        self.get_shells.clear()
        self.get_target_shell_binding.clear()
        self.get_source_shell_binding.clear()
        self._shell_stiffness_overrides = {}

    # ------------------------------------------------------------------------

    @property
    def virtual_mesh(self):
        """
        :return: Virtual
        :rtype: str
        """
        return self._virtual_mesh

    @decorator.memoize
    def get_virtual_triangles(self, threshold=0.001):
        """
        :param float threshold:
        :return: Virtual triangles
        :rtype: list[int]
        :raise RuntimeError: When minimum length surpasses threshold.
        """
        if self.virtual_mesh is None:
            return []

        idx = {}
        mesh_fn = api.conversion.get_mesh_fn(self.virtual_mesh)

        source_points = self.get_source_points()
        virtual_points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        _, virtual_triangles = mesh_fn.getTriangles()

        for i, point in enumerate(virtual_points):
            lengths = scipy.linalg.norm(source_points - point, axis=1)
            index = lengths.argmin()

            if lengths[index] > threshold:
                raise RuntimeError("Unable to map vertex {} if the virtual mesh "
                                   "to the source mesh.".format(index))

            idx[i] = index

        return [idx[vertex] for vertex in virtual_triangles]

    def set_virtual_mesh(self, virtual_mesh):
        """
        :param str virtual_mesh:
        """
        self._virtual_mesh = virtual_mesh
        self.get_target_matrix.clear()
        self.get_virtual_triangles.clear()

    # ------------------------------------------------------------------------

    @property
    def iterations(self):
        """
        :return: Iterations
        :rtype: int
        """
        return self._iterations

    def set_iterations(self, iterations):
        """
        :param int iterations:
        :raise TypeError: When iterations is not a int.
        :raise ValueError: When iterations is lower than 0.
        """
        if not isinstance(iterations, int):
            raise TypeError("Unable to set iterations, should be of type int.")
        elif iterations < 0:
            raise ValueError("Num iterations are not allowed to be lower than 0.")

        self._iterations = iterations

    @property
    def threshold(self):
        """
        :return: Threshold
        :rtype: float
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        :param float threshold:
        :raise TypeError: When threshold is not a float or int.
        :raise ValueError: When threshold is lower or equal to 0.
        """
        if not isinstance(threshold, (float, int)):
            raise TypeError("Unable to set threshold, should be of type int/float.")
        elif threshold <= 0.0:
            raise ValueError("Threshold is not allowed to be 0.0 or lower.")

        self._threshold = threshold

    @property
    def create_colour_sets(self):
        """
        :return: Create colour sets state
        :rtype: bool
        """
        return self._create_colour_sets

    def set_create_colour_sets(self, state):
        """
        :param bool state:
        """
        if not isinstance(state, bool):
            raise TypeError("Unable to set colour set creation state, should be of type bool.")

        self._create_colour_sets = state

    # ------------------------------------------------------------------------

    @property
    def shell_stiffness(self):
        """
        :return: Default follower shell stiffness
        :rtype: float
        """
        return self._shell_stiffness

    def set_shell_stiffness(self, stiffness, index=None):
        """
        Control how a detached shell follows the solved skin. A stiffness of 1
        moves the shell with a single rigid transform, preserving its shape
        exactly while its position and orientation track the skin, which is what
        eyebrow cards need. A stiffness of 0 lets the shell deform with the skin
        via a smooth blended field, which is what eyelash strips need so they
        stay attached along a lid that changes shape.

        :param float stiffness: Value between 0 and 1.
        :param int/None index: Shell index to apply to, all shells when None.
        :raise TypeError: When stiffness is not a float or int.
        :raise ValueError: When stiffness is outside of the 0 to 1 range.
        """
        if not isinstance(stiffness, (float, int)) or isinstance(stiffness, bool):
            raise TypeError("Unable to set shell stiffness, should be of type int/float.")
        elif not 0.0 <= stiffness <= 1.0:
            raise ValueError("Shell stiffness has to be between 0.0 and 1.0.")

        if index is None:
            self._shell_stiffness = float(stiffness)
            self._shell_stiffness_overrides = {}
        else:
            self._shell_stiffness_overrides[int(index)] = float(stiffness)

    def get_shell_stiffness(self, index):
        """
        :param int index: Shell index.
        :return: Stiffness for the provided shell.
        :rtype: float
        """
        return self._shell_stiffness_overrides.get(int(index), self._shell_stiffness)

    @property
    def shell_neighbours(self):
        """
        :return: Number of skin vertices blended per follower vertex
        :rtype: int
        """
        return self._shell_neighbours

    def set_shell_neighbours(self, num):
        """
        :param int num:
        :raise TypeError: When num is not an int.
        :raise ValueError: When num is lower than 1.
        """
        if not isinstance(num, int) or isinstance(num, bool):
            raise TypeError("Unable to set shell neighbours, should be of type int.")
        elif num < 1:
            raise ValueError("Shell neighbours has to be 1 or higher.")

        self._shell_neighbours = num
        self.get_target_shell_binding.clear()
        self.get_source_shell_binding.clear()

    @property
    def preserve_source_offset(self):
        """
        :return: Preserve source offset state
        :rtype: bool
        """
        return self._preserve_source_offset

    def set_preserve_source_offset(self, state):
        """
        Transfer a shell's motion *relative* to the skin rather than simply
        replaying the skin's motion, which is on by default.

        Following alone gives a shell exactly the motion implied by the skin
        underneath it, so anything the author did on top of that is dropped. Two
        cases make that visible. A shape that moves only the eyebrow cards and
        leaves the skin alone has nothing to follow and transfers as a no-op.
        A shape that moves the jaw while deliberately holding the brows still
        drags the brows along with the cheek.

        Measuring the leftover rigid motion on the source and transplanting it
        onto the target covers both, and reduces to plain following whenever the
        author did move the shell with the skin.

        :param bool state:
        :raise TypeError: When state is not a bool.
        """
        if not isinstance(state, bool):
            raise TypeError("Unable to set preserve source offset state, should be of type bool.")

        self._preserve_source_offset = state

    @property
    def solve_detached_shells(self):
        """
        :return: Solve detached shells state
        :rtype: bool
        """
        return self._solve_detached_shells

    def set_solve_detached_shells(self, state):
        """
        Escape hatch that restores the original behaviour of pushing every shell
        through the linear solve. Only useful for comparison, an unanchored
        shell will lose its position.

        :param bool state:
        :raise TypeError: When state is not a bool.
        """
        if not isinstance(state, bool):
            raise TypeError("Unable to set solve detached shells state, should be of type bool.")

        self._solve_detached_shells = state

    # ------------------------------------------------------------------------

    @decorator.memoize
    def get_shells(self):
        """
        Connected components of the target mesh, ordered by descending vertex
        count. Index 0 is the skin, everything after it is a detached follower
        shell such as an eyebrow card or eyelash strip.

        :return: Per-shell vertex indices
        :rtype: list[numpy.Array]
        :raise RuntimeError: When target is not defined.
        """
        labels, order = shell.connected_components(self.get_target_connectivity())
        shells = [numpy.nonzero(labels == label)[0] for label in order]

        if len(shells) > 1:
            log.info("Target '%s' has %d shells, the largest (%d vertices) is treated as "
                     "the skin and the other %d are followed.",
                     self.target_mesh, len(shells), len(shells[0]), len(shells) - 1)

        return shells

    def get_skin_vertices(self):
        """
        :return: Vertex indices of the largest shell
        :rtype: numpy.Array
        """
        return self.get_shells()[0]

    def get_follower_vertices(self):
        """
        :return: Vertex indices of every shell but the largest
        :rtype: numpy.Array
        """
        shells = self.get_shells()
        if len(shells) == 1:
            return numpy.array([], dtype=numpy.int64)

        return numpy.sort(numpy.concatenate(shells[1:]))

    def get_follower_shell_labels(self):
        """
        :return: Per-follower-vertex shell index, matching the vertex order of
            :meth:`get_follower_vertices`.
        :rtype: numpy.Array
        """
        followers = self.get_follower_vertices()
        labels = numpy.zeros(len(followers), dtype=numpy.int64)
        for index, vertices in enumerate(self.get_shells()):
            if not index:
                continue

            labels[numpy.isin(followers, vertices)] = index

        return labels

    def get_shell_description(self):
        """
        Human readable summary of the detected shells, used for logging and to
        populate the interface so an artist can tell which index is which.

        :return: Per-shell index, vertex count, size and stiffness
        :rtype: list[dict]
        """
        target_points = self.get_target_points()

        description = []
        for index, vertices in enumerate(self.get_shells()):
            points = target_points[vertices]
            size = points.max(axis=0) - points.min(axis=0)
            description.append({
                "index": index,
                "vertices": len(vertices),
                "centre": points.mean(axis=0),
                "size": float(numpy.linalg.norm(size)),
                "stiffness": None if not index else self.get_shell_stiffness(index),
                "role": "skin" if not index else "follower",
            })

        return description

    def get_shell_connectivity(self, vertices):
        """
        Remap the target connectivity onto a shell's local vertex order. Shells
        are disjoint by definition, so every neighbour is inside the shell.

        :param numpy.Array vertices: Shell vertex indices.
        :return: Local per-vertex connectivity
        :rtype: list[list[int]]
        """
        connectivity = self.get_target_connectivity()
        remap = numpy.full(len(connectivity), -1, dtype=numpy.int64)
        remap[vertices] = numpy.arange(len(vertices))
        return [[int(remap[other]) for other in connectivity[index]] for index in vertices]

    @decorator.memoize
    def get_target_shell_binding(self):
        """
        :return: Follower binding built on the target rest pose
        :rtype: dict/None
        """
        followers = self.get_follower_vertices()
        if not len(followers):
            return None

        skin = self.get_skin_vertices()
        target_points = self.get_target_points()
        return shell.build_binding(
            target_points[followers],
            target_points[skin],
            self.get_shell_connectivity(skin),
            self.shell_neighbours,
        )

    @decorator.memoize
    def get_source_shell_binding(self):
        """
        Equivalent binding on the source rest pose, only needed to measure the
        authored offset when :attr:`preserve_source_offset` is enabled.

        :return: Follower binding built on the source rest pose
        :rtype: dict/None
        """
        followers = self.get_follower_vertices()
        if not len(followers):
            return None

        skin = self.get_skin_vertices()
        source_points = self.get_source_points()
        return shell.build_binding(
            source_points[followers],
            source_points[skin],
            self.get_shell_connectivity(skin),
            self.shell_neighbours,
        )

    # ------------------------------------------------------------------------

    def is_valid(self):
        """
        :return: Valid state
        :rtype: bool
        """
        is_source_valid = self.source_mesh and cmds.objExists(self.source_mesh)
        is_target_valid = self.target_mesh and cmds.objExists(self.target_mesh)
        is_virtual_valid = not self.virtual_mesh or cmds.objExists(self.virtual_mesh)
        return bool(is_source_valid and is_target_valid and is_virtual_valid)

    def is_valid_with_blend_shape(self):
        """
        :return: Valid state + blend shape
        :rtype: bool
        """
        if not self.is_valid():
            return False

        return bool(blend_shape.get_blend_shape(self.source_mesh))

    # ------------------------------------------------------------------------

    def filter_vertices(self, points):
        """
        :param numpy.Array points:
        :return: Static/Dynamic vertices
        :rtype: numpy.Array, numpy.Array
        """
        source_points = self.get_source_points()
        lengths = scipy.linalg.norm(source_points - points, axis=1)
        return numpy.nonzero(lengths <= self.threshold)[0], numpy.nonzero(lengths > self.threshold)[0]

    def calculate_area(self, points):
        """
        :param numpy.Array points:
        :return: Triangle areas
        :rtype: numpy.Array
        """
        vertex_area = numpy.zeros(shape=(len(points),))
        source_triangles = self.get_source_triangles()
        triangle_points = numpy.take(points, source_triangles, axis=0)
        triangle_points = triangle_points.reshape((len(triangle_points) // 3, 3, 3))

        length = triangle_points - triangle_points[:, [1, 2, 0], :]
        length = scipy.linalg.norm(length, axis=2)

        s = numpy.sum(length, axis=1) / 2.0
        areas = numpy.sqrt(s * (s - length[:, 0]) * (s - length[:, 1]) * (s - length[:, 2]))

        for indices, area in zip(conversion.as_chunks(source_triangles, 3), areas):
            for index in indices:
                vertex_area[index] += area

        return vertex_area

    @staticmethod
    def calculate_edge_matrix(point1, point2, point3):
        """
        :param numpy.Array point1:
        :param numpy.Array point2:
        :param numpy.Array point3:
        :return: Edge matrix
        :rtype: numpy.Array
        """
        e0 = point2 - point1
        e1 = point3 - point1
        e2 = numpy.cross(e0, e1)
        return numpy.array([e0, e1, e2]).transpose()

    def calculate_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        """
        triangles = self.get_source_triangles() + self.get_virtual_triangles()
        target_points = self.get_target_points()

        matrix = numpy.zeros((len(triangles), target_points.shape[0]))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(triangles, 3)):
            e0 = target_points[i1] - target_points[i0]
            e1 = target_points[i2] - target_points[i0]
            va = numpy.array([e0, e1]).transpose()

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            for j in range(3):
                matrix[i * 3 + j][i0] = - inv_rqt[0][j] - inv_rqt[1][j]
                matrix[i * 3 + j][i1] = inv_rqt[0][j]
                matrix[i * 3 + j][i2] = inv_rqt[1][j]

        return matrix

    def calculate_deformation_gradient(self, points):
        """
        :param numpy.Array points:
        :return: Deformation gradient
        :rtype: numpy.Array
        """
        triangles = self.get_source_triangles() + self.get_virtual_triangles()
        source_points = self.get_source_points()

        matrix = numpy.zeros((len(triangles), 3))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(triangles, 3)):
            va = self.calculate_edge_matrix(source_points[i0], source_points[i1], source_points[i2])
            vb = self.calculate_edge_matrix(points[i0], points[i1], points[i2])

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            sa = numpy.dot(vb, inv_rqt)
            sat = sa.transpose()
            matrix[i * 3: i * 3 + 3] = sat

        return matrix

    def calculate_laplacian_weights(self, points, ignore):
        """
        Calculate the laplacian weights depending on the change in per vertex
        area between the source and target points. The calculated weights are
        smoothed a number of times defined by the iterations, this will even
        out the smooth.

        :param numpy.Array points:
        :param numpy.Array ignore:
        :return: Laplacian weights
        :rtype: numpy.Array
        """
        source_area = self.get_source_area()
        target_area = self.calculate_area(points)

        # a fully degenerate vertex, every surrounding triangle a sliver, has an
        # area of zero. Dividing by it yields inf or nan, which the smoothing
        # loop below then spreads across the shell and which finally lands in
        # the vertex positions. Sliver triangles are exactly what eyelash
        # geometry is made of, so guard the ratio rather than trust the input.
        numerator = numpy.maximum(source_area, target_area)
        denominator = numpy.minimum(source_area, target_area)
        valid = denominator > EPS
        weights = numpy.zeros(len(points), dtype=float)
        weights[valid] = numerator[valid] / denominator[valid] - 1.0

        degenerate = int(numpy.count_nonzero(~valid))
        if degenerate:
            log.debug("Skipping smoothing weights for %d degenerate vertices.", degenerate)

        smoothing_matrix = self.calculate_laplacian_matrix(numpy.ones(len(points)), ignore)

        for _ in range(self.iterations):
            diff = numpy.array(smoothing_matrix.dot(weights))
            weights = weights - diff

        return weights.reshape(len(points))

    def calculate_laplacian_matrix(self, weights, ignore):
        """
        Create a laplacian smoothing matrix based on the weights, for the
        smoothing the number of vertices and vertex connectivity is used
        together with the provided weights, the weights are clamped to a
        maximum of 1. Any ignore indices will have their weights set to 0.

        :param numpy.Array weights:
        :param numpy.Array ignore:
        :return: Laplacian smoothing matrix
        :rtype: scipy.sparse.csr.csr_matrix
        """
        num = self.get_target_points().shape[0]
        connectivity = self.get_target_connectivity()

        weights[ignore] = 0
        data, rows, columns = [], [], []

        for i, weight in enumerate(weights):
            weight = min([weights[i], 1])
            indices = connectivity[i]
            z = len(indices)
            data += ([i] * (z + 1))
            rows += indices + [i]
            columns += ([-weight / float(z)] * z) + [weight]

        return scipy.sparse.coo_matrix((columns, (data, rows)), shape=(num, num)).tocsr()

    def calculate_follower_points(self, binding, skin_points, points, name=None):
        """
        Derive the positions of every detached shell from the solved skin.

        :param dict binding: Target rest pose binding.
        :param numpy.Array skin_points: Solved skin positions.
        :param numpy.Array points: Source deformed points, full mesh order.
        :param str/None name: Target name, used for error reporting.
        :return: Follower positions, matching :meth:`get_follower_vertices`.
        :rtype: numpy.Array
        :raise RuntimeError: When the derived positions are not finite.
        """
        labels = self.get_follower_shell_labels()
        stiffness = {int(label): self.get_shell_stiffness(int(label))
                     for label in numpy.unique(labels)}

        followers = self.get_follower_vertices()
        source_points = self.get_source_points()
        followed = shell.evaluate_binding(binding, skin_points, stiffness, labels)

        if self.preserve_source_offset:
            predicted = shell.evaluate_binding(
                self.get_source_shell_binding(),
                points[self.get_skin_vertices()],
                stiffness,
                labels,
            )
            followed = shell.transplant_source_offset(
                followed, predicted, points[followers], source_points[followers], labels)

        # A shell the author left untouched must stay untouched. Following would
        # otherwise drag a static eyebrow along with a moving jaw, where the
        # original tool left it alone. Deltas are measured on the source with the
        # same threshold used to pick the static vertices.
        deltas = scipy.linalg.norm(points[followers] - source_points[followers], axis=1)
        rest = binding["follower_rest"]
        for label in numpy.unique(labels):
            indices = numpy.nonzero(labels == label)[0]
            if not numpy.any(deltas[indices] > self.threshold):
                followed[indices] = rest[indices]

        if not numpy.all(numpy.isfinite(followed)):
            raise RuntimeError("Derived non-finite positions for the detached shells of "
                               "target '{}'.".format(name))

        return followed

    # ------------------------------------------------------------------------

    def calculate_points(self, points, name=None):
        """
        Solve the target positions for a single source shape. This is the entire
        numerical pipeline with no scene interaction, which keeps it testable
        without a Maya session.

        :param numpy.Array points: Source deformed points.
        :param str/None name: Target name, used for error reporting.
        :return: Solved target points, smoothing weights and deformed vertices.
            The points are None when the shape holds no deltas at all.
        :rtype: tuple[numpy.Array/None, numpy.Array/None, numpy.Array]
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When no static vertices are found.
        :raise RuntimeError: When the solve produces non-finite positions.
        """
        source_points = self.get_source_points()
        target_points = self.get_target_points()
        if source_points.shape[0] != target_points.shape[0]:
            raise RuntimeError("Vertex count between source mesh '{}' and target mesh '{}' "
                               "do not match.".format(self.source_mesh, self.target_mesh))

        static_vertices, deformed_vertices = self.filter_vertices(points)
        if not len(static_vertices):
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))
        elif not len(deformed_vertices):
            return None, None, deformed_vertices

        # detached shells are removed from the solve. The gradient operator is
        # translation invariant, so a shell containing no static vertex has an
        # undetermined translation and a singular normal matrix. Left in, its
        # zero pivot produces nan that back substitution then spreads through
        # the whole coupled system.
        follower_vertices = self.get_follower_vertices()
        binding = None if self.solve_detached_shells else self.get_target_shell_binding()
        if binding is None:
            follower_vertices = numpy.array([], dtype=numpy.int64)
            solve_vertices = deformed_vertices
        else:
            solve_vertices = numpy.setdiff1d(deformed_vertices, follower_vertices)

        skin_vertices = self.get_skin_vertices()
        if len(follower_vertices) and not len(numpy.intersect1d(static_vertices, skin_vertices)):
            raise RuntimeError("No static vertices found on the skin shell for target '{}', "
                               "the solve has nothing to anchor to. Try increasing the "
                               "threshold.".format(name))

        target_points = target_points.copy()
        if len(solve_vertices):
            target_matrix = self.get_target_matrix()

            # calculate deformation gradient, the static vertices are used to
            # anchor the static vertices in place.
            static_matrix = target_matrix[:, static_vertices]
            static_points = target_points[static_vertices, :]
            static_gradient = numpy.dot(static_matrix, static_points)
            deformation_gradient = self.calculate_deformation_gradient(points) - static_gradient

            # isolate dynamic vertices and solve their position. As it is quicker
            # to set all points rather than individual ones the entire target
            # point list is constructed.
            deformed_matrix = target_matrix[:, solve_vertices]
            deformed_matrix_transpose = deformed_matrix.transpose()
            normal_matrix = numpy.dot(deformed_matrix_transpose, deformed_matrix)
            lu, piv = scipy.linalg.lu_factor(normal_matrix)
            uts = numpy.dot(deformed_matrix_transpose, deformation_gradient)
            deformed_points = scipy.linalg.lu_solve((lu, piv), uts)

            if not numpy.all(numpy.isfinite(deformed_points)):
                raise RuntimeError(
                    "Solve for target '{}' produced non-finite positions, so the system is "
                    "singular. Either a detached shell has no static vertex to anchor it, "
                    "or the mesh has vertices that belong to no face, for example a stray "
                    "wire edge, whose position nothing constrains.".format(name))

            target_points[solve_vertices, :] = deformed_points

        # calculate the laplacian smoothing weights/matrix using the
        # per-vertex area difference, this will ensure area's with most
        # highest difference receive the most smoothing, these are applied
        # to the calculated points. Follower vertices are held out, their
        # positions are derived rather than solved, and sliver triangles make
        # the area ratio meaningless anyway.
        ignore = (static_vertices if not len(follower_vertices)
                  else numpy.union1d(static_vertices, follower_vertices))
        smoothing_weights = self.calculate_laplacian_weights(points, ignore)
        smoothing_matrix = self.calculate_laplacian_matrix(smoothing_weights, ignore)
        for _ in range(self.iterations):
            diff = numpy.array(smoothing_matrix.dot(target_points))
            target_points = target_points - diff

        # derive the detached shells from the solved skin
        if binding is not None:
            target_points[follower_vertices, :] = self.calculate_follower_points(
                binding, target_points[skin_vertices], points, name)

        return target_points, smoothing_weights, deformed_vertices

    # ------------------------------------------------------------------------

    def execute(self, points, name):
        """
        :param numpy.Array points:
        :param str name:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When no static vertices are found.
        """
        t = time.time()

        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set at least source and target.")

        target_points, smoothing_weights, deformed_vertices = self.calculate_points(points, name)
        if target_points is None:
            target = cmds.duplicate(self.target_mesh, name=name)[0]
            log.info("Transferred '{}' as a static mesh.".format(name))
            return target

        # duplicate the original target and update its points
        target = cmds.duplicate(self.target_mesh, name=name)[0]
        target_dag = api.conversion.get_dag(target)
        target_dag.extendToShape()
        target_fn = OpenMaya.MFnMesh(target_dag)
        target_fn.setPoints([OpenMaya.MPoint(point) for point in target_points], OpenMaya.MSpace.kObject)

        # create an deformed vertices and weight map colour set on the target
        # that can be used for debugging reasons.
        if self.create_colour_sets:
            vertices = set(deformed_vertices)
            vertices_colour = [[int(index in vertices)] * 3 for index in range(target_fn.numVertices)]
            weights_colour = [[weight] * 3 for weight in smoothing_weights]
            colour.create_colour_set(target, "deformed", vertices_colour)
            colour.create_colour_set(target, "weights", weights_colour)

        log.info("Transferred '{}' in {:.3f} seconds.".format(name, time.time() - t))

        return target

    def execute_from_mesh(self, mesh, name=None):
        """
        :param str mesh:
        :param str/None name:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When provided mesh is not a mesh.
        :raise RuntimeError: When mesh vertex count doesn't match source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        mesh_name = naming.get_leaf_name(mesh)
        mesh_fn = api.conversion.get_mesh_fn(mesh)
        name = name if name is not None else "{}_TGT".format(mesh_name)
        points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        return self.execute(points, name)

    def execute_from_blend_shape(self):
        """
        :return: Targets
        :rtype: list[str]
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When no blend shape is connected to the source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid_with_blend_shape():
            raise RuntimeError("Invalid transfer, set at least source with blend shape and target.")

        bs = blend_shape.get_blend_shape(self.source_mesh)
        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)

        cmds.setAttr("{}.envelope".format(bs), 1)
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 0)

        targets = []
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 1)
            points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
            cmds.setAttr("{}.{}".format(bs, name), 0)

            target = self.execute(points, name)
            targets.append(target)

        return targets
