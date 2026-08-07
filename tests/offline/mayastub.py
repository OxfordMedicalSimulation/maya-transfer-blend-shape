"""Minimal Maya stubs so the real transfer code can be exercised offline.

``transfer_blend_shape.transfer`` imports ``maya.cmds`` and ``maya.api.OpenMaya``
at module scope. Registering these stubs in ``sys.modules`` before the import
lets the actual shipped solver run under plain CPython, so the tests cover the
production code path rather than a reimplementation of it.

Only the surface the numerical pipeline touches is stubbed. Anything that would
really talk to a scene raises, so a test can never silently pass by accident.
"""
import os
import sys
import types


def _unsupported(name):
    """Callable that raises when used, so real scene calls fail loudly."""

    def raise_unsupported(*args, **kwargs):
        raise RuntimeError(
            "'{}' is not available offline, the test should only exercise the "
            "numerical pipeline.".format(name))

    return raise_unsupported


def install():
    """Register the stub modules and put the package on ``sys.path``."""
    if "maya" not in sys.modules:
        maya = types.ModuleType("maya")
        cmds = types.ModuleType("maya.cmds")

        # only the handful of query commands the maths path can reach
        cmds.objExists = lambda *args, **kwargs: True
        cmds.ls = lambda *args, **kwargs: []
        cmds.listRelatives = lambda *args, **kwargs: []
        cmds.listHistory = lambda *args, **kwargs: []
        cmds.nodeType = lambda *args, **kwargs: ""
        cmds.duplicate = _unsupported("cmds.duplicate")

        api = types.ModuleType("maya.api")
        openmaya = types.ModuleType("maya.api.OpenMaya")

        class MSpace(object):
            kObject = 0
            kWorld = 1

        openmaya.MSpace = MSpace
        openmaya.MPoint = tuple
        openmaya.MFnMesh = _unsupported("OpenMaya.MFnMesh")
        openmaya.MItMeshVertex = _unsupported("OpenMaya.MItMeshVertex")

        maya.cmds = cmds
        maya.api = api
        api.OpenMaya = openmaya

        sys.modules["maya"] = maya
        sys.modules["maya.cmds"] = cmds
        sys.modules["maya.api"] = api
        sys.modules["maya.api.OpenMaya"] = openmaya

    scripts = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


class OfflineTransfer(object):
    """Build a real ``Transfer`` fed from numpy arrays instead of a scene.

    The memoized accessors are replaced with the supplied data, which is exactly
    what they would return from a mesh. Everything downstream is untouched
    production code.
    """

    def __new__(cls, source_points, target_points, connectivity, triangles, **kwargs):
        install()
        from transfer_blend_shape import transfer as transfer_module

        instance = transfer_module.Transfer(**kwargs)
        instance._source_mesh = "source_MESH"
        instance._target_mesh = "target_MESH"

        instance.get_source_points = lambda: source_points
        instance.get_source_points.clear = lambda: None
        instance.get_target_points = lambda: target_points
        instance.get_target_points.clear = lambda: None
        instance.get_target_connectivity = lambda: connectivity
        instance.get_target_connectivity.clear = lambda: None
        instance.get_source_triangles = lambda: triangles
        instance.get_source_triangles.clear = lambda: None
        instance.get_virtual_triangles = lambda *a, **k: []
        instance.get_virtual_triangles.clear = lambda: None

        # get_source_area and get_target_matrix are memoized on top of the
        # accessors above, so they still run the real implementation
        return instance
