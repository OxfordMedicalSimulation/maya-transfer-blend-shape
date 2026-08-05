# maya-transfer-blend-shape
Retarget your blend shapes between meshes with the same topology.

<p align="center"><img src="docs/_images/transfer-blend-shape-workflow.png?raw=true"></p>

## Installation
* Extract the content of the .rar file anywhere on disk.
* Drag the transfer-blend-shape.mel file in Maya to permanently install the script.

## Usage
A button on the MiscTools shelf will be created that will allow easy access to 
the ui, this way the user doesn't need to worry about any of the code. If user 
wishes to not use the shelf button the following commands can be used. The 
transfer will only work if at least one vertex has no delta, these fixed 
vertices are used to transfer the solution to the correct position in object 
space, the threshold can be increased to make sure vertices are linked.

<p align="center"><img src="docs/_images/transfer-blend-shape-ui.png?raw=true"></p>

A virtual mesh can be used to add additional triangles to the solve, it is 
important that all vertices of the virtual mesh can be mapped to vertices on
the source mesh, this can be done by snapping them.

<p align="center"><img src="docs/_images/transfer-blend-shape-debug.png?raw=true"></p>

The number of iterations determine the amount of times the laplacian smoothing
matrix is applied to the deformed vertices. This smoothing matrix is
calculated using weights determined by area difference on a per-vertex basis.

Colour sets can be created to visualize the deformed vertices and the 
laplacian smoothing weights.

Command line:
```python
import transfer_blend_shape
transfer = transfer_blend_shape.Transfer(source_mesh, target_mesh, virtual_mesh=None, iterations=3, threshold=0.001)
transfer.execute_from_mesh(mesh, name)
transfer.execute_from_blend_shape()
```

## Detached shells
A mesh can be made up of several detached shells, for example a face with 
separate eyebrow cards and eyelash strips. Those shells cannot be part of the 
solve. The deformation gradient operator is translation invariant, so it fixes 
positions only up to one free translation per connected component, and that 
freedom is removed using the static zero-delta vertices. A shell in which every 
vertex moves has no anchor, its block of the normal matrix is singular, and the 
shell loses its position completely while keeping its shape, which is why it 
used to end up at the object origin.

The largest shell is treated as the skin and solved as before. Every other shell 
is taken out of the solve and its motion is derived from the solved skin 
instead, which also keeps sliver-heavy geometry such as eyelashes away from the 
per-triangle QR inverse and the area based smoothing weights, both of which are 
ill conditioned for slivers.

How a shell follows is controlled per shell by its stiffness. A stiffness of 1 
moves it with a single rigid transform, preserving its shape exactly while its 
position and orientation track the skin, which is what eyebrow cards need. A 
stiffness of 0 lets it deform with the skin through a smooth blended field, 
which is what eyelash strips need so they stay attached to a lid that changes 
shape. Press *Analyse shells* in the ui to list the shells, use *Select* to see 
which is which in the viewport, and set the follow mode per row.

```python
transfer = transfer_blend_shape.Transfer(source_mesh, target_mesh)
for shell in transfer.get_shell_description():
    print(shell["index"], shell["role"], shell["vertices"])

transfer.set_shell_stiffness(1.0)             # rigid, the default for all shells
transfer.set_shell_stiffness(0.0, index=1)    # let shell 1 deform, eg the lashes
transfer.execute_from_blend_shape()
```

Following gives a shell the motion implied by the skin underneath it. When a 
modeller has deliberately pushed a card further than the skin, that intent is 
not part of the skin and is lost. Enabling `preserve_source_offset` measures 
the leftover rigid motion on the source and transplants it onto the target.

The transfer still creates one mesh per shape, exactly as before. The maths runs 
without a Maya session, so it is covered by tests that can be run directly:

```
python tests/offline/test_shells.py        # regression suite
python tests/offline/diagnose_world_zero.py  # root cause demonstration
```

Display UI:
```python
import transfer_blend_shape.gui
transfer_blend_shape.gui.show()
```

## Note
This tool requires *numpy* and *scipy* to be installed to your environment. 
Using linux or Maya 2022+ on windows this can be done via a simple pip 
install. For older windows versions a custom version will have to be compiled 
against the correct VS version. 

Example images are generated using the [MetaHuman](https://www.unrealengine.com/en-US/digital-humans) 
exports for the source/target base and source jaw open shape. Target jaw open 
is generated using the tool.