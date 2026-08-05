from maya import cmds
from PySide6 import QtWidgets, QtGui, QtCore

from transfer_blend_shape import transfer
from transfer_blend_shape.gui import dcc
from transfer_blend_shape.gui import icon
from transfer_blend_shape.gui import common
from transfer_blend_shape.gui import widgets
from transfer_blend_shape.utils import undo
from transfer_blend_shape.utils import naming


__all__ = [
    "TransferBlendShapeWidget",
    "show",
]
WINDOW_TITLE = "Transfer Blend Shape"
WINDOW_ICON = icon.get_icon_file_path("TBS_icon.png")


class TransferBlendShapeWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(TransferBlendShapeWidget, self).__init__(parent)

        # variables
        self._transfer = transfer.Transfer()
        scale_factor = self.logicalDpiX() / 96.0
        label_size = QtCore.QSize(85 * scale_factor, 18 * scale_factor)
        button_size = QtCore.QSize(120 * scale_factor, 18 * scale_factor)

        # set window
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QtGui.QIcon(WINDOW_ICON))
        self.resize(450 * scale_factor, 380 * scale_factor)

        # create layout
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # create source, target and virtual widgets
        source_text = QtWidgets.QLabel(self)
        source_text.setText("Source mesh:")
        source_text.setFixedSize(label_size)
        layout.addWidget(source_text, 0, 0)

        self.source = QtWidgets.QLineEdit(self)
        self.source.setReadOnly(True)
        layout.addWidget(self.source, 0, 1)

        source_button = QtWidgets.QPushButton(self)
        source_button.setText("Set source mesh")
        source_button.setFixedSize(button_size)
        source_button.released.connect(self.set_source_from_selection)
        layout.addWidget(source_button, 0, 2)

        target_text = QtWidgets.QLabel(self)
        target_text.setText("Target mesh:")
        target_text.setFixedSize(label_size)
        layout.addWidget(target_text, 1, 0)

        self.target = QtWidgets.QLineEdit(self)
        self.target.setReadOnly(True)
        layout.addWidget(self.target, 1, 1)

        target_button = QtWidgets.QPushButton(self)
        target_button.setText("Set target mesh")
        target_button.setFixedSize(button_size)
        target_button.released.connect(self.set_target_from_selection)
        layout.addWidget(target_button, 1, 2)

        virtual_text = QtWidgets.QLabel(self)
        virtual_text.setText("Virtual mesh:")
        virtual_text.setFixedSize(label_size)
        layout.addWidget(virtual_text, 2, 0)

        self.virtual = QtWidgets.QLineEdit(self)
        self.virtual.setReadOnly(True)
        self.virtual.setPlaceholderText("optional...")
        layout.addWidget(self.virtual, 2, 1)

        virtual_button = QtWidgets.QPushButton(self)
        virtual_button.setText("Set virtual mesh")
        virtual_button.setFixedSize(button_size)
        virtual_button.released.connect(self.set_virtual_from_selection)
        layout.addWidget(virtual_button, 2, 2)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 3, 0, 1, 3)

        # create threshold widgets
        threshold_text = QtWidgets.QLabel(self)
        threshold_text.setText("Threshold:")
        layout.addWidget(threshold_text, 4, 0)

        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(0.001)
        self.threshold.valueChanged.connect(self.set_threshold)
        self.threshold.setToolTip("The threshold determines the threshold where "
                                  "vertices are considered to be static.")
        layout.addWidget(self.threshold, 4, 1, 1, 2)

        # create iterations widgets
        iterations_text = QtWidgets.QLabel(self)
        iterations_text.setText("Iterations:")
        layout.addWidget(iterations_text, 5, 0)

        self.iterations = QtWidgets.QSpinBox(self)
        self.iterations.setValue(3)
        self.iterations.setMinimum(0)
        self.iterations.valueChanged.connect(self.set_iterations)
        self.iterations.setToolTip("The iterations determine the amount of smoothing "
                                   "operations applied to the deformed vertices.")
        layout.addWidget(self.iterations, 5, 1, 1, 2)

        # create colour set widgets
        colour_sets_text = QtWidgets.QLabel(self)
        colour_sets_text.setText("Colour sets:")
        layout.addWidget(colour_sets_text, 6, 0)

        self.create_colour_sets = QtWidgets.QCheckBox(self)
        self.create_colour_sets.stateChanged.connect(self.set_create_colour_sets)
        self.create_colour_sets.setToolTip("Colour sets will be created that will visualize "
                                           "the deformed vertices and the smoothing weights.")
        layout.addWidget(self.create_colour_sets, 6, 1, 1, 2)

        # create source offset widgets
        offset_text = QtWidgets.QLabel(self)
        offset_text.setText("Source offset:")
        layout.addWidget(offset_text, 7, 0)

        self.preserve_source_offset = QtWidgets.QCheckBox(self)
        self.preserve_source_offset.stateChanged.connect(self.set_preserve_source_offset)
        self.preserve_source_offset.setToolTip(
            "Following gives a shell the motion of the skin underneath it. Enable this "
            "to also carry over motion the source author added on top of that, for "
            "example a brow card deliberately pushed further than the skin.")
        layout.addWidget(self.preserve_source_offset, 7, 1, 1, 2)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 8, 0, 1, 3)

        # create detached shell widgets
        shell_text = QtWidgets.QLabel(self)
        shell_text.setText("Detached shells:")
        layout.addWidget(shell_text, 9, 0)

        self.shell_info = QtWidgets.QLabel(self)
        self.shell_info.setText("set a target mesh...")
        layout.addWidget(self.shell_info, 9, 1)

        shell_button = QtWidgets.QPushButton(self)
        shell_button.setText("Analyse shells")
        shell_button.setFixedSize(button_size)
        shell_button.released.connect(self.populate_shells)
        layout.addWidget(shell_button, 9, 2)

        self.shells = QtWidgets.QTreeWidget(self)
        self.shells.setHeaderLabels(["Shell", "Vertices", "Follow", ""])
        self.shells.setRootIsDecorated(False)
        self.shells.setAlternatingRowColors(True)
        self.shells.setMinimumHeight(120 * scale_factor)
        self.shells.setToolTip(
            "Rigid keeps a shell's shape exactly and only moves and rotates it with the "
            "skin, which is what eyebrow cards need. Smooth lets it deform with the skin "
            "so it stays attached to a lid that changes shape, which is what eyelash "
            "strips need.")
        layout.addWidget(self.shells, 10, 0, 1, 3)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 11, 0, 1, 3)

        # create transfer widgets
        self.transfer_selection = QtWidgets.QPushButton(self)
        self.transfer_selection.setText("Transfer selection")
        self.transfer_selection.released.connect(self.transfer_from_selection)
        layout.addWidget(self.transfer_selection, 12, 0, 1, 3)

        self.transfer_blend_shape = QtWidgets.QPushButton(self)
        self.transfer_blend_shape.setText("Transfer from blend shape")
        self.transfer_blend_shape.released.connect(self.transfer_from_blend_shape)
        layout.addWidget(self.transfer_blend_shape, 13, 0, 1, 3)

        self.reset()

    # ------------------------------------------------------------------------

    @property
    def transfer(self):
        """
        :return: Transfer object
        :rtype: transfer.Transfer
        """
        return self._transfer

    @common.display_error
    def set_source_from_selection(self):
        """
        :raise RuntimeError: When nothing is selected.
        """
        selection = cmds.ls(selection=True)
        if not selection:
            raise RuntimeError("Unable to set source mesh, nothing selected.")

        self.transfer.set_source_mesh(selection[0])
        self.source.setText(naming.get_name(selection[0]))
        self.reset()

    @common.display_error
    def set_target_from_selection(self):
        """
        :raise RuntimeError: When nothing is selected.
        """
        selection = cmds.ls(selection=True)
        if not selection:
            raise RuntimeError("Unable to set target mesh, nothing selected.")

        self.transfer.set_target_mesh(selection[0])
        self.target.setText(naming.get_name(selection[0]))
        self.reset()

    @common.display_error
    def set_virtual_from_selection(self):
        """
        """
        selection = cmds.ls(selection=True)
        virtual_mesh = selection[0] if selection else None
        virtual_mesh_name = naming.get_name(selection[0]) if selection else None
        self.transfer.set_virtual_mesh(virtual_mesh)
        self.virtual.setText(virtual_mesh_name)
        self.reset()

    def set_iterations(self, iterations):
        """
        :param int iterations:
        """
        self.transfer.set_iterations(iterations)

    def set_threshold(self, threshold):
        """
        :param float threshold:
        """
        self.transfer.set_threshold(threshold)

    def set_create_colour_sets(self, state):
        """
        :param int state:
        """
        self.transfer.set_create_colour_sets(bool(state))

    def set_preserve_source_offset(self, state):
        """
        :param int state:
        """
        self.transfer.set_preserve_source_offset(bool(state))

    # ------------------------------------------------------------------------

    @common.display_error
    def populate_shells(self):
        """
        List the target's detached shells and expose a follow mode for each. The
        largest shell is the skin and is solved normally, everything after it is
        followed. Selecting a row's vertices in the viewport is the quickest way
        to tell which index is the brows and which is the lashes.

        :raise RuntimeError: When no target mesh is set.
        """
        self.shells.clear()

        if not self.transfer.target_mesh:
            raise RuntimeError("Unable to analyse shells, no target mesh set.")

        description = self.transfer.get_shell_description()
        followers = [item for item in description if item["role"] == "follower"]
        self.shell_info.setText(
            "{} shell{}, {} followed".format(
                len(description), "" if len(description) == 1 else "s", len(followers)))

        for item in description:
            widget = QtWidgets.QTreeWidgetItem(self.shells)
            widget.setText(0, "{} ({})".format(item["index"], item["role"]))
            widget.setText(1, str(item["vertices"]))

            if item["role"] == "skin":
                widget.setText(2, "solved")
                continue

            combo = QtWidgets.QComboBox(self.shells)
            combo.addItems(["Rigid", "Smooth", "Half"])
            combo.setCurrentIndex(0 if item["stiffness"] >= 1.0
                                  else 1 if item["stiffness"] <= 0.0 else 2)
            combo.currentIndexChanged.connect(
                lambda index, shell=item["index"]: self.set_shell_stiffness(shell, index))
            self.shells.setItemWidget(widget, 2, combo)

            button = QtWidgets.QPushButton(self.shells)
            button.setText("Select")
            button.released.connect(
                lambda shell=item["index"]: self.select_shell(shell))
            self.shells.setItemWidget(widget, 3, button)

        for column in range(4):
            self.shells.resizeColumnToContents(column)

    def set_shell_stiffness(self, shell, index):
        """
        :param int shell: Shell index.
        :param int index: Combo box index, rigid/smooth/half.
        """
        self.transfer.set_shell_stiffness([1.0, 0.0, 0.5][index], index=shell)

    @common.display_error
    def select_shell(self, shell):
        """
        Select a shell's vertices so the artist can see which shell a row is.

        :param int shell: Shell index.
        """
        vertices = self.transfer.get_shells()[shell]
        cmds.select(["{}.vtx[{}]".format(self.transfer.target_mesh, index)
                     for index in vertices])

    # ------------------------------------------------------------------------

    @common.display_error
    def transfer_from_selection(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                for node in cmds.ls(selection=True):
                    self.transfer.execute_from_mesh(node)

    @common.display_error
    def transfer_from_blend_shape(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                self.transfer.execute_from_blend_shape()

    def reset(self):
        """
        Enable the conversion buttons depending on the valid state of the
        transfer object and if the source has a blend shape attached.
        """
        is_valid = self.transfer.is_valid()
        self.transfer_selection.setEnabled(is_valid)
        self.transfer_blend_shape.setEnabled(bool(is_valid and self.transfer.is_valid_with_blend_shape()))

        # the shell list belongs to whichever target was set, so drop it
        self.shells.clear()
        self.shell_info.setText("set a target mesh..." if not self.transfer.target_mesh
                                else "press analyse shells...")


def show():
    parent = dcc.get_main_window()
    widget = TransferBlendShapeWidget(parent)
    widget.show()
