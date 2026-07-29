from __future__ import absolute_import

import shiboken6
from maya import OpenMayaUI
from PySide6 import QtWidgets, QtCore

__all__ = [
    "get_main_window",
    "maya_to_qt",
    "qt_to_maya",
]


def get_main_window():
    """
    Returns the Maya main window.

    Returns:
        QtWidgets.QMainWindow
    Raises:
        RuntimeError: If the main window cannot be obtained.
    """
    ptr = OpenMayaUI.MQtUtil.mainWindow()

    if ptr is None:
        raise RuntimeError("Failed to obtain a handle on the Maya main window.")

    return shiboken6.wrapInstance(int(ptr), QtWidgets.QMainWindow)


# ----------------------------------------------------------------------------


def maya_to_qt(name, type_=QtWidgets.QWidget):
    """
    Convert a Maya UI object to a Qt widget.

    Args:
        name (str): Maya UI object name.
        type_ (QWidget): Desired Qt widget type.

    Returns:
        QWidget

    Raises:
        RuntimeError: If no UI object is found.
    """
    ptr = OpenMayaUI.MQtUtil.findControl(name)

    if ptr is None:
        ptr = OpenMayaUI.MQtUtil.findLayout(name)

    if ptr is None:
        ptr = OpenMayaUI.MQtUtil.findMenuItem(name)

    if ptr is None:
        raise RuntimeError(f"Failed to obtain a handle to '{name}'.")

    return shiboken6.wrapInstance(int(ptr), type_)


def qt_to_maya(widget):
    """
    Convert a Qt widget back to its Maya UI name.

    Args:
        widget (QtWidgets.QWidget)

    Returns:
        str
    """
    ptr = shiboken6.getCppPointer(widget)[0]
    return OpenMayaUI.MQtUtil.fullName(int(ptr))