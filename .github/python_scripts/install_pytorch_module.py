import slicer.util

from MultiverSeg.SegmentEditorMultiverSegLib import DependenciesLogic

if __name__ == '__main__':
    DependenciesLogic.INTERACTIVE_MODE = False
    DependenciesLogic.installPyTorchExtensionIfNeeded()
    slicer.util.quit()
