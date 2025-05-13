import slicer.util
from MultiverSeg.SegmentEditorMultiverSegLib import InstallLogic, DependenciesLogic

if __name__ == '__main__':
    DependenciesLogic.INTERACTIVE_MODE = False
    InstallLogic.INTERACTIVE_MODE = False

    DependenciesLogic.installTorchIfNeeded()
    DependenciesLogic.installMultiverSegIfNeeded()
    InstallLogic.downloadCheckpointsIfNeeded()
    slicer.util.quit()
