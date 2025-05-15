from pathlib import Path

import slicer.util
import sys

if __name__ == '__main__':
    sys.path.append(Path(__file__).parent.joinpath("../..").resolve().as_posix())

    from MultiverSeg.SegmentEditorMultiverSegLib import InstallLogic, DependenciesLogic
    DependenciesLogic.INTERACTIVE_MODE = False
    InstallLogic.INTERACTIVE_MODE = False

    DependenciesLogic.installTorchIfNeeded()
    DependenciesLogic.installMultiverSegIfNeeded()
    InstallLogic.downloadCheckpointsIfNeeded()
    slicer.util.quit()
