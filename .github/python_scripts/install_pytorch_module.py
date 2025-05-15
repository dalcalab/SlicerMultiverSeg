import sys
from pathlib import Path

import slicer.util



if __name__ == '__main__':
    sys.path.append(Path(__file__).parent.joinpath("../..").resolve().as_posix())

    from MultiverSeg.SegmentEditorMultiverSegLib import DependenciesLogic
    DependenciesLogic.INTERACTIVE_MODE = False
    DependenciesLogic.installPyTorchExtensionIfNeeded()
    slicer.util.quit()
