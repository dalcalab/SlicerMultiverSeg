import pathlib

import numpy as np
import qt
import slicer

from MRMLCorePython import vtkMRMLVolumeNode, vtkMRMLSegmentationNode
from .SegmentationLogic import SegmentationLogic


class ContextLogic:

    def __init__(self, scriptedEffect):
        self.scriptedEffect = scriptedEffect

        self.contextRootPath = pathlib.Path(__file__).parent.joinpath("../Context").resolve()
        self.activeContext = None

    def loadContext(self):

        import torch, torchvision

        if self.activeContext is None or self.getCurrentContextSize() == 0:
            return None, None

        contextPath = self.contextRootPath.joinpath(self.activeContext)
        contextImgs = sorted(contextPath.glob("image*"))
        contextLabels = sorted(contextPath.glob("mask*"))

        i = [torchvision.io.decode_image(img, mode=torchvision.io.ImageReadMode.GRAY) for img in contextImgs]
        l = [torchvision.io.decode_image(lab, mode=torchvision.io.ImageReadMode.GRAY) for lab in contextLabels]
        return torch.stack(i), torch.stack(l)

    def getContextList(self):
        tasks = []

        for item in self.contextRootPath.glob("*"):
            if item.is_dir():
                tasks.append(item.name)

        return sorted(tasks)

    def getCurrentContextSize(self):
        contextPath = self.contextRootPath.joinpath(self.activeContext)

        return len(sorted(contextPath.glob("image*.*")))

    def getNextExampleNumber(self):
        contextPath = self.contextRootPath.joinpath(self.activeContext)

        imagesPaths = sorted(contextPath.glob("image*.*"))
        if len(imagesPaths) == 0:
            return 0
        imagesNames = map(lambda x: x.name, imagesPaths)
        imagesNumbers = map(lambda x: int(x.split("_")[-1].split('.')[0]), imagesNames)
        return max(imagesNumbers) + 1

    def saveNewExample(self, volume: vtkMRMLVolumeNode, view, segmentID, segmentationNode: vtkMRMLSegmentationNode,
                       segLogic: SegmentationLogic):
        import torch
        import torchvision
        assert self.activeContext is not None
        contextPath = self.contextRootPath.joinpath(self.activeContext)

        k = segLogic.getCurrentSliceIndex(view)

        imageArray = slicer.util.arrayFromVolume(volume).copy()
        maskArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentID)

        IJKToRAS = np.zeros((3, 3))
        volume.GetIJKToRASDirections(IJKToRAS)
        KJIToRAS = IJKToRAS.copy()
        KJIToRAS[:, 0] = IJKToRAS[:, 2]
        KJIToRAS[:, 2] = IJKToRAS[:, 0]

        imageArray = segLogic.reorderAxisToRAS(imageArray, KJIToRAS)
        maskArray = segLogic.reorderAxisToRAS(maskArray, KJIToRAS)

        imageTensor = torch.from_numpy(segLogic.extractSlice(imageArray, k, view))
        maskTensor = torch.from_numpy(segLogic.extractSlice(maskArray, k, view))

        imageTensor = segLogic.preprocessSlice(imageTensor[None])
        maskTensor = segLogic.preprocessSlice(maskTensor[None])

        n = self.getNextExampleNumber()

        torchvision.utils.save_image(imageTensor, contextPath.joinpath(f"image_{n}.png"))
        torchvision.utils.save_image(maskTensor.to(torch.float16) * 255, contextPath.joinpath(f"mask_{n}.png"))

    def exportContext(self):
        assert self.activeContext is not None, "A context must be selected to export"

        dir = qt.QFileDialog().getExistingDirectory(None, "Export to:", ".",
                                                    qt.QFileDialog().ShowDirsOnly + qt.QFileDialog().ReadOnly)

        if dir:
            import shutil
            shutil.make_archive(pathlib.Path(dir).joinpath(self.activeContext), "zip",
                                root_dir=self.contextRootPath.joinpath(self.activeContext))

    def importContext(self):
        file = qt.QFileDialog().getOpenFileName(None, "Import context", ".", "*.zip")

        if file:
            import shutil
            file = pathlib.Path(file)
            shutil.unpack_archive(file, self.contextRootPath.joinpath(file.stem), "zip")
            return file.stem
        return ''

    def deleteCurrentTask(self):
        import shutil
        shutil.rmtree(self.contextRootPath.joinpath(self.activeContext))

    def createTask(self, name: str):

        destDir = self.contextRootPath.joinpath(name)

        if destDir.is_dir():
            return False

        destDir.mkdir()
        return True

    def renameTask(self, newName):
        import shutil

        currentPath = self.contextRootPath.joinpath(self.activeContext)
        newPath = self.contextRootPath.joinpath(newName)
        if newPath.is_dir():
            return False

        shutil.move(currentPath, newPath)
        return True
