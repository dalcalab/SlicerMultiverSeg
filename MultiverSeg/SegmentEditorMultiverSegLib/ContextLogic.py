import os
import pathlib

import numpy as np
import slicer

from MRMLCorePython import vtkMRMLVolumeNode, vtkMRMLSegmentationNode
from .SegmentationLogic import SegmentationLogic


class ContextLogic:

    def __init__(self, scriptedEffect):
        self.scriptedEffect = scriptedEffect

        self.contextRootPath = pathlib.Path(__file__).parent.joinpath("../Context").resolve()
        self.activeContext = None

    def setContextRoot(self, path):
        self.contextRootPath = path

    def computeBaseContextRoot(self)->str:
        return pathlib.Path(__file__).parent.joinpath("../Context").resolve().as_posix()

    def loadImage(self, path: pathlib.Path):
        import torchvision
        try:
            from packaging.version import Version
        except ModuleNotFoundError:
            print("Installing packaging")
            slicer.util.pip_install("packaging")
            from packaging.version import Version

        # Use different method depending on the torchvision version (changed in 0.18)
        # Return a tensor
        torchvisionVersion = Version(torchvision.__version__)
        if torchvisionVersion >= Version("0.18"):
            return torchvision.io.decode_image(path.as_posix(), mode=torchvision.io.ImageReadMode.GRAY)
        else:
            return torchvision.io.read_image(path.as_posix(), mode=torchvision.io.ImageReadMode.GRAY)

    def loadContext(self):

        import torch

        if self.activeContext is None or self.getCurrentContextSize() == 0:
            return None, None

        contextPath = self.contextRootPath.joinpath(self.activeContext)
        contextImgs = sorted(contextPath.glob("image*"))
        contextLabels = sorted(contextPath.glob("mask*"))

        i = [self.loadImage(img) for img in contextImgs]
        l = [self.loadImage(lab) for lab in contextLabels]
        return torch.stack(i), torch.stack(l)  # (n*1*H*W), (n*1*H*W)

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
        maskArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentID, volume)

        sliceNodeID = f"vtkMRMLSliceNode{view}"
        sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
        axis = segLogic.computeSliceAxis(volume, sliceNode)

        imageTensor = torch.from_numpy(segLogic.extractSlice(imageArray, k, axis))
        maskTensor = torch.from_numpy(segLogic.extractSlice(maskArray, k, axis))

        imageTensor = segLogic.preprocessSlice(imageTensor[None])
        maskTensor = segLogic.preprocessSlice(maskTensor[None])

        n = self.getNextExampleNumber()

        torchvision.utils.save_image(imageTensor, contextPath.joinpath(f"image_{n}.png"))
        torchvision.utils.save_image(maskTensor.to(torch.float16) * 255, contextPath.joinpath(f"mask_{n}.png"))

    def removeExample(self, exampleNumber: int):

        assert self.activeContext is not None, "A context must be selected to proceed"

        contextPath = self.contextRootPath.joinpath(self.activeContext)
        imagePath = contextPath.joinpath(f"image_{exampleNumber}.png")
        maskPath = contextPath.joinpath(f"mask_{exampleNumber}.png")

        if not (imagePath.is_file() and maskPath.is_file()):
            raise FileNotFoundError(f"{imagePath} or {maskPath} was not found.")

        os.remove(imagePath)
        os.remove(maskPath)

    def exportContext(self, dir: str):
        assert self.activeContext is not None, "A context must be selected to export"

        if not dir:
            return False

        import shutil

        if pathlib.Path(dir).joinpath(self.activeContext+".zip").is_file():
            raise FileExistsError(f"File {self.activeContext}.zip already exist")

        shutil.make_archive(pathlib.Path(dir).joinpath(self.activeContext), "zip",
                            root_dir=self.contextRootPath.joinpath(self.activeContext))
        return True

    def importContext(self, fileName: str):

        if not fileName:
            return ''

        import shutil
        file = pathlib.Path(fileName)

        if not file.is_file(): raise FileNotFoundError(f"File {fileName} does not exist.")

        dest = self.contextRootPath.joinpath(file.stem)

        if dest.is_dir(): raise IsADirectoryError(f"Task {file.stem} already exist.")

        shutil.unpack_archive(file, dest, "zip")
        return file.stem

    def deleteCurrentTask(self):
        import shutil
        shutil.rmtree(self.contextRootPath.joinpath(self.activeContext))

    def createTask(self, name: str):

        destDir = self.contextRootPath.joinpath(name)

        if destDir.is_dir():
            return False

        destDir.mkdir(parents=True)
        return True

    def renameTask(self, newName):
        import shutil

        currentPath = self.contextRootPath.joinpath(self.activeContext)
        newPath = self.contextRootPath.joinpath(newName)
        if newPath.is_dir():
            return False

        shutil.move(currentPath, newPath)
        return True
