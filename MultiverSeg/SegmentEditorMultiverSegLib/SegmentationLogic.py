import pathlib
import typing
from typing import Optional

import numpy as np
import slicer
import vtkAddon

from MRMLCorePython import vtkMRMLSegmentationNode, vtkMRMLScalarVolumeNode, vtkMRMLSliceNode
from numpy.ma.core import maximum
from vtkSegmentationCorePython import vtkSegment, vtkSegmentation

if typing.TYPE_CHECKING:
    # https://docs.python.org/3.9/library/typing.html#typing.TYPE_CHECKING
    # Allow typing in signatures without importing at runtime
    import torchvision.transforms.v2 as torchviz
    import torch


class SegmentationLogic:

    def __init__(self, scriptedEffect):
        self.scriptedEffect = scriptedEffect
        self.model = None

        self.segmentationNode: Optional[vtkMRMLSegmentationNode] = None

        self.workingView: Optional[str] = None

        self.negSegment: Optional[vtkSegment] = None
        self.posSegment: Optional[vtkSegment] = None
        self.resSegment: Optional[vtkSegment] = None

        self.predictionCache = None
        from .ContextLogic import ContextLogic
        self.contextLogic = ContextLogic(scriptedEffect)

        self.sliceOffsetRange = (0., 0.)

    def initSegments(self):
        """
        Initialize the segments by creating the positive and negative segments.
        """
        # Get the current segment
        self.segmentationNode: vtkMRMLSegmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        segmentation: vtkSegmentation = self.segmentationNode.GetSegmentation()
        segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
        self.resSegment = segmentation.GetSegment(segmentID)

        # Create positive segment
        self.posSegment = vtkSegment()
        self.posSegment.SetName(f"{self.resSegment.GetName()}_pos")
        self.posSegment.SetColor(137. / 255., 214. / 255., 60. / 255.)

        # Create negative segment
        self.negSegment = vtkSegment()
        self.negSegment.SetName(f"{self.resSegment.GetName()}_neg")
        self.negSegment.SetColor(214. / 255., 95. / 255., 60. / 255.)

        # Add new segment to segmentation
        segmentation.AddSegment(self.posSegment)
        segmentation.AddSegment(self.negSegment)

    def initModel(self):
        """
        Verify the dependencies and initialize the model.
        :return: True if the initialization was successful, False otherwise.
        """
        from .InstallLogic import InstallLogic, DependenciesLogic

        progress = slicer.util.createProgressDialog(maximum=10, labelText="Verifying dependencies")
        slicer.app.processEvents()

        if not DependenciesLogic.installDependenciesIfNeeded():
            progress.close()
            return False
        progress.value = 6
        progress.labelText = "Loading ScribblePrompt"
        slicer.app.processEvents()

        from scribbleprompt.models.unet import ScribblePromptUNet

        progress.value = 8
        progress.labelText = "Loading MultiverSeg"
        slicer.app.processEvents()
        from multiverseg.models.sp_mvs import MultiverSeg

        if not InstallLogic.downloadCheckpointsIfNeeded():
            progress.close()
            return False

        progress.value = 9
        progress.labelText = "Initialisation of the model"
        slicer.app.processEvents()

        # Update the path to the model weights
        MultiverSeg.weights["v0"] = pathlib.Path(__file__).parent.joinpath(
            "../Resources/Checkpoints/MultiverSeg_v0_nf256_res128.pt").resolve()
        ScribblePromptUNet.weights["v1"] = pathlib.Path(__file__).parent.joinpath(
            "../Resources/Checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt").resolve()
        self.model = MultiverSeg(version="v0")

        progress.close()
        return True

    def reset(self):
        """
        Remove the pos and neg segments and reset the internal state of the logic.
        """
        if self.segmentationNode is None:
            return

        # Remove the pos/neg segment
        self.segmentationNode.GetSegmentation().RemoveSegment(self.posSegment)
        self.segmentationNode.GetSegmentation().RemoveSegment(self.negSegment)

        # Remove the internal ref to the segments
        self.negSegment = None
        self.posSegment = None
        self.resSegment = None

    def setOffsetRange(self, min: float, max: float):
        self.sliceOffsetRange = (min, max)

    def predict(self):
        """
        Launch a 2D prediction for the current slice and view.
        """
        # Get the slice number
        import torchvision.transforms.v2 as torchviz

        k = self.getCurrentSliceIndex(self.workingView)
        y, originalDim = self.rawPredictForSlice(k)

        y = torchviz.functional.resize(y[0], originalDim)[0]
        self.predictionCache = y.clone()

        y = self.thresholdPrediction(y)

        volumeNode: vtkMRMLScalarVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        segNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        segmentId = segNode.GetSegmentation().GetSegmentIdBySegment(self.resSegment)
        resultSegment = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentId, volumeNode)

        IJKToRAS = np.zeros((3, 3))
        volumeNode.GetIJKToRASDirections(IJKToRAS)
        KJIToRAS = IJKToRAS.copy()
        KJIToRAS[:, 0] = IJKToRAS[:, 2]
        KJIToRAS[:, 2] = IJKToRAS[:, 0]

        sliceNodeID = f"vtkMRMLSliceNode{self.workingView}"
        sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
        axis = self.computeSliceAxis(volumeNode, sliceNode)

        resultSegment = self.updateSlice(resultSegment, y, k, axis)

        slicer.util.updateSegmentBinaryLabelmapFromArray(resultSegment, segNode, segmentId, volumeNode)

    def thresholdPrediction(self, prediction: "torch.Tensor", threshold=0.5):
        """
        Apply a threshold to the prediction.
        """
        prediction[prediction < threshold] = 0
        prediction[prediction >= threshold] = 1
        return prediction

    def rawPredictForSlice(self, sliceNumber: int) -> tuple["torch.Tensor", "torch.Size"]:
        """
        Make a prediction for a 2D slice without post-processing
        """
        # return the raw prediction and the original dimension of the slice (for resizing)
        import torch
        # Load the context
        contextImage, contextLabel = self.contextLogic.loadContext()
        if contextImage is not None:
            contextImage = contextImage[None].to(self.model.device) / 255
            contextLabel = contextLabel[None].to(self.model.device) / 255

        # Get the nodes and segment ids
        volumeNode: vtkMRMLScalarVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        segNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        segmentId = segNode.GetSegmentation().GetSegmentIdBySegment(self.resSegment)
        posSegId = segNode.GetSegmentation().GetSegmentIdBySegment(self.posSegment)
        negSegId = segNode.GetSegmentation().GetSegmentIdBySegment(self.negSegment)

        sliceNodeID = f"vtkMRMLSliceNode{self.workingView}"
        sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
        axis = self.computeSliceAxis(volumeNode, sliceNode)

        # Getting the different arrays
        # Array from slicer.util are K-J-I indexed
        resultSegment = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentId, volumeNode)
        imageArray = slicer.util.arrayFromVolume(volumeNode).copy()
        posArray = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, posSegId, volumeNode)
        negArray = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, negSegId, volumeNode)

        # Extract the slice corresponding to the current view
        imageSlice = self.extractSlice(imageArray, sliceNumber, axis)
        prevPredSlice = self.extractSlice(resultSegment, sliceNumber, axis)
        posSlice = self.extractSlice(posArray, sliceNumber, axis)
        negSlice = self.extractSlice(negArray, sliceNumber, axis)

        # Convertion to tensors
        imageTensor = torch.from_numpy(imageSlice)
        prevPredTensor = torch.from_numpy(prevPredSlice)
        posTensor = torch.from_numpy(posSlice)
        negTensor = torch.from_numpy(negSlice)

        originalDim = imageTensor.shape

        # Pre process
        imageTensor = self.preprocessSlice(imageTensor[None])
        posTensor = self.preprocessSlice(posTensor[None], isSegmentation=True)
        negTensor = self.preprocessSlice(negTensor[None], isSegmentation=True)
        prevPredTensor = self.preprocessSlice(prevPredTensor[None], isSegmentation=True)

        scribbles = torch.cat((posTensor, negTensor), dim=0)

        y = self.model.predict(imageTensor[None],
                               scribbles=scribbles[None],
                               mask_input=prevPredTensor[None],
                               context_images=contextImage,
                               context_labels=contextLabel,
                               return_logits=False).cpu()
        return y, originalDim

    def predict3d(self):
        """
        Make a 3D prediction
        """

        sliceNodeID = f"vtkMRMLSliceNode{self.workingView}"
        sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
        appLogic = slicer.app.applicationLogic()
        sliceLogic = appLogic.GetSliceLogic(sliceNode)
        startSlice = sliceLogic.GetSliceIndexFromOffset(self.sliceOffsetRange[0]) - 1
        endSlice = sliceLogic.GetSliceIndexFromOffset(self.sliceOffsetRange[1]) - 1
        startSlice, endSlice = sorted((startSlice, endSlice))

        # Load the context
        contextImage, contextLabel = self.contextLogic.loadContext()
        if contextImage is not None:
            contextImage = contextImage[None].to(self.model.device) / 255
            contextLabel = contextLabel[None].to(self.model.device) / 255

        # Get the nodes and segment ids
        volumeNode: vtkMRMLScalarVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        segNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        segmentId = segNode.GetSegmentation().GetSegmentIdBySegment(self.resSegment)
        posSegId = segNode.GetSegmentation().GetSegmentIdBySegment(self.posSegment)
        negSegId = segNode.GetSegmentation().GetSegmentIdBySegment(self.negSegment)

        axis = self.computeSliceAxis(volumeNode, sliceNode)

        # Getting the different arrays
        # Array from slicer.util are K-J-I indexed
        resultSegment = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentId, volumeNode)
        imageArray = slicer.util.arrayFromVolume(volumeNode).copy()
        posArray = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, posSegId, volumeNode)
        negArray = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, negSegId, volumeNode)

        imageSlice = self.extractSlice(imageArray, 0, axis)
        originalDim = imageSlice.shape

        import torch
        import torchvision.transforms.v2 as torchviz
        # Convertion to tensors
        imageTensor = torch.from_numpy(imageArray)
        prevPredTensor = torch.from_numpy(resultSegment)
        posTensor = torch.from_numpy(posArray)
        negTensor = torch.from_numpy(negArray)

        # Pre process
        imageTensor = self.preprocessVolume(imageTensor[None], axis)[0]
        posTensor = self.preprocessVolume(posTensor[None], axis, isSegmentation=True)[0]
        negTensor = self.preprocessVolume(negTensor[None], axis, isSegmentation=True)[0]
        prevPredTensor = self.preprocessVolume(prevPredTensor[None], axis, isSegmentation=True)[0]

        progressDialog = slicer.util.createProgressDialog(value=startSlice - 1,
                                                          minimum=startSlice - 1,
                                                          maximum=endSlice,
                                                          labelText="Running 3d prediction...",
                                                          windowModality=2  # qt.Qt.ApplicationModal
                                                          )

        linspace = np.linspace(self.sliceOffsetRange[0],
                               self.sliceOffsetRange[1],
                               endSlice - startSlice + 1,
                               endpoint=True)

        for sliceNumber, sliceOffset in zip(range(startSlice, endSlice + 1), linspace):
            # Switch view to slice
            sliceLogic.SetSliceOffset(sliceOffset)

            # Extract the slice corresponding to the current view
            imageSlice = self.extractSlice(imageTensor, sliceNumber, axis)[None]
            prevPredSlice = self.extractSlice(prevPredTensor, sliceNumber, axis)[None]
            posSlice = self.extractSlice(posTensor, sliceNumber, axis)[None]
            negSlice = self.extractSlice(negTensor, sliceNumber, axis)[None]

            scribbles = torch.cat((posSlice, negSlice), dim=0)

            y = self.model.predict(imageSlice[None],
                                   scribbles=scribbles[None],
                                   mask_input=prevPredSlice[None],
                                   context_images=contextImage,
                                   context_labels=contextLabel,
                                   return_logits=False).cpu()
            y = torchviz.functional.resize(y[0], originalDim)[0]
            y = self.thresholdPrediction(y)

            resultSegment = self.updateSlice(resultSegment, y, sliceNumber, axis)
            progressDialog.setValue(sliceNumber)

            if progressDialog.wasCanceled:
                progressDialog.close()
                break
            slicer.app.processEvents()

        slicer.util.updateSegmentBinaryLabelmapFromArray(resultSegment, segNode, segmentId, volumeNode)

    def getCurrentSliceIndex(self, sliceColor):
        """
        Get the index of the current slice for the view sliceColor based on the offset value.
        """
        sliceNodeID = f"vtkMRMLSliceNode{sliceColor}"

        sliceNode = slicer.mrmlScene.GetNodeByID(sliceNodeID)
        appLogic = slicer.app.applicationLogic()
        sliceLogic = appLogic.GetSliceLogic(sliceNode)
        sliceOffset = sliceLogic.GetSliceOffset()
        return sliceLogic.GetSliceIndexFromOffset(sliceOffset) - 1  # slice is 1-indexed

    def computeSliceAxis(self, volumeNode: vtkMRMLScalarVolumeNode, sliceNode: vtkMRMLSliceNode):
        """
        Given the volume node and the slice node, find the axis of the volume which correspond to the stepping direction in the selected view.
        """
        # Get the slice normal vector in RAS
        sliceToRAS = sliceNode.GetSliceToRAS()
        sliceNormal = np.zeros(4)
        vtkAddon.vtkAddonMathUtilities.GetOrientationMatrixColumn(sliceToRAS, 2, sliceNormal)

        # Get the KIJ to RAS matrix
        IJKToRAS = np.zeros((3, 3))
        volumeNode.GetIJKToRASDirections(IJKToRAS)
        KJIToRAS = IJKToRAS.copy()
        KJIToRAS[:, 0] = IJKToRAS[:, 2]
        KJIToRAS[:, 2] = IJKToRAS[:, 0]

        res = KJIToRAS.T @ sliceNormal[:3]
        res = np.abs(res)

        if np.allclose(res, [1, 0, 0], atol=0.01):
            return 0
        if np.allclose(res, [0, 1, 0], atol=0.01):
            return 1
        if np.allclose(res, [0, 0, 1], atol=0.01):
            return 2
        raise ValueError(f"View {self.workingView} is not axis aligned with the volume geometry")

    def extractSlice(self, array: np.ndarray, sliceNumber: int, axis: int):
        """Extract the slice sliceNumber from the array given an axis"""
        return np.take(array, sliceNumber, axis=axis)

    def updateSlice(self, array: np.ndarray, newSlice: np.ndarray, sliceNumber: int, axis: int):
        """
        Replace the slice in array by the newSlice. sliceNumber and axis are for positional information.
        """
        if axis == 0:
            array[sliceNumber] = newSlice
        elif axis == 1:
            array[:, sliceNumber] = newSlice
        elif axis == 2:
            array[:, :, sliceNumber] = newSlice
        else:
            slicer.util.errorDisplay(f"Error during segmentation update, axis {axis} was given")
        return array

    def preprocessSlice(self, slice: "torch.Tensor", isSegmentation=False):
        """
        Preprocess a 2d slice for the model. If isSegmentation, the resulting Tensor in of type bool
        """
        # Slice of dim  of shape 1*W*H
        import torch
        import torchvision.transforms.v2 as torchviz
        if isSegmentation:
            targetDtype = torch.bool
        else:
            targetDtype = torch.float16

        # Resizing
        result = torchviz.functional.resize(slice, size=[128, 128]).to(targetDtype)

        # Bring the values between 0 and 1
        if not isSegmentation:
            result -= torch.min(result)
            result /= torch.max(result)

        return result  # 1*W*H

    def preprocessVolume(self, volume: "torch.Tensor", axis: int, isSegmentation=False):
        """
        Apply the preprocessing pipeline on a full volume given an axis. The direction of the axis is not rescaled to allow stepping through each slice.
        """
        # volume indexed RAS of shape 1*X*Y*Z
        import torch
        if isSegmentation:
            targetDtype = torch.bool
        else:
            targetDtype = torch.float16

        originalSize = volume.shape

        targetSize = [128, 128, 128]
        targetSize[axis] = originalSize[axis + 1]

        # Resizing
        result = torch.nn.functional.interpolate(volume[None].to(torch.float), targetSize, mode='trilinear').to(
            targetDtype)

        # Bring the values between 0 and 1
        if not isSegmentation:
            result -= torch.min(result)
            result /= torch.max(result)

        return result[0]  # 1*X*Y*Z
