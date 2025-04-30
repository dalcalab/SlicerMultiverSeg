import unittest
from unittest.mock import Mock
from pathlib import Path

import SampleData
import slicer.util
import torch
from SegmentEditorMultiverSegLib import SegmentationLogic


class SegmentationLogicTestCase(unittest.TestCase):
    def test_instantiation(self):
        logic = SegmentationLogic(None)

    def test_loadingModels(self):
        logic = SegmentationLogic(None)
        res = logic.initModel()

        self.assertTrue(res)
        self.assertIsNotNone(logic.model)

    def test_initSegments(self):
        dataPath = Path(__file__).parent.joinpath("../TestData/SlicerData").resolve()
        segmentationNode = slicer.util.loadSegmentation(dataPath.joinpath("1.seg.nrrd"))
        segmentID = segmentationNode.GetSegmentation().GetNthSegmentID(0)

        scriptedEffectMock = Mock()
        scriptedEffectMock.parameterSetNode().GetSegmentationNode.return_value = segmentationNode
        scriptedEffectMock.parameterSetNode().GetSelectedSegmentID.return_value = segmentID
        logic = SegmentationLogic(scriptedEffectMock)

        self.assertEqual(segmentationNode.GetSegmentation().GetNumberOfSegments(), 1)
        logic.initSegments()
        self.assertEqual(segmentationNode.GetSegmentation().GetNumberOfSegments(), 3)
        self.assertIsNotNone(logic.posSegment)
        self.assertIsNotNone(logic.negSegment)

        logic.reset()
        self.assertIsNone(logic.negSegment)
        self.assertIsNone(logic.posSegment)
        self.assertIsNone(logic.resSegment)
        self.assertEqual(segmentationNode.GetSegmentation().GetNumberOfSegments(), 1)

    def test_slicePreprocess(self):
        logic = SegmentationLogic(None)

        dummySlice = torch.rand([1, 512, 512])
        res = logic.preprocessSlice(dummySlice)
        self.assertSequenceEqual(res.shape, [1, 128, 128])
        self.assertAlmostEquals(torch.max(res).item(), 1)
        self.assertAlmostEquals(torch.min(res).item(), 0)
        self.assertIs(res.dtype, torch.float16)

        dummySlice = torch.rand([1, 129, 1000]) * 500 - 250
        res = logic.preprocessSlice(dummySlice)
        self.assertSequenceEqual(res.shape, [1, 128, 128])
        self.assertAlmostEquals(torch.max(res).item(), 1)
        self.assertAlmostEquals(torch.min(res).item(), 0)
        self.assertIs(res.dtype, torch.float16)

        dummySlice = torch.randint(2, [1, 129, 1000])
        res = logic.preprocessSlice(dummySlice, isSegmentation=True)
        self.assertSequenceEqual(res.shape, [1, 128, 128])
        self.assertIs(res.dtype, torch.bool)

    def test_volumePreprocess(self):
        logic = SegmentationLogic(None)

        logic.workingView = "Red"
        dummyVolume = torch.rand([1, 512, 512, 512])
        res = logic.preprocessVolume(dummyVolume)
        self.assertSequenceEqual(res.shape, [1, 128, 128, 512])
        self.assertIs(res.dtype, torch.float16)
        self.assertAlmostEquals(torch.max(res).item(), 1)
        self.assertAlmostEquals(torch.min(res).item(), 0)

        logic.workingView = "Green"
        dummyVolume = torch.rand([1, 1000, 43, 100])
        res = logic.preprocessVolume(dummyVolume)
        self.assertSequenceEqual(res.shape, [1, 128, 43, 128])
        self.assertIs(res.dtype, torch.float16)
        self.assertAlmostEquals(torch.max(res).item(), 1)
        self.assertAlmostEquals(torch.min(res).item(), 0)

        logic.workingView = "Yellow"
        dummyVolume = torch.randint(2, [1, 1000, 43, 100])
        res = logic.preprocessVolume(dummyVolume, isSegmentation=True)
        self.assertSequenceEqual(res.shape, [1, 1000, 128, 128])
        self.assertIs(res.dtype, torch.bool)

    def test_updateSlice(self):
        logic = SegmentationLogic(None)

        baseVolume = torch.rand([55, 66, 77])

        logic.workingView = "Red"
        updatedSlice = torch.zeros([55, 66])
        result = logic.updateSlice(baseVolume, updatedSlice, 25)
        self.assertSequenceEqual(result.shape, [55, 66, 77])
        self.assertEqual(torch.max(result[:, :, 25]).item(), 0)

        logic.workingView = "Green"
        updatedSlice = torch.zeros([55, 77])
        result = logic.updateSlice(baseVolume, updatedSlice, 20)
        self.assertSequenceEqual(result.shape, [55, 66, 77])
        self.assertEqual(torch.max(result[:, 20]).item(), 0)

        logic.workingView = "Yellow"
        updatedSlice = torch.zeros([66, 77])
        result = logic.updateSlice(baseVolume, updatedSlice, 30)
        self.assertSequenceEqual(result.shape, [55, 66, 77])
        self.assertEqual(torch.max(result[30]).item(), 0)

    def test_extractSlice(self):
        logic = SegmentationLogic(None)

        baseVolume = torch.rand([55, 66, 77])

        logic.workingView = "Red"
        result = logic.extractSlice(baseVolume, 25)
        self.assertSequenceEqual(result.shape, [55, 66])

        logic.workingView = "Green"
        result = logic.extractSlice(baseVolume, 20)
        self.assertSequenceEqual(result.shape, [55, 77])

        logic.workingView = "Yellow"
        result = logic.extractSlice(baseVolume, 30)
        self.assertSequenceEqual(result.shape, [66, 77])

    def test_rawPredict(self):
        dataPath = Path(__file__).parent.joinpath("../TestData/SlicerData").resolve()
        segmentationNode = slicer.util.loadSegmentation(dataPath.joinpath("1.seg.nrrd"))
        volNode = SampleData.downloadSample("MRHead")
        segmentID = segmentationNode.GetSegmentation().GetNthSegmentID(0)

        scriptedEffectMock = Mock()
        scriptedEffectMock.parameterSetNode().GetSourceVolumeNode.return_value = volNode
        scriptedEffectMock.parameterSetNode().GetSegmentationNode.return_value = segmentationNode
        scriptedEffectMock.parameterSetNode().GetSelectedSegmentID.return_value = segmentID
        logic = SegmentationLogic(scriptedEffectMock)

        logic.initSegments()
        logic.initModel()

        logic.workingView = "Red"
        res, originalDim = logic.rawPredictForSlice(20)
        self.assertSequenceEqual(res.shape, [1,1,128,128])

    def test_predict(self):
        dataPath = Path(__file__).parent.joinpath("../TestData/SlicerData").resolve()
        segmentationNode = slicer.util.loadSegmentation(dataPath.joinpath("1.seg.nrrd"))
        volNode = SampleData.downloadSample("MRHead")
        segmentID = segmentationNode.GetSegmentation().GetNthSegmentID(0)

        scriptedEffectMock = Mock()
        scriptedEffectMock.parameterSetNode().GetSourceVolumeNode.return_value = volNode
        scriptedEffectMock.parameterSetNode().GetSegmentationNode.return_value = segmentationNode
        scriptedEffectMock.parameterSetNode().GetSelectedSegmentID.return_value = segmentID
        logic = SegmentationLogic(scriptedEffectMock)

        logic.initSegments()
        logic.initModel()

        logic.workingView = "Red"
        logic.predict()

    def test_3dPredict(self):
        dataPath = Path(__file__).parent.joinpath("../TestData/SlicerData").resolve()
        segmentationNode = slicer.util.loadSegmentation(dataPath.joinpath("1.seg.nrrd"))
        volNode = SampleData.downloadSample("MRHead")
        segmentID = segmentationNode.GetSegmentation().GetNthSegmentID(0)

        scriptedEffectMock = Mock()
        scriptedEffectMock.parameterSetNode().GetSourceVolumeNode.return_value = volNode
        scriptedEffectMock.parameterSetNode().GetSegmentationNode.return_value = segmentationNode
        scriptedEffectMock.parameterSetNode().GetSelectedSegmentID.return_value = segmentID
        logic = SegmentationLogic(scriptedEffectMock)

        logic.initSegments()
        logic.initModel()
        logic.setOffsetRange(-10,10)

        logic.workingView = "Red"
        logic.predict3d()