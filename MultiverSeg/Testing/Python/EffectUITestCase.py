import unittest
from unittest.mock import Mock

import SampleData
import slicer
from SegmentEditorMultiverSegLib import SegmentEditorEffect
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
from vtkSegmentationCorePython import vtkSegment


class SegmentEditorEffectUITest(ScriptedLoadableModuleTest):

    @classmethod
    def setUpClass(cls):
        vol = SampleData.downloadSample("MRBrainTumor1")
        cls.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        cls.segmentationNode.CreateDefaultDisplayNodes()
        cls.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        segment = vtkSegment()
        segment.SetName("Test")
        cls.segmentationNode.GetSegmentation().AddSegment(segment, "Test")

    def setUp(self):
        self.sew = slicer.qMRMLSegmentEditorWidget()
        self.sew.show()
        self.sew.setMRMLScene(slicer.mrmlScene)

        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        self.sew.setMRMLSegmentEditorNode(segmentEditorNode)

        self.sew.setSegmentationNode(self.segmentationNode)
        self.sew.setActiveEffectByName("MultiverSeg")
        self.effect = self.sew.activeEffect().self()

    def test_sanity(self):
        self.assertIsNotNone(self.effect)
        print(self.effect)

        self.assertIsInstance(self.effect, SegmentEditorEffect)

    def test_predictionButtonsDisabled(self):
        # Mocks
        self.effect.segmentationLogic = Mock()
        self.effect.segmentationLogic.initModel.return_value = True
        self.effect.contextLogic = Mock()
        self.effect.contextLogic.getCurrentContextSize.return_value = 5

        # Should be disabled before initialization
        self.assertFalse(self.effect.predictBtn.isEnabled())
        self.assertFalse(self.effect.predict3dBtn.isEnabled())

        # Should be disabled after initialization since no context is selected
        self.effect.initializeBtn.click()
        self.assertFalse(self.effect.predictBtn.isEnabled())
        self.assertFalse(self.effect.predict3dBtn.isEnabled())

        # Should be enabled after changing context since context size is 5
        self.effect.contextComboBox.setCurrentIndex(1)
        self.assertTrue(self.effect.predictBtn.isEnabled())
        self.assertTrue(self.effect.predict3dBtn.isEnabled())

        # Should be disabled after changing context since context size is 0
        self.effect.contextLogic.getCurrentContextSize.return_value = 0
        self.effect.contextComboBox.setCurrentIndex(2)
        self.assertFalse(self.effect.predictBtn.isEnabled())
        self.assertFalse(self.effect.predict3dBtn.isEnabled())


if __name__ == '__main__':
    unittest.main()
