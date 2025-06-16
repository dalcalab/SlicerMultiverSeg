import os

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *

# Needed to give package ref into instantiated segment
from SegmentEditorMultiverSegLib import SegmentationLogic, SegmentEditorEffect, ContextLogic

from slicer import vtkMRMLScalarVolumeNode


class SegmentEditorMultiverSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MultiverSeg")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["Sebastien Goll (Kitware)"]
        self.parent.hidden = True
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("This segment editor effect allow the use of the MultiverSeg model for segmentation")
        self.parent.acknowledgementText = _("""Wong, Hallee E., et al.
"MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance."
arXiv preprint arXiv:2412.15058 (2024).
This work was funded by the Cure Overgrowth Syndromes (COSY) RHU Project (ANR-18-RHUS-005)""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.registerEditorEffect)

    def registerEditorEffect(self):
        import qSlicerSegmentationsEditorEffectsPythonQt as qSlicerSegmentationsEditorEffects
        instance = qSlicerSegmentationsEditorEffects.qSlicerSegmentEditorScriptedEffect(None)
        effectFilename = os.path.join(os.path.dirname(__file__), self.__class__.__name__ + "Lib/SegmentEditorEffect.py")
        instance.setPythonSource(effectFilename.replace("\\", "/"))
        instance.self().register()


