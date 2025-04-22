import os

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *

# Needed to give package ref into instantiated segment
from SegmentEditorMultiverSegLib import SegmentationLogic, SegmentEditorEffect, ContextLogic

from slicer import vtkMRMLScalarVolumeNode


#
# MultiverSeg
#


class SegmentEditorMultiverSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MultiverSeg")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = [
            "Sebastien Goll (Kitware)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SegmentEditorMultiverSeg">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.registerEditorEffect)

    def registerEditorEffect(self):
        import qSlicerSegmentationsEditorEffectsPythonQt as qSlicerSegmentationsEditorEffects
        instance = qSlicerSegmentationsEditorEffects.qSlicerSegmentEditorScriptedEffect(None)
        effectFilename = os.path.join(os.path.dirname(__file__), self.__class__.__name__ + "Lib/SegmentEditorEffect.py")
        instance.setPythonSource(effectFilename.replace('\\', '/'))
        instance.self().register()


