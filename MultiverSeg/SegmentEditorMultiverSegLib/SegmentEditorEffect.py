import pathlib

import slicer.util
from MRMLLogicPython import vtkMRMLSliceLogic

from SegmentEditorEffects import *
from MRMLCorePython import *

from vtkSlicerSegmentationsModuleMRMLPython import vtkMRMLSegmentEditorNode
import ctk

from .InstallLogic import DependenciesLogic
from .SegmentationLogic import SegmentationLogic


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):

    def __init__(self, scriptedEffect):
        scriptedEffect.name = "MultiverSeg"
        scriptedEffect.requireSegments = True
        scriptedEffect.perSegment = True
        scriptedEffect.showEffectCursorInSliceView = False
        scriptedEffect.showEffectCursorInThreeDView = False
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

        self.segmentationLogic = SegmentationLogic(scriptedEffect)
        self.contextLogic = self.segmentationLogic.contextLogic

        self.isInitialized = False
        self.sew = None  # qMRMLSegmentEditorWidget

    def clone(self):
        # It should not be necessary to modify this method
        import qSlicerSegmentationsEditorEffectsPythonQt as effects
        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__.replace('\\', '/'))
        return clonedEffect

    def helpText(self):
        return """<html>Use MultiverSeg model to segment a slice</html>"""

    def icon(self):
        # Icon of the effect
        iconPath = self.getIconPath("SegmentEditorMultiverSeg.png")
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def getIconPath(self, iconName: str):
        return pathlib.Path(__file__).parent.joinpath(f"../Resources/Icons/{iconName}").resolve()

    def createIconButton(self, iconName, isCheckable=False, toolTip=''):
        b = qt.QPushButton()
        b.setIcon(qt.QIcon(self.getIconPath(iconName)))
        b.setCheckable(isCheckable)
        b.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
        b.setToolTip(toolTip)
        return b

    def sourceVolumeNodeChanged(self):
        # Called
        if self.sew is not None: return

        self.sew = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        if self.sew is not None:
            self.sew.connect("currentSegmentIDChanged(QString)", self.tryToPreselectTask)

    def tryToPreselectTask(self, segmentID):
        sen = self.scriptedEffect.parameterSetNode()  # SegmentEditorNode
        assert segmentID == sen.GetSelectedSegmentID()
        segmentation = self.scriptedEffect.parameterSetNode().GetSegmentationNode().GetSegmentation()
        segName = segmentation.GetSegment(segmentID).GetName()

        self.contextComboBox.setCurrentText(segName) # Does not change if segName do not exist

    def setupOptionsFrame(self):
        # First row: view selection and initialization
        self.viewComboBox = qt.QComboBox()
        self.viewComboBox.addItems(["Red", "Green", "Yellow"])
        self.initializeBtn = qt.QPushButton("Initialize")
        initLayer = qt.QHBoxLayout()
        initLayer.addWidget(self.viewComboBox)
        initLayer.addWidget(self.initializeBtn)

        # Second row: context selection
        self.contextComboBox = qt.QComboBox()
        self.contextComboBox.addItems(["--None--"] + self.contextLogic.getContextList())
        self.editTaskButton = self.createIconButton("edit.png", True, toolTip="Edit the tasks and contexts")
        self.chooseContextRoot = self.createIconButton("folder_tree.png",
                                                       toolTip="Select a custom root directory for your tasks")
        contextSelectionLayer = qt.QHBoxLayout()
        contextSelectionLayer.addWidget(self.contextComboBox)
        contextSelectionLayer.addWidget(self.editTaskButton)
        contextSelectionLayer.addWidget(self.chooseContextRoot)

        # Warning for the user if no task is selected
        self.warningEmptyTask = qt.QLabel("Warning: No context selected - Prediction disabled")
        self.warningEmptyTask.setWordWrap(True)
        self.warningEmptyTask.setForegroundRole(qt.QPalette.BrightText)

        # Warning for the user if the current task has empty context
        self.warningEmptyContext = qt.QLabel(
            "Warning: Empty context - Prediction disabled")
        self.warningEmptyContext.setWordWrap(True)
        self.warningEmptyContext.setForegroundRole(qt.QPalette.BrightText)

        # Buttons to manage tasks and contexts from withing slicer
        self.addTaskButton = self.createIconButton("add.png", toolTip="Add a new task")
        self.removeTaskButton = self.createIconButton("remove.png", toolTip="Delete the current task")
        self.importTaskButton = self.createIconButton("import.png", toolTip="Import a task and context")
        self.exportTaskButton = self.createIconButton("export.png", toolTip="Export the current task and context")
        self.renameTaskButton = self.createIconButton("rename.png", toolTip="Rename the current task")
        self.addContextExampleButton = self.createIconButton("addImage.png", toolTip="Add an example to the context")
        self.removeContextExampleButton = self.createIconButton("removeImage.png",
                                                                toolTip="Remove an example from the context")

        # Add the button to the layout
        contextModificationLayer = qt.QHBoxLayout()
        contextModificationLayer.addWidget(self.addTaskButton)
        contextModificationLayer.addWidget(self.removeTaskButton)
        contextModificationLayer.addWidget(self.importTaskButton)
        contextModificationLayer.addWidget(self.exportTaskButton)
        contextModificationLayer.addWidget(self.renameTaskButton)
        contextModificationLayer.addWidget(self.addContextExampleButton)
        contextModificationLayer.addWidget(self.removeContextExampleButton)

        # Create a range selector for the 3d segmentation
        self.sliceRangeSelector = ctk.ctkDoubleRangeSlider()
        self.sliceRangeSelector.orientation = qt.Qt.Horizontal
        self.sliceRangeSelector.maximum = 0
        self.sliceRangeSelector.minimum = 0

        # Create the buttons for prediction and reset
        self.predictBtn = qt.QPushButton("Make 2D prediction (current slice)")
        self.predict3dBtn = qt.QPushButton("Make 3D prediction")
        self.doneBtn = qt.QPushButton("Done")

        # Add the widgets to the effect's layout
        self.scriptedEffect.addLabeledOptionsWidget("View to segment: ", initLayer)
        self.scriptedEffect.addLabeledOptionsWidget("Current task: ", contextSelectionLayer)
        self.scriptedEffect.addOptionsWidget(self.warningEmptyTask)
        self.scriptedEffect.addOptionsWidget(self.warningEmptyContext)
        self.scriptedEffect.addOptionsWidget(contextModificationLayer)
        self.scriptedEffect.addOptionsWidget(self.predictBtn)
        self.scriptedEffect.addLabeledOptionsWidget("Slice range to predict: ", self.sliceRangeSelector)
        self.scriptedEffect.addOptionsWidget(self.predict3dBtn)
        self.scriptedEffect.addOptionsWidget(self.doneBtn)

        # Connect actions for task management
        self.editTaskButton.connect("toggled(bool)", self.handleTaskEditionMode)
        self.chooseContextRoot.connect("clicked()", self.selectContextRoot)
        self.addTaskButton.connect("clicked()", self.addTask)
        self.removeTaskButton.connect("clicked()", self.removeTask)
        self.importTaskButton.connect("clicked()", self.importContext)
        self.exportTaskButton.connect("clicked()", self.exportContext)
        self.renameTaskButton.connect("clicked()", self.renameTask)
        self.addContextExampleButton.connect("clicked()", self.addImageToContext)
        self.removeContextExampleButton.connect("clicked()", self.removeImageFromContext)

        # Connect other component actions
        self.contextComboBox.connect("currentIndexChanged(int)", self.handleTaskChange)
        self.initializeBtn.connect("clicked()", self.onInit)
        self.doneBtn.connect("clicked()", self.onDone)
        self.predictBtn.connect("clicked()", self.segmentationLogic.predict)
        self.predict3dBtn.connect("clicked()", self.segmentationLogic.predict3d)

        # Connect range selector actions
        self.sliceRangeSelector.connect("minimumValueChanged(double)", self.setSliceOffset)
        self.sliceRangeSelector.connect("maximumValueChanged(double)", self.setSliceOffset)
        self.sliceRangeSelector.connect("valuesChanged(double, double)", self.segmentationLogic.setOffsetRange)

        # Disable some widgets at initialization
        self.predictBtn.setEnabled(False)
        self.doneBtn.setEnabled(False)
        self.predict3dBtn.setEnabled(False)
        self.sliceRangeSelector.setEnabled(False)
        self.handleTaskEditionMode(False)
        self.handleTaskChange(0)

    def onInit(self):
        # Called when click "initialize" button

        # Installation of PyTorch Utils extension used to install pytorch
        canContinue = DependenciesLogic.installPyTorchExtensionIfNeeded()
        if not canContinue:
            return

        # Initialize the segmentation logic
        isModelInitialized = self.segmentationLogic.initModel()
        if not isModelInitialized: return
        self.segmentationLogic.initSegments()

        # Change handle the selected view
        self.changeViewLayout()
        self.segmentationLogic.workingView = self.viewComboBox.currentText

        # Update the ui elements now enabled
        self.viewComboBox.setEnabled(False)
        self.initializeBtn.setEnabled(False)
        self.doneBtn.setEnabled(True)
        self.sliceRangeSelector.setEnabled(True)

        # Get the min/max offset for the view, and the spacing
        sliceColor = self.viewComboBox.currentText
        sliceNode: vtkMRMLSliceNode = slicer.mrmlScene.GetNodeByID(f"vtkMRMLSliceNode{sliceColor}")
        sliceLogic: vtkMRMLSliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
        range = [0, 0]
        resolution = vtk.reference(0)
        sliceLogic.GetSliceOffsetRangeResolution(range, resolution)

        # Set the min max and spacing values for the slider
        offset = sliceLogic.GetSliceOffset()
        self.sliceRangeSelector.minimum = range[0]
        self.sliceRangeSelector.minimumValue = range[0]
        self.sliceRangeSelector.maximum = range[1]
        self.sliceRangeSelector.maximumValue = range[1]
        self.sliceRangeSelector.singleStep = resolution
        sliceLogic.SetSliceOffset(offset)

        # Set the overlap mode to allow overlaps
        segmentEditorNode: vtkMRMLSegmentEditorNode = self.scriptedEffect.parameterSetNode()
        segmentEditorNode.SetOverwriteMode(segmentEditorNode.OverwriteNone)

        self.isInitialized = True
        self.handleTaskChange(self.contextComboBox.currentIndex)

    def changeViewLayout(self, layout=None):
        # Change the layout of the view
        # If layout is None, get the currently selected view color and set the layout to this view only
        layoutManager = slicer.app.layoutManager()
        if layout is None:
            sliceColor = self.viewComboBox.currentText
            layout = getattr(vtkMRMLLayoutNode, f"SlicerLayoutOneUp{sliceColor}SliceView")
        layoutManager.setLayout(layout)

    def onDone(self):
        # Called when clicked on done

        # Reset the segmentation logic
        self.segmentationLogic.reset()

        # Update the ui elements now enabled
        self.viewComboBox.setEnabled(True)
        self.initializeBtn.setEnabled(True)
        self.predictBtn.setEnabled(False)
        self.doneBtn.setEnabled(False)
        self.predict3dBtn.setEnabled(False)
        self.sliceRangeSelector.setEnabled(False)
        self.isInitialized = False

    def handleTaskChange(self, itemSelected: int):
        # Called when the selected task is changed
        # Handle which task management button is enabled
        # Show/hide appropriate warning message

        if itemSelected < 1:
            # Case where no task is selected
            self.contextLogic.activeContext = None
            self.warningEmptyTask.setVisible(True)
            self.warningEmptyContext.setVisible(False)
            self.removeTaskButton.setEnabled(False)
            self.removeContextExampleButton.setEnabled(False)
            self.addContextExampleButton.setEnabled(False)
            self.renameTaskButton.setEnabled(False)
            self.exportTaskButton.setEnabled(False)
            self.predictBtn.setEnabled(False)
            self.predict3dBtn.setEnabled(False)
            self.editTaskButton.setChecked(True)
        else:
            # Case where a task is selected
            self.contextLogic.activeContext = self.contextComboBox.currentText
            self.warningEmptyTask.setVisible(False)
            self.warningEmptyContext.setVisible(self.contextLogic.getCurrentContextSize() == 0)
            self.removeTaskButton.setEnabled(True)
            self.removeContextExampleButton.setEnabled(self.contextLogic.getCurrentContextSize() != 0)
            self.addContextExampleButton.setEnabled(True)
            self.renameTaskButton.setEnabled(True)
            self.exportTaskButton.setEnabled(True)
            self.predictBtn.setEnabled(self.contextLogic.getCurrentContextSize() != 0 and self.isInitialized)
            self.predict3dBtn.setEnabled(self.contextLogic.getCurrentContextSize() != 0 and self.isInitialized)
            self.editTaskButton.setChecked(
                self.contextLogic.getCurrentContextSize() == 0 or self.addTaskButton.isVisible())

    def handleTaskEditionMode(self, toggled: bool):
        # Handle the visibility of task management buttons when task edition is toggled
        self.addTaskButton.setVisible(toggled)
        self.removeTaskButton.setVisible(toggled)
        self.renameTaskButton.setVisible(toggled)
        self.addContextExampleButton.setVisible(toggled)
        self.removeContextExampleButton.setVisible(toggled)
        self.importTaskButton.setVisible(toggled)
        self.exportTaskButton.setVisible(toggled)

    def setSliceOffset(self, offset):
        sliceColor = self.viewComboBox.currentText
        sliceNode: vtkMRMLSliceNode = slicer.mrmlScene.GetNodeByID(f"vtkMRMLSliceNode{sliceColor}")
        sliceLogic: vtkMRMLSliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
        sliceLogic.SetSliceOffset(offset)

    def addImageToContext(self):
        initialLayout = slicer.app.layoutManager().layout
        self.changeViewLayout()
        self.addImageDialog()
        self.changeViewLayout(initialLayout)  # Restore the layout
        self.handleTaskChange(self.contextComboBox.currentIndex)

    def removeImageFromContext(self):
        imageToRemove = self.removeImageDialog()
        if imageToRemove == -1:
            return

        self.contextLogic.removeExample(imageToRemove)
        self.handleTaskChange(self.contextComboBox.currentIndex)

    def addImageDialog(self):
        # Create and handle the dialog to add an image to a context

        def onSegmentSelectionChange(segmentName):
            displayNode: vtkMRMLSegmentationDisplayNode = currentSegmentationNode.GetDisplayNode()
            segmentID = currentSegmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
            displayNode.SetAllSegmentsVisibility(False)
            displayNode.SetSegmentVisibility(segmentID, True)

        dialog = qt.QDialog()
        dialog.setWindowTitle("Image and mask selection")
        layout = qt.QVBoxLayout()

        # Fetch the different node/object needed
        currentVolume = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        currentView = self.viewComboBox.currentText
        currentSegmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        availableSegmentsID = currentSegmentationNode.GetSegmentation().GetSegmentIDs()
        availableSegmentsNames = list(
            map(lambda x: currentSegmentationNode.GetSegmentation().GetSegment(x).GetName(), availableSegmentsID))

        # Create the combo box for segment selection
        segmentComboBox = qt.QComboBox()
        segmentComboBox.addItems(availableSegmentsNames)

        # Labels to indicate the currently selected nodes/objects
        l = qt.QLabel(f"Current volume: {currentVolume.GetName()}")
        l.setToolTip("Source volume selected in your segment editor")
        l.setCursor(qt.Qt.WhatsThisCursor)
        layout.addWidget(l)

        l = qt.QLabel(f"Current view: {currentView}")
        l.setToolTip("View selected in the MultiverSeg effect")
        l.setCursor(qt.Qt.WhatsThisCursor)
        layout.addWidget(l)

        l = qt.QLabel(f"Current segmentation: {currentSegmentationNode.GetName()}")
        l.setToolTip("Segmentation selected in your segment editor")
        l.setCursor(qt.Qt.WhatsThisCursor)
        layout.addWidget(l)

        layout.addWidget(qt.QLabel("\nSelect the mask to export:"))
        layout.addWidget(segmentComboBox)

        # Dialog buttons
        buttonBox = qt.QDialogButtonBox()
        buttonBox.addButton(buttonBox.Cancel)
        buttonBox.addButton(buttonBox.Save)
        buttonBox.button(buttonBox.Save).setEnabled(segmentComboBox.currentIndex != -1)
        buttonBox.rejected.connect(dialog.reject)
        buttonBox.accepted.connect(dialog.accept)
        layout.addWidget(buttonBox)

        segmentComboBox.connect("currentTextChanged(QString)", onSegmentSelectionChange)
        onSegmentSelectionChange(segmentComboBox.currentText)
        dialog.setLayout(layout)

        if dialog.exec():
            # Add the image on dialog validation
            segmentId = currentSegmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentComboBox.currentText)
            self.contextLogic.saveNewExample(currentVolume, currentView, segmentId, currentSegmentationNode,
                                             self.segmentationLogic)

        currentSegmentationNode.GetDisplayNode().SetAllSegmentsVisibility(True)

    def removeImageDialog(self):
        # Create and handle the dialog to remove an image from a context

        dialog = qt.QDialog()
        dialog.setWindowTitle("Select the example to remove")
        layout = qt.QVBoxLayout()

        l = qt.QLabel(f"Select which example to remove:")
        layout.addWidget(l)

        contextPath = self.contextLogic.contextRootPath.joinpath(self.contextLogic.activeContext)
        contextContent = contextPath.glob(("image*"))
        contextNumbers = sorted(map(lambda x: int(x.stem[6:]), contextContent))
        imageNames = list(map(lambda x: f"Image #{x}", contextNumbers))

        imageComboBox = qt.QComboBox()
        imageComboBox.addItems(imageNames)
        layout.addWidget(imageComboBox)

        imagePreview = qt.QLabel()

        def blendImages(base: qt.QImage, mask: qt.QImage):
            result = base.copy()

            width = base.width()
            height = base.height()

            for y in range(height):
                for x in range(width):

                    overlayColor = mask.pixelColor(x, y)
                    if overlayColor == qt.QColor("white"):
                        redBaseColor = base.pixelColor(x, y).red()
                        result.setPixelColor(x, y, qt.QColor(redBaseColor, 0, 0))

            return result

        image = qt.QImage(contextPath.joinpath(f"image_{contextNumbers[0]}.png").resolve())
        mask = qt.QImage(contextPath.joinpath(f"mask_{contextNumbers[0]}.png").resolve())

        imagePreview.setPixmap(qt.QPixmap().fromImage(blendImages(image, mask)))
        imagePreview.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(imagePreview)

        def changePreviewImage(name):
            imageNumber = int(name.split("#")[1])
            i = qt.QImage(contextPath.joinpath(f"image_{imageNumber}.png").resolve())
            m = qt.QImage(contextPath.joinpath(f"mask_{imageNumber}.png").resolve())
            imagePreview.setPixmap(qt.QPixmap().fromImage(blendImages(i, m)))

        imageComboBox.connect("currentTextChanged(QString)", changePreviewImage)

        # Dialog buttons
        buttonBox = qt.QDialogButtonBox()
        buttonBox.addButton(buttonBox.Cancel)
        deleteButton = buttonBox.addButton("Delete", buttonBox.DestructiveRole)
        buttonBox.rejected.connect(dialog.reject)
        deleteButton.connect("clicked()", dialog.accept)
        layout.addWidget(buttonBox)

        dialog.setLayout(layout)

        while dialog.exec() == 1:
            if slicer.util.confirmYesNoDisplay(
                    "Are you sure you want to delete this image ?\nThis action is irreversible.",
                    "Delete this image ?"):
                return int(imageComboBox.currentText.split('#')[1])  # Number of image to delete
        return -1

    def importContext(self):

        fileName = qt.QFileDialog().getOpenFileName(None, "Import context", ".", "*.zip")
        try:
            contextName = self.contextLogic.importContext(fileName)
            if contextName != '':
                self.contextComboBox.addItem(contextName)
                self.contextComboBox.setCurrentText(contextName)
        except FileNotFoundError:
            slicer.util.errorDisplay(f"The file {fileName} could not be found.", "Import failed")
        except IsADirectoryError:
            taskName = pathlib.Path(fileName).stem
            slicer.util.errorDisplay(f"A task named {taskName} already exist.", "Import failed")

    def addTask(self):
        segmentation = self.scriptedEffect.parameterSetNode().GetSegmentationNode().GetSegmentation()
        segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
        segName = segmentation.GetSegment(segmentID).GetName()
        name = qt.QInputDialog().getText(None, "New task", "Name of the task:",0,segName)

        if name != "":

            if self.contextLogic.createTask(name):
                self.contextComboBox.addItem(name)
                self.contextComboBox.setCurrentText(name)
            else:
                slicer.util.errorDisplay(f"Task creation failed because a task with the name name already exist.",
                                         "Creation failed")

    def removeTask(self):

        if slicer.util.confirmYesNoDisplay("Are you sure you want to delete this task ?\nThis action is irreversible.",
                                           "Delete this task ?"):
            currentId = self.contextComboBox.currentIndex
            self.contextLogic.deleteCurrentTask()
            self.contextComboBox.removeItem(currentId)
            self.contextComboBox.setCurrentIndex(0)

    def renameTask(self):
        newName = qt.QInputDialog().getText(None, "Rename task",
                                            f"New name for task {self.contextComboBox.currentText}:",
                                            qt.QLineEdit().Normal, self.contextComboBox.currentText)
        if newName != '':

            currentId = self.contextComboBox.currentIndex
            if self.contextLogic.renameTask(newName):
                self.contextComboBox.removeItem(currentId)
                self.contextComboBox.addItem(newName)
                self.contextComboBox.setCurrentText(newName)
            else:
                slicer.util.errorDisplay(f"A task named {newName} already exist.", "Error")

    def exportContext(self):
        dir = qt.QFileDialog().getExistingDirectory(None, "Export to:", ".",
                                                    qt.QFileDialog().ShowDirsOnly + qt.QFileDialog().ReadOnly)

        if dir:
            try:
                res = self.contextLogic.exportContext(dir)
                if res:
                    slicer.util.infoDisplay("Task exported successfully!", "Task exported")
                else:
                    slicer.util.errorDisplay("Task exported failed.", "Export failed")
            except FileExistsError:
                slicer.util.errorDisplay(f"The file {self.contextLogic.activeContext}.zip already exist.",
                                         "Export failed")

    def selectContextRoot(self):
        self.contextRootSelectionDialog()

        # Remove every task
        self.contextComboBox.setCurrentIndex(0)
        while self.contextComboBox.count > 1:
            self.contextComboBox.removeItem(1)

        self.contextComboBox.addItems(self.contextLogic.getContextList())

    def contextRootSelectionDialog(self):

        dialog = qt.QDialog()
        dialog.setWindowTitle("Select tasks root")
        layout = qt.QVBoxLayout(dialog)

        text = qt.QLabel("Select the root directory for your tasks:")
        layout.addWidget(text)

        directoryButton = ctk.ctkDirectoryButton()
        directoryButton.directory = self.contextLogic.contextRootPath
        layout.addWidget(directoryButton)

        buttons = qt.QDialogButtonBox()
        buttons.addButton(buttons.Ok)
        buttons.addButton(buttons.Cancel)
        resetButton = buttons.addButton(buttons.Reset)
        layout.addWidget(buttons)

        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        resetButton.clicked.connect(lambda: updateDirectoryButton(self.contextLogic.computeBaseContextRoot()))

        def updateDirectoryButton(path):
            directoryButton.directory = path

        if dialog.exec():
            self.contextLogic.setContextRoot(pathlib.Path(directoryButton.directory))
