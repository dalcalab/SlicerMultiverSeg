import shutil
import sys
from pathlib import Path

import SampleData
import slicer.util


class InstallLogic:
    CKPT_DIR = Path(__file__).parent.joinpath("../Resources/Checkpoints").resolve()
    MULTIVERSEG_FILE_NAME = "MultiverSeg_v0_nf256_res128.pt"
    MULTIVERSEG_DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/5sup9s5l9xkp28wbwkm9g/MultiverSeg_v0_nf256_res128.pt?rlkey=tquwxhezdgl2v8a2akvco5jim&st=1gnbqplc&dl=1"

    SCRIBBLEPROMPT_FILE_NAME = "ScribblePrompt_unet_v1_nf192_res128.pt"
    SCRIBBLEPROMPT_DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/pnw88n05irnv5z1snlklr/ScribblePrompt_unet_v1_nf192_res128.pt?rlkey=dr8xvkf0wj2r082h1zzpcmz5o&dl=1"

    @classmethod
    def downloadCheckpointsIfNeeded(cls):
        try:
            if (cls.downloadMultiverSegCheckpointIfNeeded() and cls.downloadScribblePromptCheckpointIfNeeded()):
                return True
            else:
                return False
        except Exception as e:

            slicer.util.errorDisplay(e, "Error during model download")
            raise e

    @classmethod
    def downloadMultiverSegCheckpointIfNeeded(cls):

        modelPath = cls.CKPT_DIR.joinpath("MultiverSeg_v0_nf256_res128.pt")

        if not modelPath.is_file():
            if slicer.util.confirmOkCancelDisplay("The MultiverSeg model is required. Confirm to download (74MB)."):

                cls._downloadModel(cls.MULTIVERSEG_FILE_NAME, cls.MULTIVERSEG_DOWNLOAD_URL)
                return True  # Model downloaded correctly
            else:
                return False  # User did not accept to download the model
        return True  # No need to download the model

    @classmethod
    def downloadScribblePromptCheckpointIfNeeded(cls):
        modelPath = cls.CKPT_DIR.joinpath(cls.SCRIBBLEPROMPT_FILE_NAME)

        if not modelPath.is_file():
            if slicer.util.confirmOkCancelDisplay("The ScribblePrompt model is required. Confirm to download (16MB)."):

                cls._downloadModel(cls.SCRIBBLEPROMPT_FILE_NAME, cls.SCRIBBLEPROMPT_DOWNLOAD_URL)

                return True  # Model downloaded correctly
            else:
                return False  # User did not accept to download the model
        return True  # No need to download the model

    @classmethod
    def _downloadModel(cls, modelName, modelURI):
        modelPath = cls.CKPT_DIR.joinpath(modelName)
        slicer.progressWindow = slicer.util.createProgressDialog()
        slicer.progressWindow.setLabelText(f"Downloading {modelName.split('_')[0]} checkpoint...")
        sampleDataLogic = SampleData.SampleDataLogic()
        sampleDataLogic.logMessage = lambda msg, lvl=None: cls.reportProgress(sampleDataLogic, msg, lvl)

        fileDest = sampleDataLogic.downloadFileIntoCache(modelURI, modelName)

        if sampleDataLogic.downloadPercent and sampleDataLogic.downloadPercent == 100:
            shutil.copyfile(fileDest, modelPath)
            slicer.progressWindow.close()

    @staticmethod
    def reportProgress(logic, msg, level=None):

        if level is not None:
            print(f"{level}: {msg}")

        # Abort download if cancel is clicked in progress bar
        if slicer.progressWindow.wasCanceled:
            raise Exception("Download aborted")

        # Update progress window
        slicer.progressWindow.show()
        slicer.progressWindow.activateWindow()
        slicer.progressWindow.setValue(int(logic.downloadPercent))

        # Process events to allow screen to refresh
        slicer.app.processEvents()


class DependenciesLogic:

    # Install dependencies to torch and multiverseg if not already installed
    @classmethod
    def installDependenciesIfNeeded(cls):
        # Return True when all is setup correctly
        if not cls.installTorchIfNeeded():
            print("Torch was not installed correctly", file=sys.stderr)
            return False

        if not cls.installMultiverSegIfNeeded():
            print("MultiverSeg package was not installed correctly", file=sys.stderr)
            return False

        return True

    # Install the PyTorch Utils extension if not already installed
    @classmethod
    def installPyTorchExtensionIfNeeded(cls):
        # Return True when all is setup correctly
        # Return False when PyTorch extension is not installed or slicer need to restart
        # Throw an error if something went wrong with the installation
        try:
            import PyTorchUtils  # noqa
            return True
        except ModuleNotFoundError:
            ret = slicer.util.confirmOkCancelDisplay("""This module requires PyTorch extension. Would you like to install it?
                
Slicer will need to be restarted before continuing the install.""", "PyTorch extension not found.")
            if ret:
                cls.installPyTorchExtension()
            return False  # Need restart or not installed

    # Install the PyTorch Utils extension
    @staticmethod
    def installPyTorchExtension():
        extensionManager = slicer.app.extensionsManagerModel()
        extName = "PyTorch"
        if extensionManager.isExtensionInstalled(extName):
            return

        if not extensionManager.installExtensionFromServer(extName):
            raise RuntimeError("Failed to install PyTorch extension from the servers. "
                               "Manually install to continue.")

    # Check if multiverseg is installed, and prompt the user to install it if not
    @classmethod
    def installMultiverSegIfNeeded(cls):
        try:
            import multiverseg
            return True
        except ModuleNotFoundError:
            ret = slicer.util.confirmOkCancelDisplay(
                "This module requires the MultiverSeg python package. Would you like to install it?",
                "MultiverSeg package not found.")
            if ret:
                cls.installMultiverSeg()
                return True

            return False

    # Install multiverseg
    @classmethod
    def installMultiverSeg(cls):
        slicer.util.pip_install("git+https://github.com/halleewong/MultiverSeg.git")

    # Check if torch is installed and install it if not
    @classmethod
    def installTorchIfNeeded(cls):
        try:
            import torch
            import torchvision
            return True
        except ModuleNotFoundError:
            return cls.installTorch()

    # Install torch through PyTorch Utils extension
    @classmethod
    def installTorch(cls):
        import PyTorchUtils
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        res = torchLogic.installTorch(askConfirmation=True)

        if res is None:
            raise RuntimeError("Failed to install torch and torchvision. "
                               "Manually install through PyTorch Utils extension to continue.")
        return res
