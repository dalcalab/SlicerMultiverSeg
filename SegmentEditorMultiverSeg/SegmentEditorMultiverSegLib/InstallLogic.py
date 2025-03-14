import shutil
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
            if (cls.downloadMultiverSegCheckpointIfNeeded() and
                    cls.downloadScribblePromptCheckpointIfNeeded()):
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

        fileDest = sampleDataLogic.downloadFileIntoCache(modelURI,
                                                         modelName)

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
