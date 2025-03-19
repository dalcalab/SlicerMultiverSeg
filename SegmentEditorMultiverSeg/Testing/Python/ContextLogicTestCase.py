import unittest
from pathlib import Path

import slicer.util
from SegmentEditorMultiverSegLib import ContextLogic, SegmentationLogic


class ContextLogicTestCase(unittest.TestCase):

    def test_instantiation(self):
        logic = ContextLogic(None)

    def test_getCurrentContextSize(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()
        logic.activeContext = "empty_context"

        self.assertEqual(logic.getCurrentContextSize(), 0)

        logic.activeContext = "context_1"
        self.assertEqual(logic.getCurrentContextSize(), 3)

    def test_loadContext(self):
        logic = ContextLogic(None)

        res = logic.loadContext()
        self.assertEqual(res, (None, None))

        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()
        logic.activeContext = "empty_context"
        res = logic.loadContext()
        self.assertEqual(res, (None, None))

        logic.activeContext = "context_1"
        res = logic.loadContext()
        self.assertEqual(len(res), 2)
        images, masks = res
        self.assertSequenceEqual(images.shape, [3, 1, 128, 128])
        self.assertSequenceEqual(masks.shape, [3, 1, 128, 128])

    def test_getContextList(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        contextList = logic.getContextList()
        self.assertCountEqual(contextList, ['empty_context', "context_1"])

    def test_getNextExampleNumber(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        logic.activeContext = "empty_context"
        nextN = logic.getNextExampleNumber()
        self.assertEqual(nextN, 0)

        logic.activeContext = "context_1"
        nextN = logic.getNextExampleNumber()
        self.assertEqual(nextN, 4)

    def test_saveNewExample(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()
        logic.activeContext = "empty_context"

        ressourcePath = Path(__file__).parent.joinpath("../TestData/SlicerData").resolve()
        vol = slicer.util.loadVolume(ressourcePath.joinpath("1.nrrd"))
        seg = slicer.util.loadSegmentation(ressourcePath.joinpath("1.seg.nrrd"))
        segID = seg.GetSegmentation().GetNthSegmentID(0)

        segLogic = SegmentationLogic(None)

        logic.saveNewExample(vol, "Red", segID, seg, segLogic)

        contextPath = logic.contextRootPath.joinpath("empty_context")
        self.assertEqual(len(list(contextPath.glob("*"))), 2)

        import torchvision, torch
        i = torchvision.io.decode_image(contextPath.joinpath("image_0.png"), mode=torchvision.io.ImageReadMode.GRAY)
        m = torchvision.io.decode_image(contextPath.joinpath("mask_0.png"), mode=torchvision.io.ImageReadMode.GRAY)
        self.assertSequenceEqual(i.shape, [1, 128, 128])
        self.assertSequenceEqual(m.shape, [1, 128, 128])

        iTruth = torchvision.io.decode_image(ressourcePath.joinpath("image_0.png"),
                                             mode=torchvision.io.ImageReadMode.GRAY)
        mTruth = torchvision.io.decode_image(ressourcePath.joinpath("mask_0.png"),
                                             mode=torchvision.io.ImageReadMode.GRAY)

        diff = torch.abs(i - iTruth)
        self.assertEqual(torch.max(diff).item(), 0)

        diff = torch.abs(m - mTruth)
        self.assertEqual(torch.max(diff).item(), 0)

        import shutil
        shutil.rmtree(contextPath)
        contextPath.mkdir()

    def test_exportContext(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        self.assertRaises(AssertionError, logic.exportContext, 'aaa')

        logic.activeContext = 'context_1'

        filePath = Path(__file__).parent.joinpath("../TestData/context_1.zip").resolve()
        self.assertFalse(filePath.is_file())
        logic.exportContext(filePath.parent.as_posix())
        self.assertTrue(filePath.is_file())

        import os
        os.remove(filePath)
        self.assertFalse(filePath.is_file())

        res = logic.exportContext(None)
        self.assertFalse(filePath.is_file())
        self.assertFalse(res)

        res = logic.exportContext("")
        self.assertFalse(filePath.is_file())
        self.assertFalse(res)

    def test_importContext(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        res = logic.importContext(None)
        self.assertEqual(res, '')
        res = logic.importContext("")
        self.assertEqual(res, '')

        d = Path(__file__).parent.joinpath("../TestData").resolve()
        self.assertRaises(FileNotFoundError, logic.importContext, d.joinpath("false_context.zip").resolve())

        res = logic.importContext(d.joinpath("context_to_import.zip").resolve())
        self.assertEqual(res, "context_to_import")
        self.assertTrue(d.joinpath("Context/context_to_import").is_dir())
        self.assertEqual(len(list(d.joinpath("Context/context_to_import").glob("*"))), 6)

        self.assertRaises(IsADirectoryError, logic.importContext, d.joinpath("context_to_import.zip").resolve())

        import shutil
        shutil.rmtree(d.joinpath("Context/context_to_import"))

    def test_deleteCurrentTask(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        newPath = Path(__file__).parent.joinpath("../TestData/Context/empty_context_2").resolve()
        newPath.mkdir()
        self.assertTrue(newPath.is_dir())

        logic.activeContext = "empty_context_2"
        logic.deleteCurrentTask()

        self.assertFalse(newPath.is_dir())

    def test_createTask(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()

        res = logic.createTask("empty_context")
        self.assertFalse(res)

        newPath = Path(__file__).parent.joinpath("../TestData/Context/empty_context_2").resolve()
        res = logic.createTask("empty_context_2")
        self.assertTrue(res)
        self.assertTrue(newPath.is_dir())

        import shutil
        shutil.rmtree(newPath)

    def test_renameTask(self):
        logic = ContextLogic(None)
        logic.contextRootPath = Path(__file__).parent.joinpath("../TestData/Context").resolve()
        logic.activeContext = "empty_context"

        res = logic.renameTask("context_1")
        self.assertFalse(res)

        res = logic.renameTask("empty_context_2")
        oldPath = Path(__file__).parent.joinpath("../TestData/Context/empty_context").resolve()
        newPath = Path(__file__).parent.joinpath("../TestData/Context/empty_context_2").resolve()
        self.assertTrue(res)
        self.assertFalse(oldPath.is_dir())
        self.assertTrue(newPath.is_dir())

        logic.activeContext = "empty_context_2"

        res = logic.renameTask("empty_context")
        self.assertTrue(res)
        self.assertFalse(newPath.is_dir())
        self.assertTrue(oldPath.is_dir())
