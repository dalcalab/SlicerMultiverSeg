#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentEditorMultiverSeg)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/SegmentEditorEffect.py
  ${MODULE_NAME}Lib/ContextLogic.py
  ${MODULE_NAME}Lib/InstallLogic.py
  ${MODULE_NAME}Lib/SegmentationLogic.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/add.png
  Resources/Icons/addImage.png
  Resources/Icons/edit.png
  Resources/Icons/export.png
  Resources/Icons/folder_tree.png
  Resources/Icons/import.png
  Resources/Icons/remove.png
  Resources/Icons/removeImage.png
  Resources/Icons/rename.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Additional build-time testing
  add_subdirectory(Testing)
endif()
