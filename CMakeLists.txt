cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MultiverSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/dalcalab/MultiverSegSlicer/tree/main#readme") # TODO Update when repo is renamed
set(EXTENSION_CONTRIBUTORS "Sebastien Goll (Kitware SAS)")
set(EXTENSION_DESCRIPTION "MultiverSeg is a model that enables users to rapidly segment new biomedical imaging datasets. This extension add a segmentation effect using that uses MultiverSeg.")
set(EXTENSION_ICONURL "") # TODO Update when public
set(EXTENSION_SCREENSHOTURLS "") # TODO Update when public
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentEditorMultiverSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
