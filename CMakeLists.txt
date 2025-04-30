cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerMultiverSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/dalcalab/SlicerMultiverSeg/tree/main#readme")
set(EXTENSION_CONTRIBUTORS "Sebastien Goll (Kitware SAS)")
set(EXTENSION_DESCRIPTION "MultiverSeg is a model that enables users to rapidly segment new biomedical imaging datasets. This extension add a segmentation effect using that uses MultiverSeg.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/dalcalab/SlicerMultiverSeg/refs/heads/main/MultiverSeg/Resources/Icons/SegmentEditorMultiverSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/dalcalab/SlicerMultiverSeg/refs/heads/main/Screenshots/Demo_1.png https://raw.githubusercontent.com/dalcalab/SlicerMultiverSeg/refs/heads/main/Screenshots/Demo_2.png https://raw.githubusercontent.com/dalcalab/SlicerMultiverSeg/refs/heads/main/Screenshots/Demo_3.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MultiverSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
