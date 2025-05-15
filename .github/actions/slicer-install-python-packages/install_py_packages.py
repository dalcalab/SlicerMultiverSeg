import sys

import slicer.util

if __name__ == "__main__":

    print("Packages found:",sys.argv[1:])

    for package in sys.argv[1:]:
        slicer.util.pip_install(package)
    slicer.util.quit()