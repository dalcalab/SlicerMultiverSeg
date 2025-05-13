import sys
import slicer.util

if __name__ == '__main__':

    extension_names = sys.argv[1:]
    print("Found extensions: ", extension_names)

    em = slicer.app.extensionsManagerModel()
    em.setInteractive(False)

    for name in extension_names:
        em.installExtensionFromServer(name, False)

    slicer.util.quit()
