import sys
from pathlib import Path

import slicer.testing

if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise AttributeError(f"run_tests.py requires 2 arguments, found {sys.argv[1:]}")

    root = Path(sys.argv[1])
    files = list(root.glob(sys.argv[2]))

    print(f"Found {len(files)} test file(s).")

    for file in files:
        slicer.testing.runUnitTest(file.parent.as_posix(), file.stem)
