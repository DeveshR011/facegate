"""FaceGate - AI face unlock for Windows apps and the lock screen."""

import os as _os

# numpy's OpenBLAS otherwise starts a thread per core and commits ~750 MB for their buffers at
# import. FaceGate's matrix work is a few hundred dot products per frame; one thread is plenty.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.3.0"
APP_NAME = "FaceGate"
