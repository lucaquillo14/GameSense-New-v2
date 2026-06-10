import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.upload_limits import MAX_UPLOAD_BYTES, MAX_VIDEO_DURATION_S


class UploadLimitsTests(unittest.TestCase):
    def test_max_upload_is_250mb(self):
        self.assertEqual(MAX_UPLOAD_BYTES, 250 * 1024 * 1024)

    def test_max_duration_is_60_seconds(self):
        self.assertEqual(MAX_VIDEO_DURATION_S, 60.0)


if __name__ == "__main__":
    unittest.main()
