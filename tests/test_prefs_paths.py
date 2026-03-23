from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


class PrefsPathTests(unittest.TestCase):
    def setUp(self):
        for name in ["src.voicehub.prefs"]:
            sys.modules.pop(name, None)

    def tearDown(self):
        os.environ.pop("VOICEHUB_PREFS_DIR", None)
        os.environ.pop("VOICEHUB_HOME", None)
        for name in ["src.voicehub.prefs"]:
            sys.modules.pop(name, None)

    def test_default_prefs_dir_is_user_profile_not_cwd(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            os.environ.pop("VOICEHUB_PREFS_DIR", None)
            os.environ["VOICEHUB_HOME"] = str(Path(home_dir) / ".voicehub")
            old_cwd = os.getcwd()
            try:
                os.chdir(cwd_dir)
                import src.voicehub.prefs as prefs
                prefs = importlib.reload(prefs)
                path = prefs.prefs_path()
            finally:
                os.chdir(old_cwd)
        self.assertEqual(path, Path(home_dir) / ".voicehub" / "preferences" / "config.json")
        self.assertNotEqual(path, Path(cwd_dir) / "preferences" / "config.json")


if __name__ == "__main__":
    unittest.main()
