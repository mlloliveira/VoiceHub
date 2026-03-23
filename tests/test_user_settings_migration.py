from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


class UserSettingsMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["VOICEHUB_PREFS_DIR"] = self.tmpdir.name

    def tearDown(self):
        os.environ.pop("VOICEHUB_PREFS_DIR", None)
        self.tmpdir.cleanup()
        for name in ["src.voicehub.user_settings", "src.voicehub.prefs"]:
            sys.modules.pop(name, None)

    def _prefs_file(self) -> Path:
        return Path(self.tmpdir.name) / "config.json"

    def test_new_user_defaults_to_xtts(self):
        import src.voicehub.prefs as prefs
        import src.voicehub.user_settings as user_settings
        importlib.reload(prefs)
        importlib.reload(user_settings)
        self.assertEqual(user_settings.get_settings().tts_default_family, "XTTS")

    def test_existing_user_without_family_is_migrated_to_xtts(self):
        self._prefs_file().write_text(json.dumps({"settings": {"whisper_beam_size": 7}}), encoding="utf-8")
        import src.voicehub.prefs as prefs
        import src.voicehub.user_settings as user_settings
        importlib.reload(prefs)
        importlib.reload(user_settings)
        settings = user_settings.get_settings()
        self.assertEqual(settings.tts_default_family, "XTTS")
        saved = json.loads(self._prefs_file().read_text(encoding="utf-8"))
        self.assertEqual(saved["settings"]["tts_default_family"], "XTTS")


if __name__ == "__main__":
    unittest.main()
