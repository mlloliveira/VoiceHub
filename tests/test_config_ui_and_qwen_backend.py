from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


class ConfigUISaveTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["VOICEHUB_PREFS_DIR"] = self.tmpdir.name
        for name in [
            "src.voicehub.prefs",
            "src.voicehub.user_settings",
            "src.voicehub.config_ui",
        ]:
            sys.modules.pop(name, None)
        import src.voicehub.prefs as prefs
        import src.voicehub.user_settings as user_settings
        import src.voicehub.config_ui as config_ui
        self.prefs = importlib.reload(prefs)
        self.user_settings = importlib.reload(user_settings)
        self.config_ui = importlib.reload(config_ui)

    def tearDown(self):
        os.environ.pop("VOICEHUB_PREFS_DIR", None)
        self.tmpdir.cleanup()
        for name in [
            "src.voicehub.prefs",
            "src.voicehub.user_settings",
            "src.voicehub.config_ui",
        ]:
            sys.modules.pop(name, None)

    def test_save_cfg_updates_only_active_family_chunk_value(self):
        s = self.user_settings.get_settings()
        self.assertEqual(s.xtts_max_chars_per_chunk, 200)
        self.assertEqual(s.qwen_max_chars_per_chunk, 512)

        self.config_ui._save_cfg(
            0.0, 0.9, 5, True, 5.0,
            "Qwen", 1400, "1.7B", 50.0, 5.0, True,
            0.7, 0.9, 4096, "",
        )
        s = self.user_settings.get_settings()
        self.assertEqual(s.qwen_max_chars_per_chunk, 1400)
        self.assertEqual(s.xtts_max_chars_per_chunk, 200)

        self.config_ui._save_cfg(
            0.0, 0.9, 5, True, 5.0,
            "XTTS", 180, "0.6B", 300.0, 5.0, True,
            0.7, 0.9, 4096, "",
        )
        s = self.user_settings.get_settings()
        self.assertEqual(s.qwen_max_chars_per_chunk, 1400)
        self.assertEqual(s.xtts_max_chars_per_chunk, 180)


    def test_apply_cfg_runtime_updates_in_memory_without_save(self):
        prefs_before = self.prefs.get_pref("settings", {})
        self.assertEqual(prefs_before.get("tts_default_family", "XTTS"), "XTTS")

        self.config_ui._apply_cfg_runtime(
            0.0, 0.9, 5, True, 5.0,
            "Qwen", 777, "0.6B", 45.0, 5.0, True,
            0.7, 0.9, 4096, "",
        )
        s = self.user_settings.get_settings()
        self.assertEqual(s.tts_default_family, "Qwen")
        self.assertEqual(s.qwen_model_size, "0.6B")
        self.assertEqual(s.qwen_max_chars_per_chunk, 777)
        self.assertEqual(s.qwen_clone_ref_max_seconds, 45.0)

        prefs_after = self.prefs.get_pref("settings", {})
        self.assertEqual(prefs_after.get("tts_default_family", "XTTS"), "XTTS")

    def test_slider_update_uses_saved_family_specific_value(self):
        self.user_settings.update_settings(qwen_max_chars_per_chunk=1330, xtts_max_chars_per_chunk=175)
        qwen_update = self.config_ui._slider_update_for_family("Qwen")
        xtts_update = self.config_ui._slider_update_for_family("XTTS")
        self.assertEqual(qwen_update["value"], 1330)
        self.assertEqual(qwen_update["maximum"], 2048)
        self.assertEqual(xtts_update["value"], 175)
        self.assertEqual(xtts_update["maximum"], 400)

    def test_qwen_prompt_visibility_helper(self):
        self.assertTrue(self.config_ui._qwen_prompt_visibility("Qwen")["visible"])
        self.assertFalse(self.config_ui._qwen_prompt_visibility("XTTS")["visible"])

    def test_save_cfg_can_persist_qwen_model_size(self):
        self.config_ui._save_cfg(
            0.0, 0.9, 5, True, 5.0,
            "Qwen", 900, "0.6B", 50.0, 5.0, True,
            0.7, 0.9, 4096, "",
        )
        s = self.user_settings.get_settings()
        self.assertEqual(s.qwen_model_size, "0.6B")

    def test_unpack_chunk_accepts_dict_with_numpy_audio(self):
        from src.voicehub.asr import _unpack_chunk

        pcm = (np.arange(16, dtype=np.int16) - 8).reshape(-1)
        sr, y = _unpack_chunk({"sample_rate": 16000, "data": pcm})
        self.assertEqual(sr, 16000)
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(y.ndim, 1)


class QwenBackendWrapperTests(unittest.TestCase):
    def test_synthesize_qwen_clone_retries_without_instruct_on_typeerror(self):
        from src.voicehub import qwen_backend

        class DummyModel:
            def __init__(self):
                self.calls = []

            def generate_voice_clone(self, **kwargs):
                self.calls.append(kwargs)
                if "instruct" in kwargs:
                    raise TypeError("unexpected keyword instruct")
                return [np.zeros(8, dtype=np.float32)], 24000

        dummy = DummyModel()
        with patch.object(qwen_backend, "get_qwen_clone_model", return_value=dummy):
            wavs, sr = qwen_backend.synthesize_qwen_clone(
                ["hello"],
                language="English",
                voice_clone_prompt={"foo": "bar"},
                instruct="Please speak faster.",
                max_new_tokens=256,
            )

        self.assertEqual(sr, 24000)
        self.assertEqual(len(wavs), 1)
        self.assertEqual(len(dummy.calls), 2)
        self.assertIn("instruct", dummy.calls[0])
        self.assertNotIn("instruct", dummy.calls[1])

    def test_synthesize_qwen_custom_retries_without_stopping_criteria(self):
        from src.voicehub import qwen_backend
        from threading import Event

        class DummyModel:
            def __init__(self):
                self.calls = []

            def generate_custom_voice(self, **kwargs):
                self.calls.append(kwargs)
                if "stopping_criteria" in kwargs:
                    raise TypeError("unexpected keyword stopping_criteria")
                return [np.zeros(8, dtype=np.float32)], 24000

        dummy = DummyModel()
        with patch.object(qwen_backend, "get_qwen_custom_model", return_value=dummy), \
             patch.object(qwen_backend, "_generation_stop_kwargs", return_value={"stopping_criteria": object()}):
            wavs, sr = qwen_backend.synthesize_qwen_custom(
                ["hello"],
                language="English",
                speaker="Ryan",
                instruct="Speak faster.",
                max_new_tokens=256,
                stop_event=Event(),
            )

        self.assertEqual(sr, 24000)
        self.assertEqual(len(wavs), 1)
        self.assertEqual(len(dummy.calls), 2)
        self.assertIn("stopping_criteria", dummy.calls[0])
        self.assertNotIn("stopping_criteria", dummy.calls[1])


class ASRAudioNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_normalize_audio_to_wav_accepts_path_tuple_and_dict(self):
        from src.voicehub.asr import _normalize_audio_to_wav

        wav = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
        src = Path(self.tmpdir.name) / "src.wav"
        sf.write(src, wav, 16000)

        out1 = _normalize_audio_to_wav(str(src))
        self.assertEqual(out1, str(src))

        out2 = _normalize_audio_to_wav((16000, wav))
        self.assertTrue(Path(out2).exists())

        out3 = _normalize_audio_to_wav({"sample_rate": 16000, "data": wav})
        self.assertTrue(Path(out3).exists())


class UIBuildSmokeTests(unittest.TestCase):
    def test_build_app_constructs_without_component_errors(self):
        import os, sys
        os.environ.setdefault("VOICEHUB_PREFS_DIR", "preferences")
        if "src" not in sys.path:
            sys.path.insert(0, "src")
        from src.voicehub.ui import build_app

        app = build_app()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()


class ASRDeviceSelectionTests(unittest.TestCase):
    def test_fw_model_uses_cpu_when_cuda_unavailable(self):
        from src.voicehub import asr

        captured = {}
        class DummyFW:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
        with patch.object(asr, '_fw_model', None),              patch.object(asr, 'FW_AVAILABLE', True),              patch.object(asr, 'WhisperModel', DummyFW),              patch.object(asr, '_torch_cuda_available', return_value=False):
            asr.get_fw_model()
        self.assertEqual(captured['device'], 'cpu')

    def test_ow_model_uses_cpu_when_cuda_unavailable(self):
        from src.voicehub import asr

        captured = {}
        class DummyOW:
            def load_model(self, *args, **kwargs):
                captured.update(kwargs)
                return object()
        with patch.object(asr, '_ow_model', None),              patch.object(asr, 'OWHISPER_AVAILABLE', True),              patch.object(asr, 'owhisp', DummyOW()),              patch.object(asr, '_torch_cuda_available', return_value=False):
            asr.get_ow_model()
        self.assertEqual(captured['device'], 'cpu')


class CloneRefConfigTests(unittest.TestCase):
    def test_save_cfg_persists_family_specific_clone_cap(self):
        from src.voicehub import config_ui
        from src.voicehub.user_settings import get_settings

        config_ui._save_cfg(
            0.0, 0.9, 5, True, 5.0,
            "Qwen", 512, "1.7B", 50, 5.0, True, 0.7, 0.9, 4096, ""
        )
        s = get_settings()
        self.assertEqual(s.qwen_clone_ref_max_seconds, 50.0)

        config_ui._save_cfg(
            0.0, 0.9, 5, True, 5.0,
            "XTTS", 200, "1.7B", 300, 5.0, True, 0.7, 0.9, 4096, ""
        )
        s = get_settings()
        self.assertEqual(s.xtts_clone_ref_max_seconds, 300.0)

    def test_clone_slider_update_uses_saved_family_value(self):
        from src.voicehub import config_ui
        from src.voicehub.user_settings import update_settings

        update_settings(xtts_clone_ref_max_seconds=300.0, qwen_clone_ref_max_seconds=50.0)
        qwen_update = config_ui._clone_slider_update_for_family("Qwen")
        xtts_update = config_ui._clone_slider_update_for_family("XTTS")
        self.assertEqual(qwen_update["value"], 50.0)
        self.assertEqual(xtts_update["value"], 300.0)


class XTTSAudioCompatTests(unittest.TestCase):
    def test_torchaudio_loader_compat_patches_when_torchcodec_missing(self):
        import types
        import src.voicehub.tts as tts

        dummy_torchaudio = types.SimpleNamespace(load=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("orig")))
        dummy_torch = types.SimpleNamespace(from_numpy=lambda arr: arr)

        with patch.dict("sys.modules", {"torchaudio": dummy_torchaudio, "torch": dummy_torch}, clear=False),              patch.dict("sys.modules", {"torchcodec": None}, clear=False):
            tts._TORCHAUDIO_LOAD_PATCHED = False
            tts._ensure_xtts_audio_loader_compat()

        self.assertTrue(tts._TORCHAUDIO_LOAD_PATCHED)
        self.assertNotEqual(getattr(dummy_torchaudio.load, "__name__", ""), "<lambda>")
