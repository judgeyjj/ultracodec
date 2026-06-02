"""Smoke tests for the UltraCodec project skeleton.

These tests do **not** verify any model behaviour — they merely check that
imports succeed, configurations parse and the data utilities are usable
in a CPU-only environment without network access.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestImports(unittest.TestCase):
    def test_package_imports(self) -> None:
        import ultracodec  # noqa: F401
        from ultracodec import data, losses, metrics, model, utils  # noqa: F401

        self.assertEqual(ultracodec.__version__, "0.1.0")

    def test_data_module_exposes_classes(self) -> None:
        from ultracodec.data import (
            AudioTransform,
            LibriSpeechDataset,
            MixedDataset,
            VCTKDataset,
        )

        self.assertTrue(callable(AudioTransform))
        self.assertTrue(callable(VCTKDataset))
        self.assertTrue(callable(LibriSpeechDataset))
        self.assertTrue(callable(MixedDataset))

    def test_utils_module_exposes_helpers(self) -> None:
        from ultracodec.utils import (
            AverageMeter,
            EMA,
            get_device,
            init_logger,
            set_seed,
        )

        set_seed(123)
        self.assertIsNotNone(init_logger("test"))
        self.assertIn(get_device().type, {"cpu", "cuda", "mps"})
        meter = AverageMeter("loss")
        meter.update(0.5, 2)
        meter.update(1.0, 1)
        self.assertAlmostEqual(meter.avg, 2.0 / 3.0)


class TestConfigs(unittest.TestCase):
    """Verify the YAML configuration files parse and merge cleanly."""

    def setUp(self) -> None:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf not installed")
        self.OmegaConf = OmegaConf

    def _load_with_defaults(self, path: Path):
        cfg = self.OmegaConf.load(path)
        parents = []
        if isinstance(cfg.get("defaults", None), list):
            for entry in cfg.pop("defaults"):
                parents.append(self.OmegaConf.load(path.parent / f"{entry}.yaml"))
        return self.OmegaConf.merge(*parents, cfg) if parents else cfg

    def test_base_config(self) -> None:
        cfg = self._load_with_defaults(REPO_ROOT / "configs" / "base.yaml")
        self.assertEqual(cfg.model.name, "ultracodec")
        self.assertEqual(cfg.model.sample_rate, 16000)
        self.assertTrue(cfg.model.afr.enabled)
        self.assertEqual(
            cfg.data.vctk.root,
            "/data01/audio_group/m24_yuanjiajun/AP-BWE/VCTK-Corpus-0.92/wav_test",
        )

    def test_stage_configs(self) -> None:
        for name in ["train_stage1.yaml", "train_stage2.yaml", "train_stage3.yaml", "eval.yaml"]:
            cfg = self._load_with_defaults(REPO_ROOT / "configs" / name)
            self.assertIn("model", cfg)


class TestAudioTransform(unittest.TestCase):
    def test_transform_pads_short_clip(self) -> None:
        from ultracodec.data import AudioTransform

        wav = torch.randn(1, 8000)
        transform = AudioTransform(target_sample_rate=16000, segment_length=48000)
        out = transform(wav, sample_rate=16000)
        self.assertEqual(out.shape, (1, 48000))

    def test_transform_crops_long_clip(self) -> None:
        from ultracodec.data import AudioTransform

        wav = torch.randn(1, 96000)
        transform = AudioTransform(
            target_sample_rate=16000,
            segment_length=48000,
            random_crop=False,
        )
        out = transform(wav, sample_rate=16000)
        self.assertEqual(out.shape, (1, 48000))

    def test_transform_resamples(self) -> None:
        from ultracodec.data import AudioTransform

        wav = torch.randn(1, 24000)  # 1 s at 24 kHz
        transform = AudioTransform(target_sample_rate=16000, segment_length=-1)
        out = transform(wav, sample_rate=24000)
        self.assertEqual(out.shape[0], 1)
        # Roughly 1 s of audio at 16 kHz.
        self.assertAlmostEqual(out.shape[-1], 16000, delta=8)


class TestDatasetClasses(unittest.TestCase):
    def test_vctk_handles_missing_root(self) -> None:
        from ultracodec.data import VCTKDataset

        ds = VCTKDataset(
            root_dir="/non/existent/vctk/root",
            split="test",
            sample_rate=16000,
            segment_length=16000,
        )
        self.assertEqual(len(ds), 0)

    def test_librispeech_handles_missing_root(self) -> None:
        from ultracodec.data import LibriSpeechDataset

        ds = LibriSpeechDataset(
            root_dir="/non/existent/librispeech",
            split="test-clean",
            download=False,
            sample_rate=16000,
            segment_length=16000,
        )
        self.assertEqual(len(ds), 0)

    def test_mixed_dataset_concatenates(self) -> None:
        from ultracodec.data import MixedDataset
        from ultracodec.data.dataset import VCTKDataset

        empty1 = VCTKDataset(root_dir="/x", split="train", sample_rate=16000, segment_length=16000)
        empty2 = VCTKDataset(root_dir="/y", split="test", sample_rate=16000, segment_length=16000)
        mixed = MixedDataset([empty1, empty2])
        self.assertEqual(len(mixed), 0)


class TestPlaceholderModel(unittest.TestCase):
    """Verify the placeholder model exposed by ``ultracodec.model`` once filled
    in still imports cleanly today (it is allowed to be empty)."""

    def test_model_module_exists(self) -> None:
        import ultracodec.model as mdl

        self.assertIsNotNone(mdl)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
