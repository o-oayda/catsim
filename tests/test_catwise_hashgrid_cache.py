import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from astropy.table import Table
from catsim.simulator import Catwise


class CatwiseHashgridCacheTests(unittest.TestCase):
    def model(self, root, no_ecl=False):
        path = root / 'catalogue.fits'
        if not path.exists():
            Table({
                'w1': [12., 13.], 'w2': [11., 12.],
                'w1cov': [100., 110.], 'w2cov': [95., 100.],
                'w1e': [0.1, 0.2], 'w2e': [0.2, 0.3],
            }).write(path)
        model = Catwise.__new__(Catwise)
        model.cfg = SimpleNamespace(source_catalogue_path=str(path), hashgrid_cache_dir=str(root / 'cache'), base_mask_version='S21')
        model.cut_path = '0p5_17p0'
        model.nside = 64
        model.use_noecl_mask = no_ecl
        model.catalogue_is_loaded = False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model._source_catalogue_spec = Mock(return_value={'filename': path.name, 'sha256': digest})
        model.determine_masked_pixels = Mock()
        def mask():
            model.masked_catalogue = model.catalogue
        model.make_masked_catalogue = Mock(side_effect=mask)
        return model

    def test_build_and_reuse_without_source_for_both_masks(self):
        for no_ecl in (False, True):
            with self.subTest(no_ecl=no_ecl), tempfile.TemporaryDirectory() as tmp:
                model = self.model(Path(tmp), no_ecl)
                model._initialise_mag_cov_hashgrid()
                cache = model._hashgrid_cache_path(not no_ecl)
                self.assertTrue(cache.exists())
                self.assertEqual(model.hashgrid.metadata['mask_north_ecliptic'], not no_ecl)
                np.testing.assert_array_equal(model.hashgrid.grid_step, [0.1, 0.01, 0.1, 0.02])
                self.assertTrue(model.hashgrid.metadata['project_coverage'])
                model.determine_masked_pixels.assert_called_once_with(mask_north_ecliptic=not no_ecl)
                with patch.object(model, 'load_catalogue') as load:
                    model.cfg.source_catalogue_path = None
                    model.catalogue_is_loaded = False
                    model._initialise_mag_cov_hashgrid()
                    load.assert_not_called()

    def test_missing_source_and_mismatch_do_not_write_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(Path(tmp))
            cache = model._hashgrid_cache_path(True)
            model.cfg.source_catalogue_path = None
            with self.assertRaisesRegex(FileNotFoundError, 'source_catalogue_path'):
                model._initialise_mag_cov_hashgrid()
            model.cfg.source_catalogue_path = str(Path(tmp) / 'catalogue.fits')
            model._source_catalogue_spec.return_value['sha256'] = '0' * 64
            with patch('catsim.simulator.Table.read') as read:
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    model._initialise_mag_cov_hashgrid()
                read.assert_not_called()
            self.assertFalse(cache.exists())

    def test_invalid_and_stale_caches_rebuild(self):
        for state in ('pointer', 'legacy', 'stale', 'corrupt'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                model = self.model(Path(tmp))
                cache = model._hashgrid_cache_path(True)
                cache.parent.mkdir(parents=True)
                if state in ('pointer', 'corrupt'):
                    cache.write_text('version https://git-lfs.github.com/spec/v1\n' if state == 'pointer' else 'broken')
                else:
                    metadata = model._hashgrid_metadata([0.1, 0.01, 0.1, 0.02], True, True)
                    metadata['source_catalogue_sha256'] = 'wrong'
                    import json
                    np.savez(cache, **({'metadata': json.dumps(metadata)} if state == 'stale' else {'x': [1]}))
                model._initialise_mag_cov_hashgrid()
                self.assertEqual(model.hashgrid.metadata['source_catalogue_sha256'], model.source_catalogue_sha256)
                model.make_masked_catalogue.assert_called_once()

    def test_mask_versions_have_separate_cache_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(Path(tmp))
            old = model._hashgrid_cache_path(True)
            model.cfg.base_mask_version = 'S22'
            self.assertNotEqual(old, model._hashgrid_cache_path(True))
            self.assertNotEqual(model._hashgrid_cache_path(False), model._hashgrid_cache_path(True))

    def test_default_precompute_delegation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(Path(tmp))
            model.create_mag_cov_hashgrid = Mock()
            model._create_default_mag_cov_hashgrid()
            model.create_mag_cov_hashgrid.assert_called_once_with(grid_step=[0.1, 0.01, 0.1, 0.02], project_coverage=True)

    def test_tracked_catalogue_manifest(self):
        model = Catwise.__new__(Catwise)
        model.cut_path = '0p5_17p0'
        spec = model._source_catalogue_spec()
        self.assertEqual(spec['sha256'], 'f9e4543f673f4ac51f77dccc028655411df953a9822ca4735529974a36ab4dd4')
        model.cut_path = 'unregistered'
        with self.assertRaisesRegex(ValueError, 'No trusted source'):
            model._source_catalogue_spec()
