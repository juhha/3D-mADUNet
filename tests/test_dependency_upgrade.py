"""CPU regression checks for the security dependency upgrades.

Run from the repository root with: python -m unittest discover -s tests -v
"""

import copy
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np
import torch
from monai.data import PersistentDataset
from monai.transforms import Invertd
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import yaml

from codes.data_utils import define_trans
from codes.loss.generator_loss import PerceptionLoss3D
from codes.trainer.srgan_trainer import SRGANTrainer


ROOT = Path(__file__).resolve().parents[1]


class UnsupportedCheckpointValue:
    """A custom pickle global that the weights-only loader must reject."""


class DependencyUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        with (ROOT / 'options/brats_t2toflair.yaml').open() as stream:
            self.opts = yaml.safe_load(stream)
        self.opts['checkpoint'] = str(self.directory / 'checkpoint')
        self.opts['train_opt'].update(patch_size=[16, 16, 16], num_patch=1)
        for network in self.opts['network_opt'].values():
            network['params']['base_channels'] = 4
        # Test the real GAN update without downloading pretrained VGG weights.
        self.opts['loss_opt']['generator'].pop('perception_loss')
        self.opts['scaler_opt']['use_scaler'] = False
        self.opts['autocast']['enabled'] = False
        self.args = SimpleNamespace(fold=0, device='cpu', progress=None)

    def test_nifti_preprocessing_cache_and_inverse(self):
        record = {}
        rng = np.random.default_rng(42)
        for key in ('t2', 'flair'):
            path = self.directory / f'{key}.nii.gz'
            volume = rng.uniform(1, 100, (20, 20, 20)).astype(np.float32)
            nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
            record[key] = str(path)
        train_transform, eval_transform, cache_transform = define_trans(self.opts)
        for patch in train_transform(copy.deepcopy(record)):
            self.assertEqual(tuple(patch['t2'].shape), (1, 16, 16, 16))
            self.assertTrue(torch.isfinite(patch['t2']).all())
        evaluated = eval_transform(copy.deepcopy(record))
        evaluated['pred'] = evaluated['flair'].clone()
        inverted = Invertd(
            keys='pred', orig_keys='flair', transform=eval_transform
        )(evaluated)
        self.assertEqual(tuple(inverted['pred'].shape), (1, 20, 20, 20))
        dataset = PersistentDataset(
            [record], transform=cache_transform,
            cache_dir=self.directory / 'cache',
        )
        first = dataset[0]['t2']
        second = dataset[0]['t2']
        torch.testing.assert_close(first, second)
        self.assertTrue(list((self.directory / 'cache').glob('*.pt')))

    def test_gan_training_checkpoint_resume_and_sliding_inference(self):
        trainer = SRGANTrainer(self.opts, self.args)
        batch = {'t2': torch.rand(1, 1, 16, 16, 16),
                 'flair': torch.rand(1, 1, 16, 16, 16)}
        before = next(trainer.net_g.parameters()).detach().clone()
        trainer.train_epoch([batch])
        self.assertFalse(torch.equal(before, next(trainer.net_g.parameters())))
        self.assertTrue((Path(trainer.cp_dir) / 'latest.pt').is_file())
        self.assertTrue((Path(trainer.cp_dir) / 'train_samples/00001.png').is_file())
        trainer.net_g.eval()
        direct = trainer.predict(batch['t2'])
        resumed = SRGANTrainer(self.opts, self.args)
        self.assertEqual(resumed.curr_epoch, 1)
        self.assertTrue(resumed.optimizer_g.state)
        resumed.net_g.eval()
        torch.testing.assert_close(direct, resumed.predict(batch['t2']))
        sliding = resumed.predict(batch['t2'], sliding_inference=True)
        self.assertEqual(tuple(sliding.shape), tuple(batch['t2'].shape))
        self.assertTrue(torch.isfinite(sliding).all())

    def test_checkpoint_loader_rejects_custom_pickle_globals(self):
        checkpoint = Path(self.opts['checkpoint'])
        checkpoint.mkdir()
        torch.save({'unexpected': UnsupportedCheckpointValue()}, checkpoint / 'latest.pt')
        with self.assertRaises(pickle.UnpicklingError):
            SRGANTrainer(self.opts, self.args)

    def test_torchvision_perceptual_loss_backward(self):
        backbone = resnet18(weights=None).eval()
        extractor = create_feature_extractor(backbone, ['layer1'])
        loss_function = PerceptionLoss3D(extractor, torch.nn.L1Loss())
        source = torch.rand(1, 1, 16, 16, 4, requires_grad=True)
        target = torch.rand_like(source)
        loss = loss_function(source, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(source.grad).all())


if __name__ == '__main__':
    unittest.main()
