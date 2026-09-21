import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fer_model import ClassBalancedFocalLabelSmoothing, MobileNetV3FER


class ComponentTests(unittest.TestCase):
    def test_model_output_shape(self):
        model = MobileNetV3FER(pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, 96, 96))
        self.assertEqual(tuple(output.shape), (2, 7))

    def test_loss_is_finite_and_differentiable(self):
        criterion = ClassBalancedFocalLabelSmoothing(
            class_counts=[100, 10, 80, 150, 90, 85, 60],
            gamma=2.0,
            smoothing=0.05,
        )
        logits = torch.randn(4, 7, requires_grad=True)
        labels = torch.tensor([0, 1, 3, 6])
        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
