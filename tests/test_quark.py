# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from heretic.config import ExportStrategy
from heretic.main import obtain_export_strategy
from heretic.model import Model


class FakeQParamsLinear:
    in_features: int
    out_features: int
    weight_quantizer: Any
    logical_weight: torch.Tensor
    dequantization_calls: int

    def __init__(self, logical_weight: torch.Tensor):
        super().__init__()
        self.in_features = logical_weight.shape[1]
        self.out_features = logical_weight.shape[0]
        self.weight = nn.Parameter(
            # Deliberately unrelated to the logical shape. Production code must
            # not derive Linear dimensions from Quark's packed storage layout.
            torch.arange(15, dtype=torch.uint8).view(3, 5),
            requires_grad=False,
        )
        self.weight_quantizer = SimpleNamespace(real_quantized=True)
        self.logical_weight = logical_weight
        self.dequantization_calls = 0

    def _get_qweight(self, weight: torch.Tensor) -> torch.Tensor:
        self.assert_packed_weight_unchanged(weight)
        self.dequantization_calls += 1
        return self.logical_weight

    def assert_packed_weight_unchanged(self, weight: torch.Tensor) -> None:
        torch.testing.assert_close(weight, self.weight)


class FakePeftLinear:
    def __init__(self, base_layer: nn.Module):
        self.base_layer = base_layer


class QuarkWeightTests(unittest.TestCase):
    @patch("heretic.model.is_quark_qparams_linear", return_value=True)
    def test_uses_native_dequantization_without_modifying_packed_weight(self, _):
        logical_weight = torch.arange(8, dtype=torch.float16).view(2, 4)
        base_layer = FakeQParamsLinear(logical_weight)
        packed_before = base_layer.weight.detach().clone()

        result = Model._get_dequantized_weight(FakePeftLinear(base_layer))  # type: ignore[arg-type]

        self.assertEqual(base_layer.dequantization_calls, 1)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, logical_weight.float())
        torch.testing.assert_close(base_layer.weight, packed_before)
        self.assertEqual(base_layer.weight.dtype, torch.uint8)

    @patch("heretic.model.is_quark_qparams_linear", return_value=True)
    def test_rejects_incorrect_logical_shape(self, _):
        base_layer = FakeQParamsLinear(torch.zeros(2, 4))
        base_layer.logical_weight = torch.zeros(2, 2)

        with self.assertRaisesRegex(RuntimeError, "unexpected logical shape"):
            Model._get_dequantized_weight(FakePeftLinear(base_layer))  # type: ignore[arg-type]


class QuarkExportTests(unittest.TestCase):
    def test_forces_adapter_export(self):
        settings = SimpleNamespace(export_strategy=None)
        model = SimpleNamespace(is_quark_quantized=True)

        strategy = obtain_export_strategy(settings, model)  # type: ignore[arg-type]

        self.assertEqual(strategy, ExportStrategy.ADAPTER)

    def test_rejects_explicit_normal_merge(self):
        settings = SimpleNamespace(export_strategy=ExportStrategy.MERGE)
        model = SimpleNamespace(is_quark_quantized=True)

        with self.assertRaisesRegex(RuntimeError, "cannot safely write"):
            obtain_export_strategy(settings, model)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
