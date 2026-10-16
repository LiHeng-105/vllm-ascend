from types import MappingProxyType
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.awq_config import (
    AscendAWQConfig,
    _detect_unquantized_modules_from_checkpoint,
)
from vllm_ascend.quantization.awq_method import AscendAWQLinearMethod, AscendAWQMoEMethod
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.utils import detect_quantization_method
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


class TestAWQQuantization(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AscendAWQConfig.from_config(self.sample_config)
        self.awq_quant_config.packed_modules_mapping = MappingProxyType({})

        mock_ascend_config = MagicMock()
        mock_ascend_config.eplb_config.dynamic_eplb = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.quantization.awq_method.get_ascend_config",
            return_value=mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self.addCleanup(self._patch_get_ascend_config.stop)

    def test_detect_quantization_method_from_config(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                }
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)

            detected = detect_quantization_method(tmpdir)
            self.assertEqual(detected, AWQ_QUANTIZATION_METHOD)


class TestAWQQuantization(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AscendAWQConfig.from_config(self.sample_config)
        self.awq_quant_config.packed_modules_mapping = MappingProxyType({})

        mock_ascend_config = MagicMock()
        mock_ascend_config.eplb_config.dynamic_eplb = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.quantization.awq_method.get_ascend_config",
            return_value=mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self.addCleanup(self._patch_get_ascend_config.stop)

    def test_init(self):
        self.assertEqual(self.awq_quant_config.group_size, 128)
        self.assertEqual(self.awq_quant_config.weight_bits, 4)
        self.assertTrue(self.awq_quant_config.zero_point)
        self.assertEqual(self.awq_quant_config.modules_to_not_convert, ["visual"])

    def test_init_with_invalid_bits(self):
        invalid_config = self.sample_config.copy()
        invalid_config["bits"] = 8
        with self.assertRaises(ValueError):
            AscendAWQConfig.from_config(invalid_config)

    def test_get_name(self):
        self.assertEqual(self.awq_quant_config.get_name(), AWQ_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = self.awq_quant_config.get_supported_act_dtypes()
        self.assertIn(torch.float16, supported_dtypes)
        self.assertIn(torch.bfloat16, supported_dtypes)
        self.assertEqual(len(supported_dtypes), 2)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AscendAWQConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendAWQConfig.get_config_filenames()
        self.assertIn("quant_config.json", filenames)
        self.assertIn("quantize_config.json", filenames)
        self.assertEqual(len(filenames), 2)

    def test_from_config(self):
        config = AscendAWQConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AscendAWQConfig)

    def test_detect_unquantized_modules_from_checkpoint(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # No index file: function should return input unchanged.
            result = _detect_unquantized_modules_from_checkpoint(
                tmpdir, ["visual"])
            self.assertEqual(result, ["visual"])

            # Checkpoint with only qweight: no extra modules added.
            index = {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.qweight": "model-00001.safetensors",
                    "model.layers.0.mlp.shared_expert.gate_proj.qweight": "model-00001.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            result = _detect_unquantized_modules_from_checkpoint(
                tmpdir, ["visual"])
            self.assertEqual(result, ["visual"])

            # Mixed checkpoint: experts quantized, attention/shared expert fp16.
            mixed_index = {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model-00001.safetensors",
                    "model.layers.0.mlp.shared_expert.gate_proj.weight": "model-00001.safetensors",
                    "model.layers.0.mlp.shared_expert_gate.weight": "model-00001.safetensors",
                    "model.layers.0.mlp.experts.0.gate_proj.qweight": "model-00001.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(mixed_index, f)
            result = _detect_unquantized_modules_from_checkpoint(
                tmpdir, ["visual"])
            self.assertIn("self_attn", result)
            self.assertIn("mlp.shared_expert", result)
            self.assertIn("mlp.shared_expert_gate", result)
            self.assertIn("visual", result)

    def test_get_quant_method_for_linear(self):
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        quant_method = self.awq_quant_config.get_quant_method(
            linear_layer, "visual")
        self.assertIsInstance(quant_method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        quant_method = self.awq_quant_config.get_quant_method(
            linear_layer, "attn")
        self.assertIsInstance(quant_method, AscendAWQLinearMethod)

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_config = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = fused_moe_config

        # Test quantized layer
        quant_method = self.awq_quant_config.get_quant_method(
            fused_moe_layer, "attn")
        self.assertIsInstance(quant_method, AscendAWQMoEMethod)


class TestAWQLinearAscendMethod(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AscendAWQConfig.from_config(self.sample_config)
        self.method = AscendAWQLinearMethod(self.awq_quant_config)

    def test_create_weights(self):
        with patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0), \
             patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1):

            layer = MagicMock(spec=LinearBase)
            self.method.create_weights(
                layer=layer,
                input_size_per_partition=128,
                output_partition_sizes=[64],
                input_size=128,
                output_size=64,
                params_dtype=torch.float16,
            )
            layer.register_parameter.assert_any_call("qweight", ANY)
            layer.register_parameter.assert_any_call("qzeros", ANY)
            layer.register_parameter.assert_any_call("scales", ANY)

    def test_process_weights_after_loading(self):
        layer = MagicMock(spec=LinearBase)
        layer.qweight = torch.randint(10, (64, 128), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.qweight[0][0] = 0x75316420
        layer.qzeros = torch.randint(
            10, (1, 128 // self.awq_quant_config.group_size),
            dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.qzeros[0][0] = 0x75316420
        layer.scales = torch.randn(1,
                                   128 // self.awq_quant_config.group_size,
                                   dtype=torch.float16)

        self.method.process_weights_after_loading(layer)
        # unpacked and signed number. eg: 0 -> 1000b(-8 in int4) -> 0x8 in uint32
        self.assertEqual(layer.qweight[0][0].to(torch.uint32), 0xFEDCBA98)

    def test_apply(self):
        with patch.object(torch_npu, "npu_weight_quant_batchmatmul", create=True) as mock_func:
            layer = MagicMock(spec=LinearBase)
            layer.qweight = torch.randint(10, (64, 128), dtype=torch.int32)
            layer.qzeros = torch.randint(
                -8, 8,
                (8, 128 // self.awq_quant_config.group_size)).to(torch.float16)
            layer.scales = torch.randn(1,
                                       128 // self.awq_quant_config.group_size,
                                       dtype=torch.float16)

            x = torch.randn(2, 16, 128, dtype=torch.float16)
            self.method.apply(layer, x)
            mock_func.assert_called_once()


class TestAWQMoEAscendMethod(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AscendAWQConfig.from_config(self.sample_config)
        self.moe_config = MagicMock(spec=FusedMoEConfig)

        mock_ascend_config = MagicMock()
        mock_ascend_config.eplb_config.dynamic_eplb = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.quantization.awq_method.get_ascend_config",
            return_value=mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self.addCleanup(self._patch_get_ascend_config.stop)

        self.method = AscendAWQMoEMethod(self.awq_quant_config, self.moe_config)

    def test_create_weights(self):
        layer = MagicMock(spec=FusedMoE)
        self.method.create_weights(
            layer,
            num_experts=4,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )

        layer.register_parameter.assert_any_call("w13_qweight", ANY)
        layer.register_parameter.assert_any_call("w2_qweight", ANY)
        layer.register_parameter.assert_any_call("w13_scales", ANY)
        layer.register_parameter.assert_any_call("w2_scales", ANY)
        layer.register_parameter.assert_any_call("w13_qzeros", ANY)
        layer.register_parameter.assert_any_call("w2_qzeros", ANY)

    def test_process_weights_after_loading(self):
        layer = MagicMock(spec=FusedMoE)
        layer.register_parameter = lambda name, param: setattr(
            layer, name, param)
        layer.w13_qweight = torch.randint(10, (4, 128, 256), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w13_qweight[0][0][0] = 0x75316420
        layer.w13_qzeros = torch.randint(10, (4, 2), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w13_qzeros[0][0] = 0x75316420
        layer.w13_scales = torch.randn(4, 2, dtype=torch.float16)

        layer.w2_qweight = torch.randint(10, (4, 256, 128), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w2_qweight[0][0][0] = 0x75316420
        layer.w2_qzeros = torch.randint(10, (4, 2), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w2_qzeros[0][0] = 0x75316420
        layer.w2_scales = torch.randn(4, 2, dtype=torch.float16)

        self.method.process_weights_after_loading(layer)

        # unpacked and signed number. eg: 0 -> 1000b(-8 in int4) -> 0x8 in uint32
        self.assertEqual(layer.w13_qweight[0][0][0].to(torch.uint32),
                         0xFEDCBA98)
        self.assertEqual(layer.w2_qweight[0][0][0].to(torch.uint32),
                         0xFEDCBA98)

    def test_apply(self):
        layer = MagicMock(spec=FusedMoE)
        layer.w13_qweight = torch.randint(10, (4, 128, 256), dtype=torch.int32)
        layer.w13_scales = torch.randn(4, 2, dtype=torch.float16)
        layer.w13_qzeros = torch.randint(
            -8, 8, (4, 2, 256)).to(torch.float16)
        layer.w2_qweight = torch.randint(10, (4, 256, 128), dtype=torch.int32)
        layer.w2_scales = torch.randn(4, 2, dtype=torch.float16)
        layer.w2_qzeros = torch.randint(
            -8, 8, (4, 2, 128)).to(torch.float16)
        layer.swiglu_limit = 0.0

        topk_weights = torch.randn(8, 2, dtype=torch.float16)
        topk_ids = torch.randint(0, 4, (8, 2), dtype=torch.int32)
        x = torch.randn(8, 128, dtype=torch.float16)

        mock_comm_method = MagicMock()
        expected_result = FusedExpertsResult(routed_out=x)
        mock_comm_method.fused_experts.return_value = expected_result

        mock_extra_ctx = MagicMock()
        mock_extra_ctx.moe_comm_method = mock_comm_method

        with patch("vllm_ascend.quantization.awq_method._EXTRA_CTX", mock_extra_ctx):
            result = self.method.apply(layer, x, topk_weights, topk_ids, top_k=2)

        self.assertIs(result, expected_result)
        mock_comm_method.fused_experts.assert_called_once()
        fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIsInstance(fused_input, MoEFusedExpertsInput)
        self.assertEqual(fused_input.quant.quant_type, QuantType.AWQ)
        self.assertFalse(fused_input.need_trans)
        self.assertIs(fused_input.weights.w1, layer.w13_qweight)
        self.assertIs(fused_input.weights.w2, layer.w2_qweight)
        self.assertIs(fused_input.weights.w1_scale, layer.w13_scales)
        self.assertIs(fused_input.weights.w2_scale, layer.w2_scales)
        self.assertIs(fused_input.weights.w1_offset, layer.w13_qzeros)
        self.assertIs(fused_input.weights.w2_offset, layer.w2_qzeros)
