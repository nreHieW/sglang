"""Kimi K3's dense FFN declares the TP width its MLP is built with, so the
boundary owes the MLP's sum over that width whatever --moe-dense-tp-size
says: the MLP shards over the full TP group, or over attention TP with
--enable-dense-mlp-attn-tp."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
from torch import nn

from sglang.srt.layers import linear
from sglang.srt.layers.communication import k3_sp_collective
from sglang.srt.layers.layer_boundary import BatchVariant, SumGroup, layer_stack
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Layer 0 is dense.
CONFIG = SimpleNamespace(
    is_moe=True, num_experts=8, first_k_dense_replace=1, moe_layer_freq=1
)


PARALLEL = dict(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)


def dense_layer(*, dense_attn_tp):
    """A K3 dense layer at TP 4 = attention DP 2 x attention TP 2, with the
    MLP K3 builds for it."""
    parallel = fixture.parallel_of(**PARALLEL, enable_dense_mlp_attn_tp=dense_attn_tp)
    with (
        patch.object(kimi_k3, "get_parallel", lambda: parallel),
        patch.object(linear, "get_parallel", lambda: parallel),
        patch.object(kimi_k3, "is_dp_attention_enabled", return_value=True),
    ):
        mlp = kimi_k3.KimiK3MLP(
            hidden_size=64, intermediate_size=128, hidden_act="silu"
        )
    layer = kimi_k3.KimiK3DecoderLayer.__new__(kimi_k3.KimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = mlp
    layer.use_attn_residuals = False
    layer.is_block_write_layer = False
    layer._is_moe_layer = False
    layer._sp_moe = False
    layer._dp_attention = True
    layer.all_reduce_fusion = False
    layer._ffn_writes_stream = False
    layer.input_layernorm = fixture.Norm()
    layer.post_attention_layernorm = fixture.Norm()
    return layer


def declare(layer):
    parallel = fixture.parallel_of(**PARALLEL)
    with (
        patch.object(kimi_k3, "get_parallel", lambda: parallel),
        patch.object(k3_sp_collective, "enabled", return_value=False),
        patch.object(kimi_k3, "_carries_bank_slices", return_value=False, create=True),
        patch.object(kimi_k3, "_shards_moe_rows", return_value=False, create=True),
        fixture.planning(parallel, boundary_reduction="ar"),
        layer_stack(),
    ):
        layer._declare_stages(CONFIG, 0, None)
    return layer.ffn_boundary


class TestKimiK3DenseFfnWidth(CustomTestCase):
    def test_a_full_tp_mlp_owes_its_tp_sum(self):
        layer = dense_layer(dense_attn_tp=False)
        self.assertEqual(layer.mlp.down_proj.tp_size, 4)
        ffn = declare(layer)
        self.assertEqual(ffn.declaration.dense_tp_size, 4)
        path = ffn.plan.paths[BatchVariant.ORDINARY]
        # Every DP rank's rows, gathered for the MLP's TP shards, and the sum
        # over those shards still owed.
        self.assertEqual(path.entry.input_rows.sharded, frozenset())
        self.assertIs(path.output.group, SumGroup.TP)

    def test_an_attention_tp_mlp_owes_its_attention_tp_sum(self):
        layer = dense_layer(dense_attn_tp=True)
        self.assertEqual(layer.mlp.down_proj.tp_size, 2)
        ffn = declare(layer)
        self.assertEqual(ffn.declaration.dense_tp_size, 2)
        path = ffn.plan.paths[BatchVariant.ORDINARY]
        self.assertEqual(path.entry.input_rows.sharded, frozenset({TokenAxis.ATTN_DP}))
        self.assertIs(path.output.group, SumGroup.ATTN_TP)


if __name__ == "__main__":
    unittest.main()
