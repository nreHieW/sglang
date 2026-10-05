"""DeepSeek's one-shot MHA hands attention K and V for every token of the
sequences, the cached prefix too, and declares their extent: their row count
says nothing about the queries'."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOneShotMhaExtent(CustomTestCase):
    def test_one_shot_mha_declares_its_key_value_extent(self):
        calls = []

        def attn_mha(q, k, v, forward_batch, **kwargs):
            calls.append(kwargs)
            return torch.zeros(q.shape[0], 2, 2)

        attn = SimpleNamespace(
            attn_mha=attn_mha,
            num_local_heads=2,
            v_head_dim=2,
            o_proj=lambda x: (x, None),
        )
        attn.forward_normal_core = DeepseekMHAForwardMixin.forward_normal_core.__get__(
            attn
        )
        forward_batch = SimpleNamespace(
            extend_prefix_lens_cpu=[0],
            num_prefix_chunks=None,
            mha_return_lse=True,
            set_attn_attend_prefix_cache=lambda attend: None,
        )
        # 8 query rows (5 real, padded to 8) and 8 K/V rows (3 cached + 5).
        DeepseekMHAForwardMixin.forward_normal_one_shot_core(
            attn,
            torch.zeros(8, 2, 4),
            torch.zeros(8, 2, 4),
            torch.zeros(8, 2, 2),
            forward_batch,
        )
        self.assertEqual(calls[0]["key_value_num_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
