"""The FA4 wrappers hand a caller's output buffer to the kernel, which writes the
attention output into it, except where they pad MLA heads."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import flash_attention_v4 as fa4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashAttentionV4Out(CustomTestCase):
    def setUp(self):
        self.calls = []

        def kernel(**kwargs):
            # The kernel's own output: kwargs["out"] when given, else new rows
            # of q's heads at v's head size.
            self.calls.append(kwargs)
            q, v = kwargs["q"], kwargs["v"]
            out = kwargs.get("out")
            if out is None:
                out = q.new_empty((*q.shape[:-1], v.shape[-1]))
            out.fill_(3.0)
            return out, q.new_zeros(q.shape[:-1])

        for patcher in (
            patch.object(fa4, "_flash_attn_varlen_func", kernel),
            patch.object(fa4, "_kernel_takes_out", True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def mha(rows=5, heads=2, head_dim=8):
        q = torch.randn(rows, heads, head_dim)
        return q, torch.randn(rows, heads, head_dim), torch.randn(rows, heads, head_dim)

    def test_the_varlen_kernel_writes_into_out(self):
        q, k, v = self.mha()
        out = torch.zeros_like(q)
        result = fa4.flash_attn_varlen_func(q, k, v, out=out)
        self.assertIs(self.calls[-1]["out"], out)
        self.assertEqual(result.data_ptr(), out.data_ptr())
        torch.testing.assert_close(out, torch.full_like(out, 3.0))

    def test_the_paged_kernel_writes_into_out(self):
        q, k, v = self.mha()
        out = torch.zeros_like(q)
        result, lse = fa4.flash_attn_with_kvcache(
            q, k, v, cache_seqlens=5, out=out, return_softmax_lse=True
        )
        self.assertIs(self.calls[-1]["out"], out)
        self.assertEqual(result.data_ptr(), out.data_ptr())
        self.assertEqual(lse.shape, q.shape[:-1])

    def test_without_out_the_kernel_allocates(self):
        q, k, v = self.mha()
        fa4.flash_attn_varlen_func(q, k, v)
        self.assertNotIn("out", self.calls[-1])

    def test_padded_mla_heads_leave_out_untouched(self):
        # Three query heads per KV head are padded to four for the packed
        # kernel, so its output does not fit out.
        rows, kv_heads, q_heads = 5, 2, 6
        q = torch.randn(rows, q_heads, 4)
        qv = torch.randn(rows, q_heads, 16)
        k = torch.randn(rows, kv_heads, 4)
        v = torch.randn(rows, kv_heads, 16)
        out = torch.zeros(rows, q_heads, 16)
        result = fa4.flash_attn_varlen_func(q, k, v, qv=qv, out=out)
        self.assertNotIn("out", self.calls[-1])
        self.assertEqual(self.calls[-1]["q"].shape[-2], 2 * 4)
        self.assertEqual(result.shape, out.shape)
        self.assertEqual(out.count_nonzero().item(), 0)

    def test_a_kernel_without_out_returns_its_own_output(self):
        q, k, v = self.mha()
        out = torch.zeros_like(q)
        with patch.object(fa4, "_kernel_takes_out", False):
            result = fa4.flash_attn_varlen_func(q, k, v, out=out)
        self.assertNotIn("out", self.calls[-1])
        self.assertNotEqual(result.data_ptr(), out.data_ptr())

    def test_the_kernel_takes_out_when_it_declares_it(self):
        def with_out(q, k, v, out=None):
            return out

        def without_out(q, k, v):
            return q

        for kernel, takes in ((with_out, True), (without_out, False)):
            with self.subTest(kernel=kernel.__name__):
                self.assertEqual(fa4._takes_out(kernel), takes)


if __name__ == "__main__":
    unittest.main()
