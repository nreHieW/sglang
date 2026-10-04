import unittest

import torch

from sglang.kernels.ops.activation import situ_and_mul
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_BETA = 4.0
_LINEAR_BETA = 25.0


def _situ_reference(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


class TestSituAndMul(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_situ_and_mul(self):
        generator = torch.Generator(device="cuda").manual_seed(2)
        hidden_size = 1024
        storage = torch.randn(
            (7, 2 * hidden_size + 16),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_up = storage[:, : 2 * hidden_size]
        output = torch.empty(
            (gate_up.shape[0], hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        returned = situ_and_mul(gate_up, output, beta=_BETA, linear_beta=_LINEAR_BETA)

        self.assertIs(returned, output)
        torch.testing.assert_close(
            returned.float(),
            _situ_reference(gate_up).to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=4e-2,
        )

    def test_a_row_stride_off_the_vector_grid(self):
        # A slice of a fused GEMM output with a few more columns: its row
        # stride is not a whole number of the kernel's 32-byte vectors of the
        # wider dtype, so the kernel reads a dense copy of it.
        generator = torch.Generator(device="cuda").manual_seed(3)
        hidden_size = 1024
        for dtype, extra in ((torch.bfloat16, 3), (torch.float32, 2)):
            with self.subTest(dtype=dtype):
                storage = torch.randn(
                    (7, 2 * hidden_size + extra),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                gate_up = storage[:, : 2 * hidden_size]
                widest = max(gate_up.element_size(), 2)
                self.assertNotEqual(gate_up.stride(0) * widest % 32, 0)

                returned = situ_and_mul(
                    gate_up, None, beta=_BETA, linear_beta=_LINEAR_BETA
                )

                self.assertEqual(returned.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    returned.float(),
                    _situ_reference(gate_up).to(torch.bfloat16).float(),
                    rtol=2e-2,
                    atol=4e-2,
                )


if __name__ == "__main__":
    unittest.main()
