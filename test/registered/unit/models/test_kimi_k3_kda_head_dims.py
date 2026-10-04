"""Kimi K3's KDA layers size their values by the KDA head dim, which a
checkpoint may set apart from the MLA layers' value head size."""

import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KDA_HEAD_DIM, MLA_V_HEAD_DIM, HEADS, HIDDEN = 32, 48, 4, 64


@pytest.fixture(scope="module")
def gloo_world():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    publish_build_topology(tp_size=1)
    initialize_model_parallel(backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def build(tp_size):
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        v_head_dim=MLA_V_HEAD_DIM,
        linear_attn_config={
            "head_dim": KDA_HEAD_DIM,
            "num_heads": HEADS,
            "short_conv_kernel_size": 4,
        },
    )
    # Attention TP 1 under TP tp_size: with tp_size > 1 the layer projects
    # q, k and v unfused, through a QKV projection sized by the value dim.
    parallel = SimpleNamespace(
        tp_size=tp_size, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
    )
    with (
        patch.object(kimi_k3, "get_parallel", lambda: parallel),
        get_context().override_server_args(),
    ):
        return kimi_k3.KimiK3DeltaAttention(
            layer_idx=0, hidden_size=HIDDEN, config=config
        )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_values_take_the_kda_head_dim(gloo_world, tp_size):
    layer = build(tp_size)
    assert layer.head_v_dim == KDA_HEAD_DIM
    assert layer.attn.head_v_dim == KDA_HEAD_DIM
    qkv_proj = getattr(layer, "qkv_proj", None)
    assert (qkv_proj is not None) == (tp_size > 1)
    if qkv_proj is not None:
        assert qkv_proj.v_head_size == KDA_HEAD_DIM
