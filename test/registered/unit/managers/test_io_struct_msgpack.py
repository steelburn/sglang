import unittest
import warnings
from array import array
from typing import Any, Optional

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BaseReq,
    enc_hook,
    hook_custom_types,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.observability import trace as trace_module  # noqa: E402
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    MetricsCollectorWrapper,
)
from sglang.srt.observability.trace import TraceSpan  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MsgpackPayload(BaseReq, kw_only=True):
    tensor: torch.Tensor
    scalar_tensor: torch.Tensor
    np_array: np.ndarray
    int_array: array
    np_scalar: Any


class RuntimeHandlePayload(BaseReq, kw_only=True):
    metrics_collector: Optional[MetricsCollectorWrapper] = None
    span: Optional[TraceSpan] = None


hook_custom_types(MsgpackPayload, RuntimeHandlePayload)


class TestIoStructMsgpack(CustomTestCase):
    def test_tensor_enc_hook_uses_serializable_dtype_and_bytes(self):
        shape, dtype, raw_data = enc_hook(torch.tensor(7, dtype=torch.int64))

        self.assertEqual(shape, torch.Size([]))
        self.assertEqual(dtype, "int64")
        self.assertIsInstance(raw_data, bytes)
        self.assertEqual(len(raw_data), 8)

    def test_tensor_numpy_and_array_round_trip(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()[1:]
        scalar_tensor = torch.tensor(7, dtype=torch.int64)
        np_array = np.arange(12, dtype=np.float32).reshape(3, 4).T[1:]
        int_array = array("i", [1, 2, 3])

        payload = MsgpackPayload(
            tensor=tensor,
            scalar_tensor=scalar_tensor,
            np_array=np_array,
            int_array=int_array,
            np_scalar=np.float32(1.25),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            rebuilt = msgpack_decode(msgpack_encode(payload))

        self.assertIsInstance(rebuilt, MsgpackPayload)
        self.assertEqual(rebuilt.tensor.dtype, tensor.dtype)
        self.assertEqual(rebuilt.tensor.shape, tensor.shape)
        self.assertTrue(torch.equal(rebuilt.tensor, tensor))
        self.assertEqual(rebuilt.scalar_tensor.dtype, scalar_tensor.dtype)
        self.assertEqual(rebuilt.scalar_tensor.shape, scalar_tensor.shape)
        self.assertTrue(torch.equal(rebuilt.scalar_tensor, scalar_tensor))
        self.assertEqual(rebuilt.np_array.dtype, np_array.dtype)
        self.assertEqual(rebuilt.np_array.shape, np_array.shape)
        self.assertTrue(np.array_equal(rebuilt.np_array, np_array))
        self.assertEqual(rebuilt.int_array, int_array)
        self.assertEqual(rebuilt.np_scalar, 1.25)
        self.assertIsInstance(rebuilt.np_scalar, float)

    def test_process_local_runtime_handles_are_dropped(self):
        if trace_module.opentelemetry_imported:
            span = trace_module.trace.NonRecordingSpan(
                trace_module.trace.INVALID_SPAN_CONTEXT
            )
        else:
            span = TraceSpan()

        payload = RuntimeHandlePayload(
            metrics_collector=MetricsCollectorWrapper(object()),
            span=span,
        )

        rebuilt = msgpack_decode(msgpack_encode(payload))

        self.assertIsNone(rebuilt.metrics_collector)
        self.assertIsNone(rebuilt.span)


if __name__ == "__main__":
    unittest.main()
