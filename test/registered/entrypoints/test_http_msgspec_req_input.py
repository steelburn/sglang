import dataclasses
import unittest
from typing import Annotated, List

import msgspec
from fastapi import Body, FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BaseReq,
    Function,
    ParseFunctionCallReq,
    SetInternalStateReq,
    Tool,
    UpdateWeightFromDiskReqInput,
    VertexGenerateReqInput,
    _msgpack_decoder,
    _msgpack_encoder,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ToyReqInput(BaseReq, kw_only=True):
    required: str
    count: int = 3
    values: List[str] = msgspec.field(default_factory=list)
    mode: str = "ok"

    def __post_init__(self):
        if self.mode != "ok":
            raise ValueError(f"Invalid mode: {self.mode!r}")


class TestHttpMsgspecReqInput(CustomTestCase):
    def test_pydantic_type_adapter_constructs_msgspec_struct(self):
        adapter = TypeAdapter(ToyReqInput)

        obj = adapter.validate_python({"required": "x"})
        self.assertFalse(dataclasses.is_dataclass(obj))
        self.assertIsInstance(obj, ToyReqInput)
        self.assertEqual(obj.required, "x")
        self.assertEqual(obj.count, 3)
        self.assertEqual(obj.values, [])
        self.assertIsNone(obj.rid)

        first = adapter.validate_python({"required": "x"})
        second = adapter.validate_python({"required": "y"})
        first.values.append("mutated")
        self.assertEqual(second.values, [])

        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            adapter.validate_python({"required": "x", "mode": "bad"})

    def test_msgspec_req_input_is_fastapi_body_param(self):
        app = FastAPI()

        @app.post("/toy")
        def toy(obj: Annotated[ToyReqInput, Body()]):
            return {
                "required": obj.required,
                "count": obj.count,
                "values": obj.values,
                "rid": obj.rid,
                "http_worker_ipc": obj.http_worker_ipc,
            }

        openapi = app.openapi()
        operation = openapi["paths"]["/toy"]["post"]
        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)
        schema_properties = openapi["components"]["schemas"]["ToyReqInput"][
            "properties"
        ]
        self.assertIn("rid", schema_properties)
        self.assertIn("http_worker_ipc", schema_properties)

        client = TestClient(app)
        response = client.post(
            "/toy",
            json={
                "required": "x",
                "rid": "accepted",
                "http_worker_ipc": "worker-0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "required": "x",
                "count": 3,
                "values": [],
                "rid": "accepted",
                "http_worker_ipc": "worker-0",
            },
        )

        response = client.post("/toy", json={"count": 4})
        self.assertEqual(response.status_code, 422)

    def test_update_weight_from_disk_req_input_fastapi_body_validation(self):
        app = FastAPI()

        @app.post("/update_weights_from_disk")
        def update_weights_from_disk(
            obj: Annotated[UpdateWeightFromDiskReqInput, Body()],
        ):
            return {
                "model_path": obj.model_path,
                "load_format": obj.load_format,
                "abort_all_requests": obj.abort_all_requests,
                "weight_version": obj.weight_version,
                "is_async": obj.is_async,
                "torch_empty_cache": obj.torch_empty_cache,
                "keep_pause": obj.keep_pause,
                "recapture_cuda_graph": obj.recapture_cuda_graph,
                "token_step": obj.token_step,
                "flush_cache": obj.flush_cache,
                "manifest": obj.manifest,
                "rid": obj.rid,
            }

        openapi_schema = app.openapi()["components"]["schemas"][
            "UpdateWeightFromDiskReqInput"
        ]
        self.assertIn("model_path", openapi_schema["properties"])
        self.assertIn("rid", openapi_schema["properties"])

        client = TestClient(app)
        response = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "rid": "accepted"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model_path": "/tmp/model",
                "load_format": None,
                "abort_all_requests": False,
                "weight_version": None,
                "is_async": False,
                "torch_empty_cache": False,
                "keep_pause": False,
                "recapture_cuda_graph": False,
                "token_step": 0,
                "flush_cache": True,
                "manifest": None,
                "rid": "accepted",
            },
        )

        response = client.post(
            "/update_weights_from_disk",
            json={"load_format": "auto"},
        )
        self.assertEqual(response.status_code, 422)

    def test_set_internal_state_req_uses_direct_body(self):
        app = FastAPI()

        @app.post("/set_internal_state")
        def set_internal_state(obj: Annotated[SetInternalStateReq, Body()]):
            return {"server_args": obj.server_args}

        client = TestClient(app)
        response = client.post(
            "/set_internal_state",
            json={"server_args": {"pp_max_micro_batch_size": 8}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"server_args": {"pp_max_micro_batch_size": 8}},
        )

        response = client.post(
            "/set_internal_state",
            json={"obj": {"server_args": {"pp_max_micro_batch_size": 8}}},
        )
        self.assertEqual(response.status_code, 422)

    def test_abort_req_accepts_http_rid(self):
        obj = TypeAdapter(AbortReq).validate_python(
            {"rid": "request-id", "abort_all": False}
        )
        self.assertEqual(obj.rid, "request-id")

        app = FastAPI()

        @app.post("/abort_request")
        def abort_request(obj: Annotated[AbortReq, Body()]):
            return {"rid": obj.rid, "abort_all": obj.abort_all}

        openapi_schema = app.openapi()["components"]["schemas"]["AbortReq"]
        self.assertIn("rid", openapi_schema["properties"])

        client = TestClient(app)
        response = client.post(
            "/abort_request",
            json={"rid": "request-id", "abort_all": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"rid": "request-id", "abort_all": False},
        )

    def test_parse_function_call_req_uses_direct_body(self):
        app = FastAPI()

        @app.post("/parse_function_call")
        def parse_function_call(obj: Annotated[ParseFunctionCallReq, Body()]):
            return {
                "text": obj.text,
                "tool_name": obj.tools[0].function.name,
                "tool_type": obj.tools[0].type,
                "tool_call_parser": obj.tool_call_parser,
                "rid": obj.rid,
                "http_worker_ipc": obj.http_worker_ipc,
            }

        operation = app.openapi()["paths"]["/parse_function_call"]["post"]
        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)
        openapi_schema = app.openapi()["components"]["schemas"][
            "ParseFunctionCallReq"
        ]
        self.assertIn("text", openapi_schema["properties"])
        self.assertIn("rid", openapi_schema["properties"])
        self.assertIn("http_worker_ipc", openapi_schema["properties"])

        client = TestClient(app)
        response = client.post(
            "/parse_function_call",
            json={
                "text": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_call_parser": "llama3",
                "rid": "request-id",
                "http_worker_ipc": "worker-0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
                "tool_name": "weather",
                "tool_type": "function",
                "tool_call_parser": "llama3",
                "rid": "request-id",
                "http_worker_ipc": "worker-0",
            },
        )

        response = client.post(
            "/parse_function_call",
            json={
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "weather"},
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_vertex_generate_req_input_uses_direct_body(self):
        app = FastAPI()

        @app.post("/vertex_generate")
        def vertex_generate(vertex_req: Annotated[VertexGenerateReqInput, Body()]):
            return {
                "instances": vertex_req.instances,
                "parameters": vertex_req.parameters,
                "rid": vertex_req.rid,
                "http_worker_ipc": vertex_req.http_worker_ipc,
            }

        operation = app.openapi()["paths"]["/vertex_generate"]["post"]
        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)
        openapi_schema = app.openapi()["components"]["schemas"][
            "VertexGenerateReqInput"
        ]
        self.assertIn("instances", openapi_schema["properties"])
        self.assertIn("parameters", openapi_schema["properties"])
        self.assertIn("rid", openapi_schema["properties"])

        client = TestClient(app)
        response = client.post(
            "/vertex_generate",
            json={
                "instances": [{"prompt": "hello"}],
                "parameters": {"temperature": 0.1},
                "rid": "request-id",
                "http_worker_ipc": "worker-0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "instances": [{"prompt": "hello"}],
                "parameters": {"temperature": 0.1},
                "rid": "request-id",
                "http_worker_ipc": "worker-0",
            },
        )

        response = client.post(
            "/vertex_generate",
            json={"parameters": {"temperature": 0.1}},
        )
        self.assertEqual(response.status_code, 422)

    def test_new_base_req_types_round_trip_through_msgpack(self):
        parse_req = ParseFunctionCallReq(
            text="hello",
            tools=[
                Tool(
                    function=Function(
                        name="weather",
                        parameters={"type": "object", "properties": {}},
                    )
                )
            ],
            rid="parse-rid",
        )
        rebuilt_parse_req = _msgpack_decoder.decode(_msgpack_encoder.encode(parse_req))

        self.assertIsInstance(rebuilt_parse_req, ParseFunctionCallReq)
        self.assertEqual(rebuilt_parse_req.rid, "parse-rid")
        self.assertEqual(rebuilt_parse_req.tools[0].function.name, "weather")
        self.assertEqual(
            rebuilt_parse_req.tools[0].function.parameters,
            {"type": "object", "properties": {}},
        )

        vertex_req = VertexGenerateReqInput(
            instances=[{"prompt": "hello"}],
            parameters={"temperature": 0.1},
            rid="vertex-rid",
        )
        rebuilt_vertex_req = _msgpack_decoder.decode(
            _msgpack_encoder.encode(vertex_req)
        )

        self.assertIsInstance(rebuilt_vertex_req, VertexGenerateReqInput)
        self.assertEqual(rebuilt_vertex_req.rid, "vertex-rid")
        self.assertEqual(rebuilt_vertex_req.instances, [{"prompt": "hello"}])
        self.assertEqual(rebuilt_vertex_req.parameters, {"temperature": 0.1})


if __name__ == "__main__":
    unittest.main()
