import asyncio
import os
import signal
from argparse import Namespace
from typing import Sequence

try:
    import torch  # type: ignore
except ImportError:
    print("torch is not installed. Please install it with `uv pip install torch`.")
    exit(1)

try:
    import uvloop  # type: ignore
except ImportError:
    print("uvloop is not installed. Please install it with `uv pip install uvloop`.")
    exit(1)

try:
    from fastapi import Request  # type: ignore
except ImportError:
    print("fastapi is not installed. Please install it with `uv pip install fastapi`.")
    exit(1)

try:
    from vllm.distributed.device_communicators.pynccl import (  # type: ignore
        PyNcclCommunicator,
    )
    from vllm.distributed.parallel_state import get_world_group  # type: ignore
    from vllm.distributed.utils import StatelessProcessGroup  # type: ignore
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
    from vllm.entrypoints.launcher import serve_http  # type: ignore
    from vllm.entrypoints.openai.api_server import (  # type: ignore
        build_app,
        create_server_socket,
        init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import (  # type: ignore
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.usage.usage_lib import UsageContext  # type: ignore
    from vllm.utils import FlexibleArgumentParser, set_ulimit  # type: ignore
except ImportError:
    print("vLLM is not installed. Please install it with `pip install vllm`.")
    exit(1)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Weight update throttling
MAX_CONCURRENT_WEIGHT_UPDATES = 10
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)

# Track background tasks for cleanup
background_tasks = set()


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. Call close_communicator first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)  # type: ignore
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            torch_dtype (`str`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)  # type: ignore
        self.pynccl_comm.broadcast(weight, src=self.client_rank)  # type: ignore
        self.pynccl_comm.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


async def run_server(args: Namespace):
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    def create_background_task(coro):
        """Create a background task and track it for cleanup"""
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return task

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = (
        "verifiers.inference.vllm_server.WeightSyncWorkerExtension"
    )
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    app = build_app(args)

    @app.get("/health")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.
        """
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.post("/init_communicator")
    async def init_communicator(request: Request):
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")
        create_background_task(
            engine.collective_rpc("init_communicator", args=(host, port, world_size))
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(request: Request):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        data = await request.json()
        name = data.get("name")
        dtype = data.get("dtype")
        shape = data.get("shape")
        shape_tuple = tuple(shape)

        async def throttled_update():
            async with weight_update_semaphore:
                await engine.collective_rpc(
                    "update_named_param", args=(name, dtype, shape_tuple)
                )

        # fire and forget with throttling
        create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache(request: Request):
        # fire and forget
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks():
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator(request: Request):
        # fire and forget
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    vllm_config = await engine.get_vllm_config()
    await init_app_state(engine, vllm_config, app.state, args)
    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # Cancel and wait for background tasks
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    sock.close()


def main():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronization"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args() or Namespace()
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
