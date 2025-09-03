import os

from fastvideo.worker.gpu_worker import WorkerWrapperBase


try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup
    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x doesn't expose `available_resources_per_node`
        from ray._private.state import state as _state
        available_resources_per_node = _state._available_resources_per_node

    # TODO(xingyu): START FROM HERE!!!
    class RayWorkerWrapper(WorkerWrapperBase):
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazily initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Since the compiled DAG runs a main execution
            # in a different thread that calls cuda.set_device.
            # The flag indicates is set_device is called on
            # that thread.
            self.compiled_dag_cuda_device_set = False


        def get_node_ip(self) -> str:
            # TODO(xingyu): return empty str for now
            return ""
            # return get_ip()

        def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
            node_id = ray.get_runtime_context().get_node_id()
            # TODO(xingyu): hardcode this for now
            device_key = "GPU"
            gpu_ids = ray.get_runtime_context().get_accelerator_ids(
            )[device_key]
            return node_id, gpu_ids


        def setup_device_if_necessary(self):
            # TODO(swang): This is needed right now because Ray CG executes
            # on a background thread, so we need to reset torch's current
            # device.
            # We can remove this API after it is fixed in compiled graph.
            assert self.worker is not None, "Worker is not initialized"
            if not self.compiled_dag_cuda_device_set:
                if current_platform.is_tpu():
                    # Not needed
                    pass
                else:
                    current_platform.set_device(self.worker.device)

                self.compiled_dag_cuda_device_set = True


        def override_env_vars(self, vars: dict[str, str]):
            os.environ.update(vars)

    ray_import_err = None
except ImportError as e:
    ray = None
    ray_import_err = str(e)
    RayWorkerWrapper = None


def initialize_ray_cluster(
    parallel_config=None, ray_address: str | None = None
):
    pass
