# SPDX-License-Identifier: Apache-2.0

import fastvideo.envs as envs

from typing import Any, Callable, override
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.executor import DistributedExecutorBase 
from fastvideo.worker.ray_utils import initialize_ray_cluster


class RayDistributedExecutor(DistributedExecutorBase):
    """Ray-based distributed executor"""

    def _init_executor(self) -> None:
        initialize_ray_cluster()

    def _init_worker_ray(self, placement_group, **ray_remote_kwargs):
        pass

    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        responses = self.collective_rpc("execute_forward",
                                        kwargs={
                                            "forward_batch": forward_batch,
                                            "fastvideo_args": fastvideo_args,
                                        })
        output = responses[0]["output_batch"]

        logging_info = None
        if envs.FASTVIDEO_STAGE_LOGGING:
            logging_info = responses[0]["logging_info"]

        result_batch = ForwardBatch(data_type=forward_batch.data_type,
                                    output=output,
                                    logging_info=logging_info)
        return result_batch

    @override
    def _run_workers(self, method: str | Callable, *args, async_run_tensor_parallel_workers_only: bool = False, max_concurrent_workers: int | None = None, **kwargs) -> Any:
        pass
