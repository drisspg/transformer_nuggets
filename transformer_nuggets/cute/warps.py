import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# from transformer_nuggets.cute.base import CuteOp


class SyncedProducerConsumer:
    """A kernel that syncs producer and consumer threads."""

    def __init__(self):
        super().__init__()

    @cute.kernel()
    def kernel(self, res: cute.Tensor):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        NUM_STAGES: cute.Constexpr = 8

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            staging_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, NUM_STAGES], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 64)

        # Warp 0
        producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)
        # Warp 1
        consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)

        pipeline = cutlass.pipeline.PipelineAsync.create(
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=storage.tma_mbar_ptr.data_ptr(),
        )

        staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(NUM_STAGES))
        staging_smem.fill(0)
        cute.arch.sync_threads()

        producer, consumer = pipeline.make_participants()

        # Producer warp
        if warp_idx == 0:
            for i in cutlass.range(cute.size(res)):
                # Producer: Wait for data buffer is available
                handle = producer.acquire_and_advance()
                # Producer: Write data to shared memory
                with cute.arch.elect_one():
                    cute.printf("Producer: index: {}", handle.index)
                    staging_smem[handle.index] = 1.0 * i
                    # cute.testing.assert_(
                    #     handle.index < NUM_STAGES,
                    #     f"Index out of bounds: {handle.index}",
                    # )
                # Producer: Signal data is ready for consumption
                handle.commit()
            producer.tail()

        # Consumer warp
        if warp_idx == 1:
            for i in cutlass.range(cute.size(res)):
                # Consumer: Wait for producer to signal when data is available for use
                handle = consumer.wait_and_advance()
                # Conumer: consumes data
                with cute.arch.elect_one():
                    cute.printf("Consumer: index: {}", handle.index)
                    res[i] = staging_smem[handle.index]
                # Conumer: Signal data buffer is ready for write
                handle.release()

    @cute.jit()
    def __call__(self, input_cute: cute.Tensor) -> cute.Tensor:
        self.kernel(input_cute).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    def interface(self, inpt: torch.Tensor) -> None:
        inpt_cute = from_dlpack(inpt)
        # This throws LLVM error
        compiled = cute.compile(self.kernel, inpt_cute, options="--enable-assertions")
        compiled(inpt_cute)

        # This throws works
        # self(inpt_cute)


if __name__ == "__main__":
    synced_producer_consumer = SyncedProducerConsumer()
    inpt = torch.zeros((8,), device="cuda")
    synced_producer_consumer.interface(inpt)
    print(inpt)
