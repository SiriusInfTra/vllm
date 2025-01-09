from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from vllm.core.block.common import (CopyOnWriteTracker, RefCounter,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import Block, BlockAllocator

BlockId = int
Refcount = int

_use_llm_server_kv_cache_pool = False
try:
    import llm_server
    _use_llm_server_kv_cache_pool = llm_server.use_kv_cache_pool()
    if _use_llm_server_kv_cache_pool:
        llm_server.info_with_frame("[NaiveDynamicBlockAllocator] use llm_server kv-cache pool")
except:
    raise ImportError("llm_server not found")


@dataclass
class UnifiedBlock:
    nbytes: int = 160 * 1024 * 1024
    free_blocks_id: list[int]
    used_block_id: list[int]

    def __init__(self, mem_block_id_begin: int):
        self.mem_block_id_begin = mem_block_id_begin
        block_id_begin = (mem_block_id_begin * 32 // 40)
        self.free_blocks_id: list[BlockId] = [
            block_id_begin + k for k in range(160 // 40)
        ]
        self.used_block_id: list[BlockId] = []

        



class NaiveDynamicBlockAllocator(BlockAllocator):

    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):
        assert _use_llm_server_kv_cache_pool, "NaiveDynamicBlockAllocator only works with llm_server kv-cache pool"

        if block_ids is None:
            block_ids = range(num_blocks)

        self.unified_blocks_by_free_num = [
            list[UnifiedBlock]() for _ in range((160 * 1024 * 1024) // (32 * 1024 * 1024))
        ] # num_free_pages_id -> list[UnifiedBlock]
        self.unified_blocks_by_block_id = dict[int, UnifiedBlock]() # block_id -> UnifiedBlock

        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._refcounter = RefCounter(
            all_block_indices=self._all_block_indices)
        self._create_block = create_block
        self._block_size = block_size

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly(),
            allocator=self,
        )

    def allocate_immutable(self, prev_block: Optional[Block],
                           token_ids: List[int]) -> Block:
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """
        block = self.allocate_mutable(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_mutable(self, prev_block: Optional[Block]) -> Block:
        """Allocates a new mutable block, linked to the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        block_id = self._allocate_new_block_id()

        return self._create_block(
            prev_block=prev_block,
            token_ids=[],
            block_id=block_id,
            block_size=self._block_size,
            allocator=self,
        )

    def free(self, block: Block) -> None:
        self._free_block_id(block.block_id)  

        # Mark the block as having no allocation.
        block.block_id = None

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            refcount = self._refcounter.incr(block.block_id)
            assert refcount != 1, "can't fork free'd block"

            forked_blocks.append(
                self._create_block(
                    prev_block=prev_block,
                    token_ids=block.token_ids,
                    block_id=block.block_id,
                    block_size=self._block_size,
                    allocator=self,
                ))
            prev_block = forked_blocks[-1]

        return forked_blocks
    
    def get_num_free_blocks(self) -> int:
        return llm_server.get_num_free_blocks()

    def _allocate_new_block_id(self) -> BlockId:
        for free_unified_blocks in self.unified_blocks_by_free_num[1:]:
            if free_unified_blocks:
                unified_block = free_unified_blocks.pop()
                break
        else:
            # alloc mem block here
            mem_block_id_begin = llm_server.alloc_mem_blocks(160 // 32)
            unified_block = UnifiedBlock(mem_block_id_begin)
            for block_id in unified_block.free_blocks_id:
                unified_block.free_blocks_id.append(block_id)
                self.unified_blocks_by_block_id[block_id] = free_unified_blocks
            self.unified_blocks_by_free_num[len(unified_block.free_blocks_id)].append(unified_block)
        block_id = unified_block.free_blocks_id.pop()
        unified_block.used_block_id.append(block_id)
        self.unified_blocks_by_free_num[len(unified_block.free_blocks_id)].append(unified_block)
        self._refcounter.incr(block_id)
        return block_id

    def _free_block_id(self, block_id: BlockId) -> None:
        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            unified_block = self.unified_blocks_by_block_id[block_id]
            del self.unified_blocks_by_free_num[len(unified_block.free_blocks_id)]
            unified_block.free_blocks_id.append(block_id)
            unified_block.used_block_id.remove(block_id)
            if len(unified_block.used_block_id) == 0:
                self.unified_blocks_by_free_num[len(unified_block.free_blocks_id)].remove(unified_block)
                for block_id in unified_block.free_blocks_id:
                    del self.unified_blocks_by_block_id[block_id]
                llm_server.dealloc_mem_blocks(unified_block.mem_block_id_begin, 160 // 32)
            else:
                self.unified_blocks_by_free_num[len(unified_block.free_blocks_id)].append(unified_block)

    @property
    def refcounter(self):
        return self._refcounter

    @property
    def all_block_ids(self):
        return self._all_block_indices

    def cow_block_if_not_appendable(self, block: Block) -> Optional[BlockId]:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            Optional[BlockId]: The block index of the new block if a copy-on
                -write operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        return self._cow_tracker.cow_block_if_not_appendable(block)

    def clear_copy_on_writes(self) -> Dict[BlockId, List[BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            Dict[BlockId, List[BlockId]]: A dictionary mapping source
                block indices to lists of destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_computed(self) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

