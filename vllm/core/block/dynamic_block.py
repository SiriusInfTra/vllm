from __future__ import annotations
import os, sys
import time, math
from typing import Dict, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar

import torch
import builtins


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

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception as e:
    exit(1234)
from dataclasses import dataclass, field
import heapq
import math
from typing import List, Dict, Optional


T = TypeVar('T')

class PriorityQueue(Generic[T]):
    def __init__(self):
        self._queue: list[T] = []
        self._index = 0
        
    def append(self, item: T) -> None:
        heapq.heappush(self._queue, item)
        self._index += 1
        
    def pop(self) -> T:
        if not self._queue:
            raise IndexError("pop from an empty queue")
        return heapq.heappop(self._queue)
    
    def peek(self) -> T:
        if not self._queue:
            raise IndexError("peek at an empty queue")
        return self._queue[0]
    
    def empty(self) -> bool:
        return len(self._queue) == 0

    def size(self) -> int:
        return len(self._queue)
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._queue)

LLM_NUM_LAYERS = llm_server.get_num_layers()
BLOCK_NBYTES = int(llm_server.get_block_nbytes())
BLOCK_PER_LAYER_NBYTES = int(BLOCK_NBYTES // llm_server.get_num_layers())
assert BLOCK_NBYTES % llm_server.get_num_layers() == 0

_MPOOL_PAGE_NBYTES = 32 * 5 * 1024 * 1024
PAGE_NBYTES = (
    _MPOOL_PAGE_NBYTES * BLOCK_PER_LAYER_NBYTES // 
    math.gcd(_MPOOL_PAGE_NBYTES, BLOCK_PER_LAYER_NBYTES)
)
N_BLOCKS_PER_PAGE = PAGE_NBYTES // BLOCK_PER_LAYER_NBYTES
assert(PAGE_NBYTES % BLOCK_PER_LAYER_NBYTES == 0)

llm_server.info_with_frame(
    '[DynamicBlockAllocator] '
    f'BLOCK_NBYTES={BLOCK_NBYTES} | '
    f'BLOCK_PER_LAYER_NBYTES={BLOCK_PER_LAYER_NBYTES} | '
    f'PAGE_NBYTES={PAGE_NBYTES} | '
    f'N_BLOCKS_PER_PAGE={N_BLOCKS_PER_PAGE}'
)

# assert(N_BLOCKS_PER_PAGE == 512)
# assert(BLOCK_PER_LAYER_NBYTES == 2 * 81920 * 2)

class MemPage:
    def __init__(self, page_id: int) -> None:
        self.page_id = page_id
        self.op_c = 0 # operation counter, for invalidating the free queue
        block_id_begin = page_id // BLOCK_PER_LAYER_NBYTES
        assert(page_id % PAGE_NBYTES == 0)
        self.free_layer_block_ids = list(range(
            block_id_begin, block_id_begin + N_BLOCKS_PER_PAGE))
        self.used_layer_block_ids = list[int]()
        
    @property
    def n_free(self) -> int:
        return len(self.free_layer_block_ids)
    
    @property
    def no_free(self) -> bool:
        return len(self.free_layer_block_ids) == 0
        
    def alloc_block_ids(self, n: int = 1) -> List[int]:
        N = len(self.free_layer_block_ids)
        allocated_blocks = self.free_layer_block_ids[-n:]
        assert len(allocated_blocks) == n
        self.free_layer_block_ids = self.free_layer_block_ids[:-n]
        assert len(allocated_blocks) + len(self.free_layer_block_ids) == N
        self.used_layer_block_ids.extend(allocated_blocks)
        # assert (len(self.used_layer_block_ids) 
        #         + len(self.free_layer_block_ids)
        #     ) == N_BLOCKS_PER_PAGE
        return allocated_blocks

    def free_block_ids(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            self.used_layer_block_ids.remove(block_id)
            # assert block_id not in self.free_layer_block_ids
            self.free_layer_block_ids.append(block_id)


@dataclass(order=True)
class MemPageItem:
    n_free: int = field(init=False, compare=True)
    page: MemPage = field(init=True, compare=False)
    op_c: int = field(init=False, compare=False)

    def __post_init__(self):
        self.n_free = self.page.n_free
        self.op_c = self.page.op_c
    
    @property
    def valid(self) -> bool:
        return self.op_c == self.page.op_c


class PageManager:
    def __init__(self, 
                 n_blocks_per_page: int, 
                 translator: PerLayerBlockTranslator) -> None:
        self.pages_by_n_free = PriorityQueue[MemPageItem]()
        self.pages_by_id = dict[int, MemPageItem]()
        self.pages_by_blk = dict[int, MemPage]()
        self.n_blocks_per_page = n_blocks_per_page
        self.translator = translator
        num_free_blks, num_used_blks = llm_server.get_num_layer_block_info()
        llm_server.info(f'PageManager init with {num_free_blks} free blocks, '
                        f'used block {num_used_blks}')

    def _new_page(self, n = 1) -> None:
        page_ids = llm_server.alloc_kv_cache_page(PAGE_NBYTES, n)                                                                                                                                            
        assert len(page_ids) == n, f'Not enough test page ids: {len(page_ids)} : {n}'
        for page_id in page_ids:
            page = MemPage(page_id)
            item = MemPageItem(page=page)
            self.pages_by_id[page_id] = item
            for block_id in page.free_layer_block_ids:
                self.pages_by_blk[block_id] = page
            self.pages_by_n_free.append(item)
    
    def _update_page(self, page: MemPage) -> None:
        page.op_c += 1 # invalidate the free queue
        self.pages_by_n_free = [item for item in self.pages_by_n_free 
                                if item.page.page_id != page.page_id]

        if page.n_free == self.n_blocks_per_page:
            assert len(page.used_layer_block_ids) == 0
            del self.pages_by_id[page.page_id]
            for block_id in page.free_layer_block_ids:
                del self.pages_by_blk[block_id]
            llm_server.free_kv_cache_page(PAGE_NBYTES, [page.page_id])
        else:
            item = MemPageItem(page=page)
            self.pages_by_id[page.page_id] = item 
            self.pages_by_n_free.append(item)
    
    def get_num_free_blocks(self) -> int:
        num_free_blks, num_used_blks = llm_server.get_num_layer_block_info()
        return num_free_blks // LLM_NUM_LAYERS
    
    def alloc_block_ids(self, n: int) -> List[int]:
        allocated_blocks: list[int] = []
        while (next_allocate_n := n - len(allocated_blocks)) > 0:
            if len(self.pages_by_n_free) == 0:
                n_pages = math.ceil((n - len(allocated_blocks)) / self.n_blocks_per_page)
                self._new_page(n_pages)
            item = self.pages_by_n_free.pop()
            if item.op_c != item.page.op_c: # invalid item
                continue
            page = item.page
            if page.n_free > next_allocate_n:
                allocated_blocks.extend(page.alloc_block_ids(next_allocate_n))
                item.n_free = page.n_free
                self.pages_by_n_free.append(item)
            else:
                allocated_blocks.extend(page.alloc_block_ids(page.n_free))
                assert page.no_free, f"Page {page.page_id} is not empty: {page.n_free}"
        llm_server.update_num_free_layer_blocks(-n)
        assert len(allocated_blocks) == n, \
            f'Allocated {len(allocated_blocks)} blocks instead of {n}'
        return allocated_blocks

    def free_block_ids(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            page_id = self.pages_by_blk[block_id].page_id
            item = self.pages_by_id[page_id]
            assert item.valid, f'Invalid page id {page_id}'
            page = item.page
            page.free_block_ids([block_id])
            self._update_page(page)
        llm_server.update_num_free_layer_blocks(len(block_ids))

@dataclass
class LayerBlockMeta:
    block_id_by_layer: list[int]


INVALID_MAPPING = -666
class PerLayerBlockTranslator:
    def __init__(self, block_ids: list[int]):
        self.block_id_by_blk_layer = torch.zeros(
            (len(block_ids), LLM_NUM_LAYERS), dtype=torch.int32
        ) + INVALID_MAPPING
        self.available_block_ids = block_ids.copy()
        self.available_block_ids.remove(0)
        self.used_block_ids = set()
        self.always_available_page_id = None
    
    def put_block(self, per_layer_block_ids: list[int]) -> BlockId:
        block_id = self.available_block_ids.pop()
        self.block_id_by_blk_layer[block_id] = torch.tensor(
            per_layer_block_ids, dtype=torch.int32)
        assert block_id != 0
        return block_id
    
    def pop_block(self, block_id: BlockId):
        self.available_block_ids.append(block_id)
        ret = self.block_id_by_blk_layer[block_id].tolist()
        self.block_id_by_blk_layer[block_id] = INVALID_MAPPING
        return ret
    

class NaiveDynamicBlockAllocator(BlockAllocator):
    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):
        assert _use_llm_server_kv_cache_pool, \
                "NaiveDynamicBlockAllocator only works with llm_server kv-cache pool"
        self.n_layers = LLM_NUM_LAYERS
        if block_ids is None:
            block_ids = range(num_blocks)

        self.translator = PerLayerBlockTranslator(block_ids)
        self.page_manager = PageManager(
            n_blocks_per_page=N_BLOCKS_PER_PAGE,
            translator=self.translator
        )

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
        builtins.colsys_allocator = self
        logger.info('Init NaiveDynamicBlockAllocator')
 

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
        # logger.info(f'Begin allocate_mutable')
        alloc_blk_begin = time.time()
        block_id = self._allocate_new_block_id()
        alloc_blk_end = time.time()
        

        block = self._create_block(
            prev_block=prev_block,
            token_ids=[],
            block_id=block_id,
            block_size=self._block_size,
            allocator=self,
        )
        # logger.info(f'End allocate_mutable {block_id}')
        return block

    def free(self, block: Block) -> None:
        # logger.info(f'Begin free block id {block.block_id}')
        self._free_block_id(block.block_id)  

        # Mark the block as having no allocation.
        # logger.info(f'End free block id {block.block_id}')

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
        return self.page_manager.get_num_free_blocks()

    def _allocate_new_block_id(self) -> BlockId:
        per_layer_block_ids = self.page_manager.alloc_block_ids(self.n_layers)
        block_id = self.translator.put_block(per_layer_block_ids)
        self._refcounter.incr(block_id)
        return block_id

    def _free_block_id(self, block_id: BlockId) -> None:
        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            per_layer_block_ids = self.translator.pop_block(block_id)
            self.page_manager.free_block_ids(per_layer_block_ids )                                                                                                                                                   

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

