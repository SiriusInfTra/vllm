"""CacheEngine class for managing the KV cache."""
from typing import Dict, List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        use_llm_server_kv_cache_pool = False
        if device == "cuda":
            try:
                import llm_server
                use_llm_server_kv_cache_pool = llm_server.use_kv_cache_pool()
                logger.info(f'Using llm_server KV cache pool, block shape {kv_cache_shape}')
            except ImportError:
                pass

        for layer_idx in range(self.num_layers):
            if not use_llm_server_kv_cache_pool:
                kv_cache.append(
                    torch.empty(kv_cache_shape,
                                dtype=self.dtype,
                                pin_memory=pin_memory,
                                device=device))
            else:
                _kv_cache_ts: torch.Tensor = llm_server.init_kv_cache(
                    layer_idx,
                    list(kv_cache_shape),
                    str(self.dtype),
                    self.dtype.itemsize,
                )
                # logger.info(f'layer {layer_idx} block shape {list(kv_cache_shape)} dtype {self.dtype} | ts shape {_kv_cache_ts.shape} dtype {_kv_cache_ts.dtype}')
                kv_cache.append(_kv_cache_ts)
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)

        logger.info(f'[CacheEngine] block size: \n'
            f'\thead_size {head_size} num_heads {num_heads} num_layers {num_layers} '
            f'key_cache_block {key_cache_block} value_cache_block {value_cache_block} '
            f'total {total} dtype_size {dtype_size} | {dtype_size * total / 1024 / 1024} MiB')
        try:
            import llm_server
            llm_server.maybe_set_kv_cache_block_nbytes(
                block_size, num_layers, num_heads, head_size, dtype_size * total
            )
        except ImportError:
            logger.warning('llm_server not found')

        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
