import os

try:
    import llm_server
    _enable_dynamic_sm_partition = llm_server.enable_dynamic_sm_partition()
    llm_server.info_with_frame("Dynamic SM partition is enabled.")
except:
    _enable_dynamic_sm_partition = False


def num_tot_tpc():
    return 54


def get_policy():
    from scipy.interpolate import interp1d
    num_batched_tokens_list = [128, 2048]
    target_num_tpc_list     = [30,  45]
    linear_interp_prompt = interp1d(
        num_batched_tokens_list, 
        target_num_tpc_list, 
        kind='linear', 
        fill_value=(target_num_tpc_list[0], target_num_tpc_list[-1]),
        bounds_error=False
    )
    num_batched_tokens_list = [1,  128]
    target_num_tpc_list     = [10, 18]
    linear_interp_decode = interp1d(
        num_batched_tokens_list, 
        target_num_tpc_list, 
        kind='linear', 
        fill_value=(target_num_tpc_list[0], target_num_tpc_list[-1]),
        bounds_error=False
    )

    def tpc_policy(is_prompt: bool, num_batched_tokens: int):
        # return 54
        if is_prompt:
            out = linear_interp_prompt(num_batched_tokens).item()
        else:
            out = linear_interp_decode(num_batched_tokens).item()
        return round(out)
    return tpc_policy


if _enable_dynamic_sm_partition:
    import llm_server
    if 'COLSYS_VLLM_TPC' in os.environ:
        _COLSYS_TPC = int(os.environ['COLSYS_VLLM_TPC'])
    else:
        _COLSYS_TPC = -1
    llm_server.info_with_frame(f'COLSYS_VLLM_TPC: {_COLSYS_TPC}')
    assert -1 <= _COLSYS_TPC <= 54
    assert _COLSYS_TPC != 0, "COLSYS_VLLM_TPC must not be 0."

    if _COLSYS_TPC == -1:
        _COLSYS_TPC_POLICY = get_policy()
    else:
        llm_server.info_with_frame(f'Using fixed TPC: {_COLSYS_TPC} to profile the model.')
        def _tpc_policy(is_prompt: bool, num_batched_tokens: int):
            return _COLSYS_TPC
        _COLSYS_TPC_POLICY = _tpc_policy
else:
    llm_server.info(f'No TPC Set.')
    _COLSYS_TPC = -2 # deprecated

# -2 -> DISABLED
# -1 -> USE_POLICY
#  0 -> UNSET
# >0 -> USE TPC
