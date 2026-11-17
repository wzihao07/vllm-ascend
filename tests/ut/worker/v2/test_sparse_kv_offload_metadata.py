# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for reusable draft-offload metadata buffers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1 import utils as v1_utils

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.request_identity import RequestIdentityRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    update_sparse_kv_offload_metadata,
)
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState, DefaultModelState
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import sparse_kv_offload
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.sparse_kv_offload import SparseKVOffloadMetadata


@pytest.fixture(autouse=True)
def _cpu_buffers(monkeypatch):
    monkeypatch.setattr(v1_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", lambda: False)


def _metadata(*, enabled=True, num_steps=3):
    return SparseKVOffloadMetadata(
        enabled=enabled,
        num_steps=num_steps,
        max_num_reqs=4,
        max_num_tokens=8,
        device=torch.device("cpu"),
    )


def _batch(*, dummy=False):
    return SimpleNamespace(req_ids=["live-a", "live-b", "not-scheduled"], is_dummy=dummy)


def _state(*, pool=True):
    registry = RequestIdentityRegistry()
    if not pool:
        return SimpleNamespace(_offload_pool_slots=None, _offload_pool_active=None, _offload_request_identity=registry)
    slots = v1_utils.CpuGpuBuffer(4, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    active = v1_utils.CpuGpuBuffer(4, dtype=torch.bool, device=torch.device("cpu"), pin_memory=False)
    stable = v1_utils.CpuGpuBuffer(4, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    slots.cpu.copy_(torch.tensor([7, 8, 9, 10]))
    active.cpu.copy_(torch.tensor([True, False, True, False]))
    # Distinct settled prefixes per request: live-a settled one block (128),
    # live-b settled two (256); padding rows keep 0 so any leak from them is
    # visible in the assertions below.
    stable.cpu.copy_(torch.tensor([128, 256, 0, 0]))
    stable.copy_to_gpu(4)
    return SimpleNamespace(
        _offload_pool_slots=slots,
        _offload_pool_active=active,
        _offload_stable_prefixes=stable,
        _offload_request_identity=registry,
    )


def _descriptor(mode=CUDAGraphMode.FULL, *, num_reqs=4):
    return SimpleNamespace(cg_mode=mode, num_reqs=num_reqs, num_tokens=8)


def test_disabled_does_not_allocate_or_read_batch(monkeypatch):
    allocate = MagicMock(side_effect=AssertionError("disabled offload allocated buffers"))
    pin_memory = MagicMock(side_effect=AssertionError("disabled offload queried pin memory"))
    update = MagicMock(side_effect=AssertionError("disabled offload updated metadata"))
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", pin_memory)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)
    metadata = _metadata(enabled=False)

    assert metadata.req_ids_buffers == metadata.token_to_req_buffers == []
    assert metadata.build_kwargs(object(), object(), 2, object(), object(), 1) is None
    allocate.assert_not_called()
    pin_memory.assert_not_called()
    update.assert_not_called()


def test_enabled_allocates_one_pair_per_step_once(monkeypatch):
    allocate = MagicMock(wraps=v1_utils.CpuGpuBuffer)
    pin_memory = MagicMock(return_value=False)
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", pin_memory)

    metadata = _metadata()

    pin_memory.assert_called_once_with()
    assert allocate.call_count == 6
    for step in range(3):
        req_call, token_call = allocate.call_args_list[2 * step : 2 * step + 2]
        assert req_call.args == (4,)
        assert token_call.args == (8,)
        assert req_call.kwargs == {"dtype": torch.int64, "device": torch.device("cpu"), "pin_memory": False}
        assert token_call.kwargs == {"dtype": torch.int32, "device": torch.device("cpu"), "pin_memory": False}
    assert len({buffer.gpu.data_ptr() for buffer in metadata.req_ids_buffers}) == 3
    assert len({buffer.gpu.data_ptr() for buffer in metadata.token_to_req_buffers}) == 3


def test_full_graph_metadata_preserves_live_ids_pool_cpu_views_and_step(monkeypatch):
    metadata = _metadata()
    state = _state()
    batch = _batch()
    descriptor = _descriptor()
    query_start = np.array([0, 2, 5], dtype=np.int32)
    update = MagicMock(wraps=sparse_kv_offload.update_sparse_kv_offload_metadata)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)

    kwargs = metadata.build_kwargs(batch, state, 2, descriptor, query_start, 1)

    assert kwargs is not None
    update.assert_called_once()
    args = update.call_args.args
    assert args[:5] == (5, 2, 8, 4, ["live-a", "live-b"])
    assert args[5] is query_start
    assert args[6] is metadata.req_ids_buffers[1]
    assert args[7] is metadata.token_to_req_buffers[1]
    assert update.call_args.kwargs == {"request_identity": state._offload_request_identity, "is_dummy": False}
    assert set(kwargs) == {
        "req_ids_tensor",
        "token_to_req",
        "req_topk_buffer_slots",
        "req_topk_buffer_active",
        "req_topk_buffer_stable_prefixes",
        "copy_sfa_draft_index",
        "offload_dummy",
    }
    assert kwargs["req_ids_tensor"].tolist() == [1, 2, 0, 0]
    assert kwargs["token_to_req"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert kwargs["req_topk_buffer_slots"].data_ptr() == state._offload_pool_slots.cpu.data_ptr()
    assert kwargs["req_topk_buffer_slots"].tolist() == [7, 8, 9, 10]
    assert kwargs["req_topk_buffer_active"].data_ptr() == state._offload_pool_active.cpu.data_ptr()
    assert kwargs["copy_sfa_draft_index"] == 1
    # Draft steps must consume the same settled stable prefixes as the
    # target: distinct per-request values, padded rows keep their 0.
    assert kwargs["req_topk_buffer_stable_prefixes"] is not None
    assert kwargs["req_topk_buffer_stable_prefixes"].tolist() == [128, 256, 0, 0]
    assert kwargs["offload_dummy"] is False


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_eager_and_piecewise_use_actual_tokens_and_unpadded_requests(mode):
    metadata = _metadata()

    kwargs = metadata.build_kwargs(
        _batch(), _state(pool=False), 2, _descriptor(mode, num_reqs=0), np.array([0, 2, 5]), 0
    )

    assert kwargs is not None
    assert kwargs["req_ids_tensor"].shape == (2,)
    assert kwargs["token_to_req"].tolist() == [0, 0, 1, 1, 1]
    assert kwargs["req_topk_buffer_slots"] is None
    assert kwargs["req_topk_buffer_active"] is None
    assert kwargs["copy_sfa_draft_index"] == 0


def test_repeated_step_reuses_addresses_clears_padding_and_keeps_other_steps(monkeypatch):
    metadata = _metadata()
    allocate = MagicMock(side_effect=AssertionError("step execution allocated a new metadata buffer"))
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    query_start = np.array([0, 2, 5], dtype=np.int32)
    state = _state()
    step0 = metadata.build_kwargs(_batch(), state, 2, _descriptor(), query_start, 0)
    step1 = metadata.build_kwargs(_batch(), state, 2, _descriptor(), query_start, 1)
    assert step0 is not None
    assert step1 is not None
    saved_step0 = {key: step0[key].clone() for key in ("req_ids_tensor", "token_to_req")}
    previous_addresses = {key: step1[key].data_ptr() for key in ("req_ids_tensor", "token_to_req")}

    repeated = metadata.build_kwargs(
        SimpleNamespace(req_ids=["replacement"]), state, 1, _descriptor(), np.array([0, 1]), 1
    )

    assert repeated is not None
    for key in previous_addresses:
        assert repeated[key].data_ptr() == previous_addresses[key]
        assert repeated[key].data_ptr() != step0[key].data_ptr()
        assert torch.equal(step0[key], saved_step0[key])
    assert repeated["req_ids_tensor"].tolist() == [3, 0, 0, 0]
    assert repeated["token_to_req"].tolist() == [0] * 8
    assert repeated["offload_dummy"] is False
    allocate.assert_not_called()


def test_dummy_uses_distinct_nonzero_ids_when_live_request_list_is_empty(monkeypatch):
    metadata = _metadata()
    update = MagicMock(wraps=sparse_kv_offload.update_sparse_kv_offload_metadata)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)

    kwargs = metadata.build_kwargs(
        SimpleNamespace(req_ids=[], is_dummy=True), _state(pool=False), 2, _descriptor(), np.array([0, 2, 5]), 2
    )

    assert kwargs is not None
    assert update.call_args.args[4] == []
    assert update.call_args.kwargs["is_dummy"] is True
    assert kwargs["req_ids_tensor"].tolist() == [-2, -3, 0, 0]
    assert kwargs["offload_dummy"] is True
    assert kwargs["copy_sfa_draft_index"] == 2


def test_batch_with_no_live_ids_does_not_reuse_previous_request_identity():
    metadata = _metadata()
    metadata.build_kwargs(_batch(), _state(), 2, _descriptor(), np.array([0, 2, 5]), 0)

    with pytest.raises((ValueError, RuntimeError), match="request|req_ids"):
        metadata.build_kwargs(
            SimpleNamespace(req_ids=[], is_dummy=False), _state(pool=False), 2, _descriptor(), np.array([0, 2, 5]), 0
        )


def test_target_and_all_draft_steps_share_owner_namespace_after_reorder_and_dummy():
    metadata = _metadata()
    state = _state(pool=False)
    collision_ids = ["234054828099429543", "471754017250656876"]
    target_ids = v1_utils.CpuGpuBuffer(4, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False)
    target_tokens = v1_utils.CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    update_sparse_kv_offload_metadata(
        2,
        2,
        8,
        4,
        collision_ids,
        np.array([0, 1, 2]),
        target_ids,
        target_tokens,
        request_identity=state._offload_request_identity,
    )
    target_owners = target_ids.gpu[:2].tolist()
    assert target_owners[0] != target_owners[1]

    metadata.build_kwargs(SimpleNamespace(req_ids=[], is_dummy=True), state, 2, _descriptor(), np.array([0, 1, 2]), 0)
    for step in range(3):
        kwargs = metadata.build_kwargs(
            SimpleNamespace(req_ids=collision_ids[::-1], is_dummy=False),
            state,
            2,
            _descriptor(),
            np.array([0, 1, 2]),
            step,
        )
        assert kwargs is not None
        assert kwargs["req_ids_tensor"].tolist() == target_owners[::-1] + [0, 0]

    # Local state destruction (finish/cancel/preemption) creates a new incarnation.
    state._offload_request_identity.remove(collision_ids[0])
    new_kwargs = metadata.build_kwargs(
        SimpleNamespace(req_ids=collision_ids, is_dummy=False),
        state,
        2,
        _descriptor(),
        np.array([0, 1, 2]),
        0,
    )
    assert new_kwargs is not None
    assert new_kwargs["req_ids_tensor"][0] > max(target_owners)
    assert new_kwargs["req_ids_tensor"][1] == target_owners[1]


def test_model_state_removal_ends_owner_incarnation_after_upstream_removal(monkeypatch):
    state = AscendModelState.__new__(AscendModelState)
    state._offload_request_identity = RequestIdentityRegistry()
    state._offload_request_states = SimpleNamespace(remove_request=MagicMock())
    old_owner = state._offload_request_identity.get_or_create("finished")
    retained_owner = state._offload_request_identity.get_or_create("temporarily-unscheduled")
    remove = MagicMock()

    def upstream_remove(actual_state, req_id):
        remove(actual_state, req_id)
        assert actual_state is state
        assert req_id == "finished"
        assert state._offload_request_identity is not None
        assert state._offload_request_identity.get_or_create(req_id) == old_owner

    monkeypatch.setattr(DefaultModelState, "remove_request", upstream_remove)

    state.remove_request("finished")

    remove.assert_called_once_with(state, "finished")
    state._offload_request_states.remove_request.assert_called_once_with("finished")
    assert state._offload_request_identity.get_or_create("finished") > retained_owner
    assert state._offload_request_identity.get_or_create("temporarily-unscheduled") == retained_owner


def test_d2h_retire_drains_before_resetting_preempted_rows():
    calls: list[object] = []
    manager = MagicMock()

    def _retire() -> int:
        calls.append("retire")
        return 2

    def _release(slots):
        calls.append(("release", list(slots)))

    manager.retire_d2h.side_effect = _retire
    manager.release_d2h_slots.side_effect = _release
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_manager = manager
    runner.model_state = SimpleNamespace(
        _offload_pool_generations=object(),
        _offload_pool_slots=SimpleNamespace(np=np.array([3, 5])),
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-a", "req-b"],
        num_computed_tokens_np=np.array([100, 200]),
    )

    # Preemption must not raise: the previous batch drains first, then the
    # preempted request's ledger row is released for re-adoption.
    assert runner._sparse_kv_offload_d2h_retire({"req-b"}) == 2
    assert calls == ["retire", ("release", [5])]

    # Ids absent from the batch, or dummy batches, still retire (one entry
    # per call) but release nothing further.
    def release_count():
        return sum(1 for entry in calls if isinstance(entry, tuple))

    assert runner._sparse_kv_offload_d2h_retire({"req-gone"}) == 2
    assert release_count() == 1
    runner.input_batch.is_dummy = True
    assert runner._sparse_kv_offload_d2h_retire({"req-a"}) == 2
    assert release_count() == 1


def test_host_block_mirror_tracks_appends_and_overwrites():
    tables = AscendBlockTables.__new__(AscendBlockTables)
    tables.max_num_reqs = 4
    tables.num_kv_cache_groups = 1
    tables.redirect_writes_to_null_block = False
    tables.block_tables = [SimpleNamespace(gpu=SimpleNamespace(shape=(4, 6)))]
    tables.num_blocks = SimpleNamespace(np=np.zeros((1, 4), dtype=np.int32))

    mirror = tables.enable_host_block_mirror(0)
    assert mirror.shape == (4, 6)

    # Full admission (overwrite): ids land from the row start. num_blocks is
    # advanced by super().append_block_ids before the mirror write.
    tables.num_blocks.np[0, 1] = 3
    tables._mirror_block_ids(1, [5, 6, 7], overwrite=True)
    assert mirror[1, :3].tolist() == [5, 6, 7]
    assert mirror[1, 3:].tolist() == [0, 0, 0]

    # Decode delta (append): lands after the existing count.
    tables.num_blocks.np[0, 1] = 4
    tables._mirror_block_ids(1, [9], overwrite=False)
    assert mirror[1, :4].tolist() == [5, 6, 7, 9]

    # Slot reuse on readmission replaces the prefix; the stale tail matches
    # the staged-write semantics of the device table.
    tables.num_blocks.np[0, 1] = 1
    tables._mirror_block_ids(1, [8], overwrite=True)
    assert mirror[1, :4].tolist() == [8, 6, 7, 9]


def test_d2h_plan_launch_gathers_rows_from_host_mirror():
    captured: dict[str, np.ndarray] = {}
    manager = MagicMock()
    manager.d2h_host_kv_cache_group_id = 0

    def _capture_plan(computed, slots, generations, table, active):
        captured["table"] = table
        return 0

    manager.plan_d2h_requests.side_effect = _capture_plan
    mirror = np.zeros((4, 6), dtype=np.int32)
    mirror[2] = (10, 11, 12, 0, 0, 0)
    mirror[3] = (20, 21, 0, 0, 0, 0)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_manager = manager
    runner.block_tables = SimpleNamespace(host_block_mirror=mirror)
    runner.model_state = SimpleNamespace(
        # The plan slices the generations buffer, so it needs a real array.
        _offload_pool_generations=SimpleNamespace(np=np.array([4, 6])),
        _offload_pool_slots=SimpleNamespace(np=np.array([1, 0])),
        # Deliberately no .cpu on the device table: reaching the readback
        # fallback would raise, proving the plan consumed the mirror.
        block_tables=(object(),),
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-a", "req-b"],
        num_computed_tokens_np=np.array([10, 20]),
        idx_mapping_np=np.array([2, 3]),
    )

    runner._sparse_kv_offload_d2h_plan_launch()

    assert captured["table"].tolist() == [[10, 11, 12, 0, 0, 0], [20, 21, 0, 0, 0, 0]]
    manager.launch_d2h.assert_not_called()

    # Padded rows map to -1 and clamp to slot 0, matching the device gather.
    runner.input_batch.idx_mapping_np = np.array([3, -1])
    runner._sparse_kv_offload_d2h_plan_launch()
    assert captured["table"].tolist() == [[20, 21, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
