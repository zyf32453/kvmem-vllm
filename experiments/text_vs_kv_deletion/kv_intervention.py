"""Experiment-only synchronous KV/GDN snapshots on vLLM's native page pool.

The scheduler admits a prefix hit only for a bound replay request. The worker
restores bytes into that request's own pages before the model can execute.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs

from experiments.graph_cleanup.gpu_worker import Measurement, sha
from .prepare import encode, encode_variant, file_hash, load_tokenizer
from .protocol import PROBES, allocation_slots, fingerprint, require

BINDING_KEY = 'text_vs_kv_snapshot'


@dataclass
class SnapshotLoad(KVConnectorMetadata):
    request_id: str | None = None
    binding: dict | None = None
    block_ids: tuple | None = None


class SnapshotConnector(KVConnectorBase_V1, SupportsHMA):
    """Report a verified synchronous prefix hit to the native scheduler."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.instance = self._kv_transfer_config.get_from_extra_config('instance', None)
        self.pending = SnapshotLoad()
        self.restore = None
        require(bool(self.instance) and not vllm_config.cache_config.enable_prefix_caching,
                'Snapshot connector requires an isolated cache instance with APC off')

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        binding = (request.kv_transfer_params or {}).get(BINDING_KEY)
        if binding is None:
            return 0, False
        ids = request.prompt_token_ids
        boundary = binding['history_tokens']
        require(num_computed_tokens == 0 and request.num_preemptions == 0
                and binding['cache_instance'] == self.instance
                and type(boundary) is int and 0 < boundary < len(ids)
                and binding['prompt_ids_sha256'] == fingerprint(ids)
                and binding['history_ids_sha256'] == fingerprint(ids[:boundary]),
                'Warm scheduler request did not match its frozen prefix')
        return boundary, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        binding = (request.kv_transfer_params or {}).get(BINDING_KEY)
        if binding is None:
            require(num_external_tokens == 0, 'Unbound request received an external prefix')
            return
        require(num_external_tokens == binding['history_tokens'] and self.pending.request_id is None,
                'Scheduler did not allocate one complete warm prefix')
        self.pending = SnapshotLoad(request.request_id, binding, blocks.get_block_ids())

    def build_connector_meta(self, scheduler_output):
        result, self.pending = self.pending, SnapshotLoad()
        return result

    def start_load_kv(self, forward_context, **kwargs):
        request = self._get_connector_metadata()
        if request.request_id is not None:
            require(self.restore is not None, 'Warm request lacks a worker snapshot owner')
            self.restore(request, forward_context.attn_metadata)

    def wait_for_layer_load(self, layer_name):
        pass  # Loading and byte verification finish before the forward.

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass  # The explicitly seeded snapshot is captured in the model hook.

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None  # Native request ownership and release stay unchanged.


class PairedMeasurement(Measurement):
    """Reuse the qualified Shadow kernel and observe the native request boundary."""

    def __init__(self, runner, instance, warm):
        super().__init__(runner)
        self.instance = instance
        self.generation = 0
        self.snapshot = None
        self.probe_reference = {}
        self.warm = warm
        if warm:
            config = runner.vllm_config
            require(config.use_v2_model_runner and config.cache_config.mamba_cache_mode == 'none'
                    and not config.cache_config.enable_prefix_caching and config.speculative_config is None,
                    'Warm replay requires Model Runner V2, GDN none, APC off and no speculation')
            modules = config.compilation_config.static_forward_context
            by_module = {id(module): name for name, module in modules.items()}
            groups = runner.kv_cache_config.kv_cache_groups
            self.groups = {}
            self.gdn = {}
            for group_id, group in enumerate(groups):
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.first_spec
                for name in group.layer_names:
                    self.groups[name] = group_id
                    if isinstance(spec, MambaSpec):
                        module = modules[name]
                        require(isinstance(module, QwenGatedDeltaNetAttention)
                                and len(module.kv_cache) == 2, 'Unsupported recurrent cache layer')
                        self.gdn[name] = module
                    else:
                        require(type(spec) is FullAttentionSpec and spec.sliding_window is None
                                and spec.attention_chunk_size is None and not spec.non_causal,
                                'Warm replay expects causal full Attention with stable pages')
            self.native_attention = {name: by_module[id(self.modules[name])] for name in self.attention_names}
            require(len(self.gdn) == 48, 'Expected the qualified Qwen GDN layer count')
            connector = get_kv_transfer_group()
            require(isinstance(connector, SnapshotConnector) and connector.instance == instance,
                    'Worker is bound to an unexpected KV connector')
            connector.restore = self.restore
        self.handles.append(runner.model.register_forward_hook(self._after_model))

    def configure_pair(self, spec):
        require(self.active is None, 'A paired request is already active')
        self.generation += 1
        self.binding = spec['binding']
        self.first_position = None
        self.cursor = None
        self.request_id = None
        self.capture = None
        self.restore_seconds = None
        self.forward_events = []
        if spec['history'] == 'capture':
            require(self.warm and self.snapshot is None, 'A previous native snapshot is still retained')
        if spec['history'] == 'restore':
            require(self.snapshot is not None, 'No frozen native bytes available for warm replay')
            original = self.snapshot[0]
            keys = ('snapshot_sha256', 'history_tokens', 'history_ids_sha256')
            require(all(original[key] == self.binding[key] for key in keys), 'Warm replay requested a different prefix')
        if spec['probe'] == 'save':
            self.probe_reference = {}
        expected = spec.get('expected_attention_history')
        self.configure({
            **spec, 'mode': spec['attention_mode'], 'query_end': len(spec['input_ids']),
            'reference': expected is None, 'skip_native_probe': True,
        })
        if expected is not None:
            self.reference_history = expected
        return self.generation

    def current_request(self):
        if self.runner.vllm_config.use_v2_model_runner:
            indices = self.runner.req_states.req_id_to_index
            require(len(indices) == 1, 'Paired GPU runner requires one live native request')
            return next(iter(indices.items()))
        ids = self.runner.input_batch.req_ids
        require(len(ids) == 1, 'Paired GPU runner requires one live native request')
        return ids[0], 0

    def before_model(self, module, args, kwargs):
        super().before_model(module, args, kwargs)
        if self.active is None:
            return
        rid, _ = self.current_request()
        if self.first_position is None:
            restored = self.active['history'] == 'restore'
            start = self.active['query_start'] if restored else 0
            require(self.positions[0] == start, 'First model position disagrees with planned replay boundary')
            require(not restored or (self.capture is not None and self.request_id == rid),
                    'Model reached the Probe before the native prefix was restored')
            self.first_position, self.cursor, self.request_id = start, start, rid
        require(self.request_id == rid and self.positions[0] == self.cursor,
                'Native forward changed owner, skipped positions or recomputed history')
        self.cursor = self.positions[-1] + 1
        require(self.cursor <= self.active['query_end'] or self.positions[0] >= self.active['query_end'],
                'Unexpected combination of prompt and decode tokens in one forward')
        self.forward_start = torch.cuda.Event(enable_timing=True)
        self.forward_start.record()

    def record(self, kind, name, tensor):
        mode = self.active['probe']
        if mode == 'none':
            return
        start, end = self.active['query_start'], self.active['query_end']
        rows = [i for i, position in enumerate(self.positions[:tensor.shape[0]]) if start <= position < end]
        if not rows:
            return
        positions = [self.positions[i] for i in rows]
        key = f'{kind}/{name}'
        reference = self.probe_reference.setdefault(key, {})
        values = tensor[rows].detach().cpu().float()
        for position, value in zip(positions, values):
            probe_index = position - start
            evidence = {'position': position, 'probe_index': probe_index}
            if mode == 'save':
                reference[probe_index] = value.clone()
                evidence['norm'] = float(value.norm())
            else:
                require(probe_index in reference, 'Probe token has no paired baseline activation')
                base = reference[probe_index]
                norm = float(base.norm())
                distance = float((value - base).norm())
                evidence.update(relative_l2=distance / (norm + 1e-8), reference_norm=norm)
            self.metrics.setdefault(key, []).append(evidence)

    def _cache(self, name):
        cache = self.modules[name].kv_cache
        if isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        require(isinstance(cache, torch.Tensor) and cache.ndim == 4 and cache.shape[1] == 4
                and cache.shape[2] == 32 and cache.shape[3] == 512
                and cache.dtype in (torch.uint8, torch.float8_e4m3fn),
                f'Unsupported native Attention cache for {name}')
        return cache.view(torch.uint8)

    def _state_views(self, metadata, request_index, blocks=None):
        require(isinstance(metadata, dict), 'Expected single-lane native GDN metadata')
        result = {}
        for name, module in self.gdn.items():
            meta = metadata[name]
            indices = meta.non_spec_state_indices_tensor
            require(meta.spec_sequence_masks is None and indices is not None and indices.numel() == 1,
                    'Expected exactly one native non-speculative recurrent row')
            row = int(indices.item())
            conv, ssm = module.kv_cache
            require(0 < row < min(conv.shape[0], ssm.shape[0]), 'Recurrent row is outside native state storage')
            if blocks is not None:
                require(row in blocks[self.groups[name]], 'Recurrent row is not owned by the warm request')
                if meta.num_prefills:
                    initial = meta.has_initial_state
                    require(initial is not None and initial.numel() == 1 and bool(initial.item()),
                            'Warm Probe metadata would ignore the restored recurrent state')
            result[name] = (conv[row], ssm[row])
        counter = self.runner.model_state.num_accepted_tokens_gpu[request_index:request_index + 1]
        require(counter.numel() == 1, 'Missing native accepted-token counter')
        if blocks is None:
            require(int(counter.item()) == 1, 'Unexpected captured accepted-token counter')
        return result, counter

    @staticmethod
    def _kv_index(cache, slots):
        index = torch.tensor(slots, device=cache.device, dtype=torch.long)
        return index // cache.shape[2], slice(None), index % cache.shape[2], slice(None)

    def _capture_record(self, attention, states, counter):
        spec = self.active
        return {
            'snapshot_sha256': spec['snapshot_sha256'], 'cache_instance': self.instance,
            'request_id': self.request_id, 'request_generation': self.generation,
            'history_tokens': spec['query_start'], 'kernel_page_tokens': 32,
            'config_sha256': spec['config_sha256'], 'tokenizer_backend_sha256': spec['tokenizer_backend_sha256'],
            'attention': attention,
            'recurrent_state': {
                'layout': 'qwen_gdn', 'boundary_tokens': spec['query_start'],
                'conv_sha256': {name: sha(row[0]) for name, row in states.items()},
                'ssm_sha256': {name: sha(row[1]) for name, row in states.items()},
                'accepted_counter_sha256': sha(counter),
            },
        }

    def _after_model(self, module, args, result):
        spec = self.active
        if spec is None:
            return
        if self.cursor <= spec['query_start']:
            phase = 'history'
        elif self.positions[0] >= spec['query_end']:
            phase = 'decode'
        elif self.positions[0] >= spec['query_start']:
            phase = 'probe'
        else:
            phase = 'mixed_prompt'
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.forward_events.append((phase, self.forward_start, end))
        if spec['history'] == 'capture' and self.cursor >= spec['query_start']:
            self._capture_history()

    def _capture_history(self):
        spec = self.active
        require(self.snapshot is None and self.cursor == spec['query_start'],
                'Snapshot request forwarded past its history boundary')
        rid, index = self.current_request()
        require(rid == self.request_id, 'Snapshot source request changed during the forward')
        metadata = get_forward_context().attn_metadata
        state, counter = self._state_views(metadata, index)
        states = {name: (conv.cpu(), ssm.cpu()) for name, (conv, ssm) in state.items()}
        accepted = counter.cpu()
        attention, images = {}, {}
        for name in self.attention_names:
            cache = self._cache(name)
            seen = self.slots[name]
            require(len(seen) == self.cursor
                    and all(p in seen and 0 < seen[p] < cache.shape[0] * 32 for p in range(self.cursor)),
                    f'Snapshot has an incomplete or invalid history slot map at {name}')
            slots = [seen[p] for p in range(self.cursor)]
            data = cache[self._kv_index(cache, slots)].cpu()
            images[name] = data
            attention[name] = {'slot_mapping': slots, 'cache_page_count': cache.shape[0], 'kv_sha256': sha(data)}
        self.capture = self._capture_record(attention, states, accepted)
        self.snapshot = self.binding, self.capture, images, states, accepted

    def restore(self, load, metadata):
        require(self.active is not None and self.active['history'] == 'restore'
                and self.capture is None and load.binding == self.binding and self.snapshot is not None,
                'Unexpected native snapshot load')
        rid, request_index = self.current_request()
        require(rid == load.request_id and isinstance(metadata, dict), 'Warm load belongs to a different native request')
        self.request_id = rid
        _, seed, images, states, accepted = self.snapshot
        tables = self.runner.block_tables
        n = self.active['query_start']
        blocks = load.block_ids
        require(len(blocks) == len(tables.block_sizes), 'Warm load group count changed')
        for gid, group in enumerate(blocks):
            length = int(tables.num_blocks.np[gid, request_index])
            actual = tables.block_tables[gid].gpu[request_index, :length].cpu().tolist()
            require(actual == list(group), 'Scheduler allocation and worker block table disagree')
        state_views, counter = self._state_views(metadata, request_index, blocks)
        destinations = {}
        for name, native_name in self.native_attention.items():
            cache = self._cache(name)
            group = self.groups[native_name]
            meta = metadata[native_name]
            count = int(meta.num_actual_tokens)
            require(count > 0, 'Warm load has no scheduled Probe tokens')
            slots = allocation_slots(blocks[group], tables.block_sizes[group], n + count, 32, cache.shape[0])
            require(slots[n:] == meta.slot_mapping[:count].cpu().tolist(),
                    'Expanded warm cache slots disagree with the native kernel')
            destinations[name] = cache, slots[:n]
        started = time.perf_counter()
        for name, (cache, slots) in destinations.items():
            cache[self._kv_index(cache, slots)] = images[name].to(cache.device)
        for name, (conv, ssm) in state_views.items():
            conv.copy_(states[name][0])
            ssm.copy_(states[name][1])
        counter.copy_(accepted)
        torch.cuda.synchronize()
        self.restore_seconds = time.perf_counter() - started

        attention = {}
        for name, (cache, slots) in destinations.items():
            data = cache[self._kv_index(cache, slots)]
            value = sha(data)
            require(value == seed['attention'][name]['kv_sha256'], 'Restored Attention KV failed byte verification')
            attention[name] = {'slot_mapping': slots, 'cache_page_count': cache.shape[0], 'kv_sha256': value}
            self.slots[name] = dict(enumerate(slots))
        self.capture = self._capture_record(attention, state_views, counter)
        require(self.capture['recurrent_state'] == seed['recurrent_state'], 'Restored recurrent state failed byte verification')

    def finish_pair(self):
        spec = self.active
        require(self.first_position is not None and self.cursor >= spec['query_end'],
                'Native worker did not forward the complete prompt')
        start = spec['query_start']
        result = self.finish()
        seconds = dict.fromkeys(('history', 'probe', 'mixed_prompt', 'decode'), 0.0)
        for phase, begin, end in self.forward_events:
            seconds[phase] += begin.elapsed_time(end) / 1000
        return {
            'cache_instance': self.instance, 'request_id': self.request_id,
            'first_position': self.first_position,
            'history_tokens_forwarded': start - self.first_position,
            'probe_tokens_forwarded': spec['query_end'] - start,
            'decode_tokens_forwarded': self.cursor - spec['query_end'],
            'restored_history_tokens': start if spec['history'] == 'restore' and self.capture else 0,
            'restore_seconds': self.restore_seconds, 'capture': self.capture, 'cuda_forward_seconds': seconds,
            'historical_attention_sha256': result['historical_kv_sha256'],
            'probe_measurements': result['measurements'], 'native_cache_geometry': result['native_cache_geometry'],
            'attention_calls_after_boundary': result['attention_calls_after_boundary'],
            'peak_allocated_bytes': result['peak_allocated_bytes'], 'peak_reserved_bytes': result['peak_reserved_bytes'],
            'deleted_blocks': spec['deleted_blocks'], 'physically_freed_bytes': 0,
        }


class PairWorker:
    def paired_setup(self, spec, instance, warm):
        if not hasattr(self, '_paired_measurement'):
            self._paired_measurement = PairedMeasurement(self.model_runner, instance, warm)
        return self._paired_measurement.configure_pair(spec)

    def paired_finish(self):
        return self._paired_measurement.finish_pair()

    def paired_release(self):
        if hasattr(self, '_paired_measurement'):
            self._paired_measurement.snapshot = None
            self._paired_measurement.probe_reference = {}


def _duration(metrics, first, second):
    a, b = getattr(metrics, first, None), getattr(metrics, second, None)
    return b - a if a and b and math.isfinite(a) and math.isfinite(b) and b >= a else None


class NativeBackend:
    """Run one request at a time on the same model and tokenizer artifacts."""

    def __init__(self, model, tokenizer, cases, execution):
        config = cases[0][0]['config']
        require(vllm.__version__ == config['vllm_version'], 'Installed vLLM differs from the sealed runtime version')
        require((config['attention_layers'], config['recurrent_layers'], config['kernel_page_tokens']) == (16, 48, 32),
                'Expected 16 Attention layers, 48 GDN layers and 32-token kernel pages')
        model_config = Path(model) / 'config.json'
        hf = json.loads(model_config.read_text(encoding='utf-8'))
        require(bool({'Qwen3_5ForConditionalGeneration', 'Qwen3_5ForCausalLM'} & set(hf.get('architectures', []))),
                'Expected the qualified Qwen3.5-family hybrid model')
        self.tokenizer, tokenizer_evidence = load_tokenizer(tokenizer)
        for snapshot, _ in cases:
            for key, value in snapshot['tokenizer_evidence'].items():
                require(tokenizer_evidence.get(key) == value, f'Runtime tokenizer evidence differs for {key}')
            ids, offsets = encode(self.tokenizer, snapshot['prompt'])
            require(ids == snapshot['input_ids'] and offsets == snapshot['offsets'],
                    'Runtime tokenizer did not reproduce the frozen history and Probe')
        self.sampling = config['inference_params']
        self.instance = uuid4().hex
        self.warm = execution == 'warm'
        attention = self.sampling.get('attention_config', {'backend': 'FLASHINFER'})
        require(attention.get('backend') == 'FLASHINFER'
                and self.sampling.get('kv_cache_dtype', 'fp8_e4m3') == 'fp8_e4m3'
                and self.sampling.get('enforce_eager', True) is True
                and self.sampling.get('max_num_seqs', 1) == 1
                and self.sampling.get('language_model_only', True) is True,
                'Expected the isolated eager FP8 FlashInfer profile')
        args = {
            'model': str(model.resolve()), 'tokenizer': str(tokenizer.resolve()),
            'dtype': self.sampling.get('dtype', 'bfloat16'),
            'quantization': self.sampling.get('quantization', 'gptq_marlin'),
            'max_model_len': self.sampling.get('max_model_len', 34816),
            'max_num_batched_tokens': self.sampling.get('max_num_batched_tokens', 512),
            'gpu_memory_utilization': self.sampling.get('gpu_memory_utilization', .88),
            'max_num_seqs': 1, 'seed': self.sampling['seed'], 'language_model_only': True,
            'attention_config': attention, 'kv_cache_dtype': 'fp8_e4m3',
            'mamba_cache_mode': 'none', 'enable_prefix_caching': False, 'enforce_eager': True,
            'async_scheduling': False, 'disable_log_stats': False,
            'worker_extension_cls': f'{__name__}.PairWorker',
        }
        require(all(max(len(snapshot['input_ids']), len(plan['text']['input_ids']))
                    + self.sampling['max_tokens'] <= args['max_model_len'] for snapshot, plan in cases),
                'A paired request exceeds the configured context budget')
        if self.warm:
            args['kv_transfer_config'] = {
                'kv_connector': 'SnapshotConnector', 'kv_role': 'kv_consumer',
                'kv_connector_module_path': __name__, 'kv_connector_extra_config': {'instance': self.instance},
            }
        self.schema = {
            'type': 'object', 'properties': {k: {'type': 'string'} for k in PROBES},
            'required': list(PROBES), 'additionalProperties': False,
        }
        self.environment = {
            'engine_args': args, 'cache_instance': self.instance, 'vllm': vllm.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'model_config_sha256': file_hash(model_config), 'tokenizer_evidence': tokenizer_evidence,
            'timing_scope': 'Isolated instrumented requests with layer and cache evidence collection.',
        }
        self.llm = LLM(**args)

    def retokenize(self, snapshot, character_span):
        return encode_variant(self.tokenizer, snapshot, character_span)

    def run(self, spec):
        ids = spec['input_ids']
        binding = {
            'snapshot_sha256': spec['snapshot_sha256'], 'cache_instance': self.instance,
            'history_tokens': spec['query_start'], 'prompt_ids_sha256': fingerprint(ids),
            'history_ids_sha256': fingerprint(ids[:spec['query_start']]),
        }
        self.llm.collective_rpc('paired_setup', kwargs={
            'spec': dict(spec, binding=binding), 'instance': self.instance, 'warm': self.warm,
        })
        params = SamplingParams(
            temperature=self.sampling['temperature'], seed=self.sampling['seed'], max_tokens=spec['max_tokens'],
            extra_args={'kv_transfer_params': {BINDING_KEY: binding}} if spec['history'] == 'restore' else None,
            structured_outputs=StructuredOutputsParams(json=self.schema) if spec['structured'] else None,
        )
        started = time.perf_counter()
        output = self.llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        worker = self.llm.collective_rpc('paired_finish')
        require(len(worker) == 1 and len(output.outputs) == 1, 'Expected one isolated request and worker')
        answer = output.outputs[0]
        ttft = getattr(output.metrics, 'first_token_latency', None)
        return {
            'arm': spec['arm'], 'snapshot_sha256': spec['snapshot_sha256'], 'plan_sha256': spec['plan_sha256'],
            'request_id': output.request_id, 'prompt_ids_sha256': fingerprint(output.prompt_token_ids),
            'prompt_tokens': len(output.prompt_token_ids), 'num_cached_tokens': output.num_cached_tokens,
            'token_ids': list(answer.token_ids), 'text': answer.text, 'finish_reason': answer.finish_reason,
            'wall_seconds': elapsed,
            'ttft_seconds': ttft if ttft and math.isfinite(ttft) and ttft > 0 else None,
            'decode_seconds': _duration(output.metrics, 'first_token_ts', 'last_token_ts'),
            'worker': worker[0],
        }

    def release_snapshot(self):
        if self.warm:
            self.llm.collective_rpc('paired_release')

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
