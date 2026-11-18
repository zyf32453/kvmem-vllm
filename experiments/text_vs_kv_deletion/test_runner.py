"""CPU checks of Phase 2 flow and evidence rejection with a synthetic engine."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest

from experiments.text_vs_kv_deletion import test_protocol as phase1
from experiments.text_vs_kv_deletion.prepare import encode_variant, read_json, verify
from experiments.text_vs_kv_deletion.protocol import STRICT, allocation_slots, digest, fingerprint
from experiments.text_vs_kv_deletion.runner import main, run_pair


class SyntheticEngine:
    def __init__(self, snapshot, tokenizer, mutate=None):
        self.snapshot = snapshot
        self.tokenizer = tokenizer
        self.mutate = mutate
        self.calls = []
        self.released = False

    def retokenize(self, snapshot, span):
        return encode_variant(self.tokenizer, snapshot, span)

    def run(self, spec):
        self.calls.append(deepcopy(spec))
        generation = len(self.calls)
        external = str(generation)
        internal = f'{external}-eeff0022'
        history, arm = spec['history'], spec['arm']
        ids, boundary = spec['input_ids'], spec['query_start']
        config = self.snapshot['config']
        page = config['kernel_page_tokens']
        layers = [f'attention-{i}' for i in range(config['attention_layers'])]
        mapped_slots = [5000 * page + generation * page + i for i in range(boundary)]
        hashes = {name: digest(name + ('-text' if arm == 'G1' else '-original')) for name in layers}
        capture = None
        if history in ('capture', 'restore'):
            capture = {
                'snapshot_sha256': spec['snapshot_sha256'], 'cache_instance': 'test',
                'request_generation': generation, 'request_id': internal,
                'history_tokens': boundary, 'kernel_page_tokens': page,
                'config_sha256': spec['config_sha256'], 'tokenizer_backend_sha256': spec['tokenizer_backend_sha256'],
                'attention': {name: {'slot_mapping': mapped_slots, 'cache_page_count': 10000,
                                     'kv_sha256': hashes[name]} for name in layers},
                'recurrent_state': {'layout': 'qwen_gdn', 'boundary_tokens': boundary,
                                    'conv_sha256': {'state': digest('conv')}, 'ssm_sha256': {'state': digest('ssm')},
                                    'accepted_counter_sha256': digest('counter')},
            }
        measurement = {}
        if spec['probe'] != 'none':
            layer_groups = {
                'attention_pre_gate_per_head': len(layers),
                'transformer_residual_sum': len(layers) + config['recurrent_layers'],
            }
            for kind, count in layer_groups.items():
                for i in range(count):
                    measurement[f'{kind}/{i}'] = [
                        {'probe_index': j, 'position': boundary + j, 'relative_l2': 0.0}
                        for j in range(len(ids) - boundary)
                    ]
        positions = [i for block in spec['deleted_blocks'] for i in range(block * page, (block + 1) * page)]
        restored = history == 'restore'
        result = {
            'request_id': external, 'prompt_ids_sha256': fingerprint(ids), 'prompt_tokens': len(ids),
            'snapshot_sha256': spec['snapshot_sha256'], 'plan_sha256': spec['plan_sha256'],
            'num_cached_tokens': boundary if restored else 0, 'token_ids': [2] * spec['max_tokens'],
            'wall_seconds': 0.2, 'ttft_seconds': 0.1, 'decode_seconds': 0.1,
            'worker': {
                'cache_instance': 'test', 'request_id': internal, 'capture': capture,
                'first_position': boundary if restored else 0,
                'history_tokens_forwarded': 0 if restored else boundary,
                'probe_tokens_forwarded': len(ids) - boundary,
                'restored_history_tokens': boundary if restored else 0,
                'restore_seconds': 0.01 if restored else None,
                'physically_freed_bytes': 0, 'deleted_blocks': spec['deleted_blocks'],
                'cuda_forward_seconds': {'history': 0.1 if not restored else 0.0},
                'historical_attention_sha256': hashes, 'probe_measurements': measurement,
                'native_cache_geometry': {name: {
                    'shape': [10000, 4, page, 512], 'history_tokens': boundary, 'native_page_tokens': page,
                    'candidate_physical_slots': [mapped_slots[p] for p in positions],
                } for name in layers},
            },
        }
        if self.mutate:
            self.mutate(spec, result)
        return result

    def release_snapshot(self):
        self.released = True


class PairedRunnerTest(unittest.TestCase):
    def setUp(self):
        fixture = phase1.ProtocolTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build()
        self.snapshot, plans = fixture.read_case()
        self.plan = plans[STRICT]
        self.prepared = fixture.output
        self.out = fixture.root / 'paired'
        self.tokenizer = fixture.tokenizer

    def execute_pair(self, execution, mutate=None, repeat=0):
        self.engine = SyntheticEngine(self.snapshot, self.tokenizer, mutate)
        return run_pair(self.engine, self.snapshot, self.plan, execution, self.out, repeat)

    def test_warm_path_runs_zero_controls_and_uses_one_plan(self):
        result = self.execute_pair('warm')
        self.assertTrue(read_json(self.out / 'zero-deletion.json')['passed'])
        self.assertEqual([spec['arm'] for spec in self.engine.calls],
                         ['seed', 'zero-native', 'zero-restored-native', 'zero-shadow', 'G0', 'G1', 'G2'])
        self.assertTrue(all(spec['plan_sha256'] == self.plan['sha256'] for spec in self.engine.calls))
        g0, g1, g2 = self.engine.calls[-3:]
        self.assertEqual(g2['input_ids'], g0['input_ids'])
        self.assertEqual(g1['input_ids'], self.plan['text']['input_ids'])
        self.assertEqual(g2['deleted_blocks'], self.plan['geometry']['logical_blocks'])
        self.assertEqual(result['arms']['G2']['history_tokens_forwarded'], 0)
        self.assertGreater(result['arms']['G1']['history_tokens_forwarded'], 0)
        self.assertEqual(result['arms']['G2']['physically_freed_bytes'], 0)
        self.assertTrue(self.engine.released)
        audited = verify(self.prepared, capture_files=[self.out / 'seed.json', self.out / 'G2.json'])
        self.assertEqual(audited['runtime_manifests_validated'], 2)

    def test_replay_reports_independent_history_compute(self):
        result = self.execute_pair('replay', repeat=1)
        self.assertEqual([spec['arm'] for spec in self.engine.calls],
                         ['zero-native', 'zero-shadow', 'G0', 'G2', 'G1'])
        g2 = result['arms']['G2']
        self.assertEqual(g2['history_tokens_forwarded'], self.snapshot['query_start'])
        self.assertEqual(g2['restored_history_tokens'], 0)

    def test_failed_zero_control_never_runs_a_deletion(self):
        def break_zero(spec, result):
            if spec['arm'] == 'zero-shadow':
                first = next(iter(result['worker']['probe_measurements'].values()))
                first[0]['relative_l2'] = 0.05

        with self.assertRaisesRegex(ValueError, 'Zero-deletion control failed'):
            self.execute_pair('replay', break_zero)
        self.assertFalse(read_json(self.out / 'zero-deletion.json')['passed'])
        self.assertEqual(len(self.engine.calls), 2)
        self.assertTrue(self.engine.released)

    def test_warm_forward_rejects_recomputed_history(self):
        def recompute(spec, result):
            if spec['arm'] == 'zero-restored-native':
                result['worker']['first_position'] = 0

        with self.assertRaisesRegex(ValueError, 'Observed model positions'):
            self.execute_pair('warm', recompute)
        self.assertTrue(self.engine.released)

    def test_warm_forward_rejects_changed_recurrent_state(self):
        def change_state(spec, result):
            if spec['arm'] == 'zero-restored-native':
                result['worker']['capture']['recurrent_state']['ssm_sha256']['state'] = digest('different')

        with self.assertRaisesRegex(ValueError, 'Restored KV or recurrent state differs'):
            self.execute_pair('warm', change_state)

    def test_shadow_slots_must_match_the_warm_allocation(self):
        def change_page(spec, result):
            if spec['arm'] == 'G2':
                first = next(iter(result['worker']['native_cache_geometry'].values()))
                first['candidate_physical_slots'] = [s - 32 * 8 for s in first['candidate_physical_slots']]

        with self.assertRaisesRegex(ValueError, 'Shadow excluded different pages'):
            self.execute_pair('warm', change_page)

    def test_replay_rejects_a_cache_hit_in_text_arm(self):
        def fake_hit(spec, result):
            if spec['arm'] == 'G1':
                result['num_cached_tokens'] = 32

        with self.assertRaisesRegex(ValueError, 'Unexpected native cache hit'):
            self.execute_pair('replay', fake_hit)

    def test_plan_only_loads_verified_phase1_files(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(['--prepared', str(self.prepared), '--candidate', 'c01', '--plan-only', '--allow-unreviewed'])
        case = json.loads(stdout.getvalue())['cases'][0]
        self.assertEqual(case['plan_sha256'], self.plan['sha256'])
        self.assertFalse(self.out.exists())

    def test_allocator_expands_native_blocks_to_kernel_slots(self):
        slots = allocation_slots([4, 9], 64, 70, 32, 24)
        self.assertEqual(slots, list(range(256, 320)) + list(range(576, 582)))
        for blocks in ([4], [4, 4], [0, 9], [4, 24]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                allocation_slots(blocks, 64, 70, 32, 24)


if __name__ == '__main__':
    unittest.main()
