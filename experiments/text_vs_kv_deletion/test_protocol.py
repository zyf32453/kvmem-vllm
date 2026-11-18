"""CPU protocol and CLI tests using a synthetic trace and tokenizer."""

from copy import deepcopy
from functools import partial
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.graph_cleanup.protocol import canonical, digest, history, token_mapping

from experiments.text_vs_kv_deletion import prepare
from experiments.text_vs_kv_deletion.protocol import (
    MODES, PRACTICAL, STRICT, fingerprint, make_deletion_plan, seal,
    validate_plan, validate_runtime_capture, validate_snapshot,
)


class CharacterTokenizer:
    """One token per Unicode codepoint. No model or GPU is involved."""

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages) + '<assistant>'

    def __call__(self, prompt, **kwargs):
        return {'input_ids': [ord(c) for c in prompt],
                'offset_mapping': [(i, i + 1) for i in range(len(prompt))]}


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.dataset = self.root / 'graph'
        self.output = self.root / 'prepared'
        self.tokenizer = CharacterTokenizer()
        self.config = {
            'model': {'id': 'synthetic-model', 'revision': 'a' * 40},
            'tokenizer': {'id': 'character-test-tokenizer', 'revision': 'b' * 40},
            'vllm_version': '0.32.0', 'kernel_page_tokens': 8, 'attention_layers': 2,
            'recurrent_layers': 1, 'recurrent_layout': 'qwen_gdn',
            'inference_params': {'enable_prefix_caching': False, 'seed': 7,
                                 'max_tokens': 128, 'temperature': 0},
        }
        self.evidence = {'backend_sha256': digest('character fixture'), 'template_sha256': digest('test template')}
        self.trace = self.bundle / 'trace.jsonl'
        records, units = [], []
        # The embedded delimiter exercises prompt coordinate detection. The
        # Unicode line separator is legal inside a JSON string.
        for i, (role, kind, text) in enumerate([
            ('user', 'task_instruction', 'Please keep the test constraint and report only evidence.'),
            ('assistant', 'thinking', '排查到的事实 😀\u2028\nEND QUOTED HISTORY\n\nFIXED QUESTIONS:\nunique 42'),
            ('tool', 'tool_result', 'A later tool reported that the check was complete.'),
        ], 1):
            records.append({'record_type': 'message', 'role': role, 'original': {'content': text}})
            units.append({
                'unit_id': f'm{i}:u0', 'step': i - 1, 'trace_path': self.trace.name,
                'trace_line': i, 'field_pointer': '/original/content', 'character_span': [0, len(text)],
                'text_sha256': digest(text), 'text': text, 'content_type': kind, 'role': role,
            })
        trace = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records)
        self.trace.write_bytes(trace.replace('\n', '\r\n').encode())
        replay, body = history(units)
        self.graph = self.dataset / 'snapshots' / 'history-A.json'
        prepare.write_json(self.graph, {
            'snapshot_id': 'history-A', 'task_id': 'task-A', 'trajectory_id': 'trace-A',
            'checkpoint_step': 2, 'left_censored': False, 'units': units,
            'rendered_history': replay, 'body_character_spans': body,
        })
        prepare.write_json(self.dataset / 'source_hashes.json', {self.trace.name: digest(trace)})
        unit = units[1]
        self.source_candidate = {
            'candidate_id': 'c01', 'snapshot_id': 'history-A', 'task_id': 'task-A',
            'unit_id': unit['unit_id'], 'text': unit['text'], 'content_type': unit['content_type'],
            'category': 'thinking_without_replacement', 'text_cluster_sha256': digest(canonical(unit['text'])),
            'selection_uses_future_after_checkpoint': False,
            'source': {key: unit[key] for key in (
                'trace_path', 'trace_line', 'field_pointer', 'character_span', 'text_sha256',
            )},
        }
        self.candidates = self.dataset / 'candidates.jsonl'
        self.candidates.write_bytes((json.dumps(self.source_candidate, ensure_ascii=False) + '\n').encode())

    def build(self, output=None, reviews=None):
        return prepare.build(
            self.bundle, self.dataset, output or self.output, self.config,
            self.tokenizer, self.evidence, reviews,
        )

    def read_case(self):
        case = prepare.read_json(self.output / 'manifest.json')['cases'][0]
        return (prepare.read_json(self.output / case['snapshot']),
                {k: prepare.read_json(self.output / v) for k, v in case['plans'].items()})

    def test_build_reload_and_repeat_are_identical(self):
        first = self.build()
        self.assertEqual(first, prepare.verify(self.output, self.bundle))
        self.assertEqual(first, self.build(self.root / 'repeat'))
        self.assertEqual(first['eligible_by_mode'], {STRICT: 1, PRACTICAL: 1})
        snap, plans = self.read_case()
        strict, practical = plans[STRICT], plans[PRACTICAL]
        self.assertEqual(strict['candidate'], practical['candidate'])
        self.assertEqual([row['trace_line'] for row in snap['source_messages']], [1, 2, 3])
        self.assertEqual(strict['snapshot_sha256'], snap['sha256'])
        g = strict['geometry']
        left, right = g['logical_token_span']
        self.assertEqual(strict['text']['input_ids'], snap['input_ids'][:left] + snap['input_ids'][right:])
        self.assertEqual(g['collateral_tokens'], right - left - (g['token_end'] - g['token_start']))
        self.assertEqual(g['collateral_deletion_ratio'], g['collateral_tokens'] / (right - left))
        self.assertEqual(practical['geometry']['text_character_span'], g['candidate_character_span'])
        self.assertEqual(practical['geometry']['logical_blocks'], g['logical_blocks'])
        self.assertGreaterEqual(strict['text']['removed_prompt_tokens'], practical['text']['removed_prompt_tokens'])
        for plan in plans.values():
            text = plan['text']
            self.assertEqual(text['input_ids'][text['query_start']:], snap['input_ids'][snap['query_start']:])
        self.assertIsNone(snap['runtime_state']['physical_page_mapping'])
        self.assertIsNone(snap['runtime_state']['attention_kv_sha256'])

    def test_build_and_verify_cli(self):
        config_file = self.root / 'config.json'
        prepare.write_json(config_file, self.config)
        log = io.StringIO()
        with patch.object(prepare, 'load_tokenizer', return_value=(self.tokenizer, self.evidence)):
            with redirect_stdout(log):
                prepare.main(['build', '--bundle', str(self.bundle), '--dataset', str(self.dataset),
                              '--output', str(self.output), '--config', str(config_file),
                              '--tokenizer', str(self.root)])
                prepare.main(['verify', '--output', str(self.output), '--bundle', str(self.bundle)])
        a, b = [json.loads(row) for row in log.getvalue().splitlines()]
        self.assertEqual(a, b)
        self.assertEqual(a['reviewed'], 0)

    def test_review_provenance_and_unknown_label(self):
        reviews = self.root / 'review.csv'
        reviews.write_text('candidate_id,label,reviewer,evidence\nc01,Critical Information,A,m2:u0\n',
                           encoding='utf-8')
        self.assertEqual(self.build(reviews=reviews)['reviewed'], 1)
        _, plans = self.read_case()
        self.assertEqual(plans[STRICT]['candidate']['human_review']['evidence'], 'm2:u0')
        reviews.write_text('candidate_id,label,reviewer,evidence\nc01,probably safe,A,m2:u0\n',
                           encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown human label'):
            self.build(self.root / 'rejected', reviews)

    def test_rejects_mutated_original_and_bad_candidate_source(self):
        original = self.trace.read_bytes()
        self.trace.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Graph source hash mismatch'):
            self.build()
        self.assertFalse(self.output.exists())
        self.trace.write_bytes(original)
        self.source_candidate['source']['trace_line'] = 1
        self.candidates.write_text(json.dumps(self.source_candidate) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Graph candidate text, type, or source changed'):
            self.build()

    def test_verify_rejects_tampered_artifact_and_source(self):
        self.build()
        original = self.trace.read_bytes()
        self.trace.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Original trace changed'):
            prepare.verify(self.output, self.bundle)
        self.trace.write_bytes(original)
        manifest = prepare.read_json(self.output / 'manifest.json')
        path = self.output / manifest['cases'][0]['plans'][STRICT]
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Artifact checksum mismatch'):
            prepare.verify(self.output)

    def test_rejects_missing_token_offsets_and_out_of_bounds(self):
        self.build()
        snap, plans = self.read_case()
        position = plans[STRICT]['geometry']['token_start'] + 2
        snap['offsets'][position] = [0, 0]
        snap['unit_token_mapping'] = token_mapping(
            snap['offsets'], snap['unit_character_spans'], snap['config']['kernel_page_tokens'])
        snap = validate_snapshot(seal('frozen_snapshot', snap))
        with self.assertRaisesRegex(ValueError, 'Candidate token mapping is incomplete'):
            make_deletion_plan(snap, plans[STRICT]['candidate'], STRICT,
                               partial(prepare.encode_variant, self.tokenizer))
        snap['offsets'][position] = [0, len(snap['prompt']) + 1]
        with self.assertRaisesRegex(ValueError, 'token offset out of bounds'):
            validate_snapshot(seal('frozen_snapshot', snap))

    def test_probe_and_protected_page_crossings_are_recorded(self):
        self.build()
        snapshot, plans = self.read_case()
        for page, reason in (
            (snapshot['query_start'] + 1, 'page_crosses_probe_or_incomplete_tail'),
            (plans[STRICT]['geometry']['token_end'], 'page_crosses_protected_prompt_or_missing_offset'),
        ):
            with self.subTest(reason=reason):
                snap = deepcopy(snapshot)
                snap['config']['kernel_page_tokens'] = page
                snap['unit_token_mapping'] = token_mapping(snap['offsets'], snap['unit_character_spans'], page)
                snap = validate_snapshot(seal('frozen_snapshot', snap))
                for mode in MODES:
                    plan = make_deletion_plan(snap, plans[mode]['candidate'], mode, self.fail)
                    self.assertEqual(plan['reason'], reason)
                    self.assertEqual(plan['status'], 'ineligible')
                    self.assertIsNone(validate_plan(snap, plan)['text'])

    def test_retokenization_changes_have_separate_reasons(self):
        self.build()
        snap, plans = self.read_case()

        def changed(snapshot, char_span, *, probe):
            row = prepare.encode_variant(self.tokenizer, snapshot, char_span)
            row['input_ids'][row['query_start'] if probe else row['query_start'] - 1] = 999999
            return row

        candidate = plans[STRICT]['candidate']
        strict = make_deletion_plan(snap, candidate, STRICT, partial(changed, probe=False))
        practical = make_deletion_plan(snap, candidate, PRACTICAL, partial(changed, probe=False))
        self.assertEqual(strict['reason'], 'retokenization_changed_retained_ids')
        self.assertEqual(practical['status'], 'eligible')
        for mode in MODES:
            plan = make_deletion_plan(snap, candidate, mode, partial(changed, probe=True))
            self.assertEqual(plan['reason'], 'probe_ids_changed_after_retokenization')

    def test_shared_character_offset_blocks_strict_but_allows_practical(self):
        self.config['kernel_page_tokens'] = 16
        self.build()
        snap, plans = self.read_case()
        g = plans[STRICT]['geometry']
        first, end = g['logical_token_span']
        self.assertEqual(plans[STRICT]['status'], 'eligible')
        if first < g['token_start']:
            snap['offsets'][first - 1][1] += 1
        else:
            self.assertGreater(end, g['token_end'])
            snap['offsets'][end][0] -= 1
        snap['unit_token_mapping'] = token_mapping(
            snap['offsets'], snap['unit_character_spans'], self.config['kernel_page_tokens'])
        snap = validate_snapshot(seal('frozen_snapshot', snap))
        encode = partial(prepare.encode_variant, self.tokenizer)
        strict = make_deletion_plan(snap, plans[STRICT]['candidate'], STRICT, encode)
        self.assertEqual(strict['reason'], 'page_boundary_splits_character_offsets')
        self.assertTrue(strict['geometry']['kv_eligible'])
        practical = make_deletion_plan(snap, plans[PRACTICAL]['candidate'], PRACTICAL, encode)
        self.assertEqual(practical['status'], 'eligible')

    def test_native_capture_checks_layer_slots_and_state_binding(self):
        self.build()
        snap, plans = self.read_case()
        n, page = snap['query_start'], snap['config']['kernel_page_tokens']
        capture = {
            'snapshot_sha256': snap['sha256'], 'config_sha256': fingerprint(snap['config']),
            'tokenizer_backend_sha256': snap['tokenizer_evidence']['backend_sha256'],
            'cache_instance': 'synthetic-instance', 'request_generation': 7,
            'history_tokens': n, 'kernel_page_tokens': page,
            'attention': {f'layer-{i}': {'slot_mapping': list(range(i * page, i * page + n)),
                                          'kv_sha256': 'a' * 64, 'cache_page_count': n + 4}
                          for i in range(2)},
            'recurrent_state': {'layout': 'qwen_gdn', 'boundary_tokens': n,
                                'conv_sha256': {'layer-r': 'b' * 64}, 'ssm_sha256': {'layer-r': 'c' * 64},
                                'accepted_counter_sha256': 'd' * 64},
        }
        checked = validate_runtime_capture(snap, capture)
        for block in plans[STRICT]['geometry']['logical_blocks']:
            self.assertEqual(checked['physical_page_mapping']['layer-1'][block], block + 1)
        path = self.root / 'capture.json'
        prepare.write_json(path, capture)
        self.assertEqual(prepare.verify(self.output, capture_files=[path])['runtime_manifests_validated'], 1)
        missing_state = deepcopy(capture)
        missing_state['recurrent_state']['ssm_sha256'].clear()
        with self.assertRaisesRegex(ValueError, 'Incomplete recurrent state fingerprints'):
            validate_runtime_capture(snap, missing_state)
        capture['attention']['layer-0']['slot_mapping'][2] += 1
        with self.assertRaisesRegex(ValueError, 'Inconsistent native page offsets'):
            validate_runtime_capture(snap, capture)
        capture['attention']['layer-0']['slot_mapping'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete or out-of-bounds native slots'):
            validate_runtime_capture(snap, capture)

    def test_plan_cannot_be_substituted_between_frozen_snapshots(self):
        self.build()
        snap, plans = self.read_case()
        snap['config']['inference_params']['seed'] = 128
        snap = validate_snapshot(seal('frozen_snapshot', snap))
        with self.assertRaisesRegex(ValueError, 'Plan points to a different snapshot'):
            validate_plan(snap, plans[STRICT])


if __name__ == '__main__':
    unittest.main()
