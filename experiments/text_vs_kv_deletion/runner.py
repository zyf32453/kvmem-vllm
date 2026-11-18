"""Run paired G0/G1/G2 in an isolated vLLM engine from verified Phase 1 artifacts."""

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

from .prepare import read_prepared, write_json
from .protocol import MODES, STRICT, digest, fingerprint, require, sha256_hex, validate_runtime_capture

CONTROL_L2 = 0.03


def select_cases(prepared, mode, candidate_id, limit, allow_unreviewed):
    cases = []
    for snapshot, plans in prepared:
        plan = plans[mode]
        candidate = plan['candidate']
        if candidate_id is not None and candidate['candidate_id'] != candidate_id:
            continue
        if plan['status'] != 'eligible':
            require(candidate_id is None, f"Candidate is ineligible: {plan['reason']}")
            continue
        if not allow_unreviewed and candidate['human_review'] is None:
            require(candidate_id is None, 'Candidate requires human review')
            continue
        cases.append((snapshot, plan))
    require(cases, 'No eligible cases for this mode and review filter')
    return cases[:limit]


def arm_spec(snapshot, plan, arm, *, history, attention, probe, control=False, text=None):
    ids = snapshot['input_ids'] if text is None else text['input_ids']
    boundary = snapshot['query_start'] if text is None else text['query_start']
    if history == 'capture':
        ids = ids[:boundary]
    limit = snapshot['config']['inference_params']['max_tokens']
    return {
        'arm': arm, 'history': history, 'attention_mode': attention, 'probe': probe,
        'input_ids': ids, 'query_start': boundary,
        'deleted_blocks': plan['geometry']['logical_blocks'] if arm == 'G2' else [],
        'max_tokens': 1 if history == 'capture' else min(2, limit) if control else limit,
        'structured': not control and history != 'capture',
        'snapshot_sha256': snapshot['sha256'], 'plan_sha256': plan['sha256'],
        'config_sha256': fingerprint(snapshot['config']),
        'tokenizer_backend_sha256': snapshot['tokenizer_evidence']['backend_sha256'],
    }


def _same_state(snapshot, seed, restored):
    expected = validate_runtime_capture(snapshot, seed)
    actual = validate_runtime_capture(snapshot, restored)
    require(seed['cache_instance'] == restored['cache_instance']
            and seed['request_generation'] != restored['request_generation']
            and seed['request_id'] != restored['request_id'],
            'Restore must use a fresh request in the seed engine')
    require(actual['attention_kv_sha256'] == expected['attention_kv_sha256']
            and actual['recurrent_state'] == expected['recurrent_state'],
            'Restored KV or recurrent state differs from the captured bytes')


def check_arm(snapshot, spec, result, seed=None):
    """Fail closed on cache hits, changed prompts, or incomplete Probe evidence."""
    ids = spec['input_ids']
    boundary = spec['query_start']
    require(result['prompt_ids_sha256'] == fingerprint(ids), 'Engine prompt IDs changed')
    require(result['snapshot_sha256'] == spec['snapshot_sha256'] and result['plan_sha256'] == spec['plan_sha256'],
            'Engine reply was associated with a different snapshot or plan')
    require(bool(result['token_ids']), 'Engine returned no sampled token')
    report = result['worker']
    external = re.escape(result['request_id'])
    # vLLM V1 may append eight random hex digits to the internal request ID.
    require(report['cache_instance'] and re.fullmatch(external + r'(?:-[0-9a-f]{8})?', report['request_id']),
            'Worker evidence belongs to a different request')
    history = spec['history']
    restored = history == 'restore'
    start = boundary if restored else 0
    history_tokens = 0 if restored else boundary
    require(report['first_position'] == start and report['history_tokens_forwarded'] == history_tokens
            and report['probe_tokens_forwarded'] == len(ids) - boundary,
            'Observed model positions disagree with the requested history and Probe')
    cached = result['num_cached_tokens']
    require(cached is None or cached == (boundary if restored else 0), 'Unexpected native cache hit')
    require(report['physically_freed_bytes'] == 0, 'Shadow evidence reports physical reclamation')
    require(report['deleted_blocks'] == spec['deleted_blocks'], 'Observed deletion plan changed')
    capture = report['capture']
    if history == 'capture' or restored:
        require(capture is not None and capture['request_id'] == report['request_id']
                and capture['cache_instance'] == report['cache_instance'], 'Missing native capture evidence')
        validate_runtime_capture(snapshot, capture)
    if restored:
        require(seed is not None and report['restored_history_tokens'] == boundary,
                'Warm request did not restore the complete history')
        _same_state(snapshot, seed, capture)
    else:
        require(report['restored_history_tokens'] == 0, 'Cold request restored historical tokens')
    if history == 'capture':
        return None

    hashes = report['historical_attention_sha256']
    require(len(hashes) == snapshot['config']['attention_layers']
            and all(sha256_hex(value) for value in hashes.values()), 'Incomplete historical Attention KV hashes')
    if restored:
        require(hashes == {name: layer['kv_sha256'] for name, layer in capture['attention'].items()},
                'Attention consumed a history different from the restored state')
    page = snapshot['config']['kernel_page_tokens']
    geometry = report['native_cache_geometry']
    require(geometry.keys() == hashes.keys(), 'Missing native Attention cache geometry')
    positions = [p for block in spec['deleted_blocks'] for p in range(block * page, (block + 1) * page)]
    require(all(p < boundary for p in positions), 'Shadow plan crosses the Probe')
    for name, layer in geometry.items():
        slots = layer['candidate_physical_slots']
        require(layer['native_page_tokens'] == page and layer['history_tokens'] == boundary
                and len(slots) == len(positions) and len(set(slots)) == len(slots)
                and all(0 < slot < layer['shape'][0] * page and slot % page == pos % page
                        for pos, slot in zip(positions, slots)), 'Invalid physical mapping for Shadow pages')
        if restored:
            require(slots == [capture['attention'][name]['slot_mapping'][p] for p in positions],
                    'Shadow excluded different pages than the verified warm page table')
    # The pilot has one Transformer hook per Attention or GDN layer.
    expected_layers = {
        'attention_pre_gate_per_head': snapshot['config']['attention_layers'],
        'transformer_residual_sum': snapshot['config']['attention_layers'] + snapshot['config']['recurrent_layers'],
    }
    evidence = report['probe_measurements']
    count = len(ids) - boundary
    expected_indices = list(range(count))
    max_l2 = {}
    for kind, expected in expected_layers.items():
        groups = [rows for name, rows in evidence.items() if name.startswith(kind + '/')]
        require(len(groups) == expected, f'Incomplete Probe layer coverage for {kind}')
        values = []
        for rows in groups:
            require([row['probe_index'] for row in rows] == expected_indices
                    and [row['position'] for row in rows] == list(range(boundary, len(ids))),
                    f'Missing or shifted Probe token observations for {kind}')
            if spec['probe'] == 'compare':
                for row in rows:
                    value = row['relative_l2']
                    require(math.isfinite(value) and value >= 0, 'Invalid Probe relative L2')
                    values.append(value)
        if values:
            max_l2[kind] = max(values)
    require(len(evidence) == sum(expected_layers.values()), 'Unexpected Probe measurement keys')
    return max_l2


def run_pair(backend, snapshot, plan, execution, output, repeat):
    """Gate each pair on a native zero-deletion oracle using the same Probe."""
    seed = None
    restore = 'restore' if execution == 'warm' else 'compute'
    results = {}

    def run(spec, name):
        result = backend.run(spec)
        write_json(output / f'{name}.json', result)
        result['max_probe_relative_l2'] = check_arm(snapshot, spec, result, seed)
        results[name] = result
        return result

    try:
        if execution == 'warm':
            capture = arm_spec(snapshot, plan, 'seed', history='capture', attention='native', probe='none')
            seed = run(capture, 'seed')['worker']['capture']
        native = arm_spec(snapshot, plan, 'zero-native', history='compute',
                          attention='native', probe='save', control=True)
        native_result = run(native, 'zero-native')
        if execution == 'warm':
            warm = arm_spec(snapshot, plan, 'zero-restored-native', history='restore',
                            attention='native', probe='compare', control=True)
            run(warm, 'zero-restored-native')
        zero = arm_spec(snapshot, plan, 'zero-shadow', history=restore,
                        attention='masked', probe='compare', control=True)
        run(zero, 'zero-shadow')
        controls = {}
        for name in ('zero-restored-native', 'zero-shadow'):
            if name in results:
                result = results[name]
                controls[name] = {
                    'layer_max_relative_l2': result['max_probe_relative_l2'],
                    'sampled_token_equal': result['token_ids'] == native_result['token_ids'],
                }
        passed = all(row['sampled_token_equal']
                     and max(row['layer_max_relative_l2'].values()) < CONTROL_L2
                     for row in controls.values())
        write_json(output / 'zero-deletion.json', {'passed': passed, 'threshold': CONTROL_L2, 'controls': controls})
        require(passed, 'Zero-deletion control failed; see zero-deletion.json and raw Probe measurements')
        if execution == 'replay':
            require(results['zero-shadow']['worker']['historical_attention_sha256']
                    == native_result['worker']['historical_attention_sha256'],
                    'Recomputed history changed in zero-deletion control')

        g0 = arm_spec(snapshot, plan, 'G0', history=restore, attention='masked', probe='save')
        baseline = run(g0, 'G0')
        prepared_text = plan['text']
        started = time.perf_counter()
        text = backend.retokenize(snapshot, plan['geometry']['text_character_span'])
        text_seconds = time.perf_counter() - started
        require(text['input_ids'] == prepared_text['input_ids']
                and text['query_start'] == prepared_text['query_start']
                and digest(text['prompt']) == prepared_text['prompt_sha256'],
                'Runtime text edit or tokenizer disagrees with prepared G1')
        g1 = arm_spec(snapshot, plan, 'G1', history='compute', attention='masked',
                      probe='compare', text=text)
        g2 = arm_spec(snapshot, plan, 'G2', history=restore, attention='masked', probe='compare')
        g2['expected_attention_history'] = baseline['worker']['historical_attention_sha256']
        arms = (g1, g2) if repeat % 2 == 0 else (g2, g1)
        for spec in arms:
            run(spec, spec['arm'])
        require(results['G2']['worker']['historical_attention_sha256'] == g2['expected_attention_history'],
                'Shadow changed original historical Attention KV before selection')
        summary = {
            'snapshot_sha256': snapshot['sha256'], 'plan_sha256': plan['sha256'],
            'mode': plan['mode'], 'repeat': repeat, 'execution': execution,
            'candidate': plan['candidate'], 'deletion_geometry': plan['geometry'],
            'g1_text_preprocessing_seconds': text_seconds,
            'arms': {arm: {
                'prompt_tokens': results[arm]['prompt_tokens'],
                'output_tokens': len(results[arm]['token_ids']),
                'decoded_seconds': results[arm]['decode_seconds'],
                'wall_seconds': results[arm]['wall_seconds'],
                'time_to_first_token_seconds': results[arm]['ttft_seconds'],
                'cuda_forward_seconds': results[arm]['worker']['cuda_forward_seconds'],
                'history_tokens_forwarded': results[arm]['worker']['history_tokens_forwarded'],
                'restored_history_tokens': results[arm]['worker']['restored_history_tokens'],
                'restore_seconds': results[arm]['worker']['restore_seconds'],
                'output_matches_G0': baseline['token_ids'] == results[arm]['token_ids'],
                'max_probe_relative_l2': results[arm]['max_probe_relative_l2'],
                'physically_freed_bytes': results[arm]['worker']['physically_freed_bytes'],
            } for arm in ('G0', 'G1', 'G2')},
        }
        if seed is not None:
            summary['seed_wall_seconds'] = results['seed']['wall_seconds']
        write_json(output / 'pair.json', summary)
        return summary
    finally:
        unhandled = sys.exc_info()[1]
        try:
            backend.release_snapshot()
        except Exception as cleanup_error:
            if unhandled is None:
                raise
            unhandled.add_note(f'Snapshot release also failed: {cleanup_error}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, default=STRICT)
    parser.add_argument('--execution', choices=('replay', 'warm'), default='replay')
    parser.add_argument('--candidate')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--allow-unreviewed', action='store_true', help='Allow an unreviewed case for a smoke run')
    parser.add_argument('--plan-only', action='store_true', help='Validate and list cases without loading vLLM')
    for option in ('output', 'model', 'tokenizer'):
        parser.add_argument(f'--{option}', type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--repeats and --limit must be positive')
    try:
        prepared, all_cases = read_prepared(args.prepared)
        cases = select_cases(all_cases, args.mode, args.candidate, args.limit, args.allow_unreviewed)
        config = cases[0][0]['config']
        require(all(snapshot['config'] == config for snapshot, _ in cases),
                'A single study must use one model and runtime configuration')
        study = {
            'prepared_sha256': prepared['manifest_sha256'], 'config': config,
            'mode': args.mode, 'execution': args.execution, 'repeats': args.repeats,
            'smoke_run': args.allow_unreviewed or args.repeats < 3,
            'cases': [{'candidate_id': plan['candidate']['candidate_id'],
                       'snapshot_sha256': snapshot['sha256'], 'plan_sha256': plan['sha256']}
                      for snapshot, plan in cases],
        }
        if args.plan_only:
            print(json.dumps(study, ensure_ascii=False, indent=2))
            return
        if None in (args.model, args.tokenizer, args.output):
            parser.error('--model, --tokenizer and --output are required for inference')
        from .kv_intervention import NativeBackend
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'study.json', study)
        backend = None
        try:
            backend = NativeBackend(args.model, args.tokenizer, cases, args.execution)
            write_json(args.output / 'environment.json', backend.environment)
            completed = []
            for index, (snapshot, plan) in enumerate(cases, 1):
                for repeat in range(args.repeats):
                    folder = args.output / 'cases' / f'{index:04d}' / f'repeat-{repeat + 1:02d}'
                    run_pair(backend, snapshot, plan, args.execution, folder, repeat)
                    completed.append((folder / 'pair.json').relative_to(args.output).as_posix())
            write_json(args.output / 'summary.json', {'status': 'complete', 'pairs': completed})
            print(f'Completed {len(completed)} pairs at {args.output}')
        except Exception as error:
            write_json(args.output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
            raise
        finally:
            if backend is not None:
                backend.close()
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
