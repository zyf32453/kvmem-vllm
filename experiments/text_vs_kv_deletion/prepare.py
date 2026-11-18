"""Build and verify a CPU-only dataset from existing Graph Cleanup artifacts."""

import argparse
from collections import Counter
import csv
import hashlib
from functools import partial
import json
from pathlib import Path, PurePosixPath

from experiments.graph_cleanup.protocol import canonical, digest, history, token_mapping

from .protocol import (
    HISTORY_HEADER, LABELS, MODES, fingerprint, history_coordinates, make_deletion_plan,
    messages_for, probe_start, require, seal, validate_config, validate_encoding, validate_plan,
    validate_runtime_capture, validate_snapshot, verify_seal,
)

REVIEW_COLUMNS = ('candidate_id', 'label', 'reviewer', 'evidence')


def within(root, relative):
    root = root.resolve()
    require(isinstance(relative, str) and relative and '\\' not in relative
            and ':' not in relative and not PurePosixPath(relative).is_absolute(),
            'Expected a relative POSIX artifact path')
    path = (root / relative).resolve()
    require(path != root and path.is_relative_to(root), 'Artifact path escapes its input directory')
    return path


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def render(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def encode(tokenizer, prompt):
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded['input_ids'])
    offsets = [list(offset) for offset in encoded['offset_mapping']]
    validate_encoding(prompt, ids, offsets)
    return ids, offsets


def load_tokenizer(path):
    from transformers import AutoTokenizer, __version__ as transformers_version
    from tokenizers import __version__ as tokenizers_version

    require(path.is_dir(), 'Tokenizer must be an existing local directory')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True,
                                              trust_remote_code=False)
    require(tokenizer.is_fast and bool(tokenizer.chat_template),
            'The replay requires a fast tokenizer with a chat template')
    evidence = {
        'transformers_version': transformers_version,
        'tokenizers_version': tokenizers_version,
        'backend_sha256': digest(tokenizer.backend_tokenizer.to_str()),
        'template_sha256': fingerprint(tokenizer.chat_template),
    }
    return tokenizer, evidence


def graph_snapshot(bundle, graph_path, source_hashes, config, tokenizer, tokenizer_evidence):
    graph_bytes = graph_path.read_bytes()
    source = json.loads(graph_bytes)
    units = source['units']
    require(bool(units), 'Graph snapshot contains no historical units')
    trace = units[0]['trace_path']
    require(all(unit['trace_path'] == trace for unit in units), 'Snapshot uses multiple trajectories')
    trace_bytes = within(bundle, trace).read_bytes()
    trace_text = trace_bytes.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    require(digest(trace_text) == source_hashes[trace], f'Graph source hash mismatch for {trace}')
    selected_lines = {u['trace_line'] for u in units}
    source_messages = [
        {'trace_line': i, 'record': json.loads(line)}
        for i, line in enumerate(trace_text.split('\n'), 1) if i in selected_lines
    ]
    replay, body_spans = history(units)
    require(replay == source['rendered_history'] and body_spans == source['body_character_spans'],
            'Graph Cleanup rendering or source coordinates disagree')
    messages = messages_for(replay)
    prompt = render(tokenizer, messages)
    ids, offsets = encode(tokenizer, prompt)
    history_span, query_character = history_coordinates(prompt, messages, replay)
    unit_spans = {uid: [a + history_span[0], b + history_span[0]] for uid, (a, b) in body_spans.items()}
    payload = {key: source[key] for key in (
        'snapshot_id', 'task_id', 'trajectory_id', 'checkpoint_step', 'left_censored',
    )}
    payload.update(
        scope='graph_cleanup_rerendered_trace',
        source={'trace_path': trace, 'file_sha256': hashlib.sha256(trace_bytes).hexdigest(),
                'graph_trace_text_sha256': source_hashes[trace],
                'graph_snapshot_sha256': hashlib.sha256(graph_bytes).hexdigest()},
        source_messages=source_messages, units=units,
        messages=messages, prompt=prompt, input_ids=ids, offsets=offsets,
        history_character_span=history_span, probe_character_start=query_character,
        query_start=probe_start(offsets, query_character),
        unit_character_spans=unit_spans,
        unit_token_mapping=token_mapping(offsets, unit_spans, config['kernel_page_tokens']),
        config=config, tokenizer_evidence=tokenizer_evidence,
        runtime_state={
            'physical_page_mapping': None, 'attention_kv_sha256': None,
            'recurrent_state': {'layout': config['recurrent_layout'], 'captured': False},
        },
    )
    return validate_snapshot(seal('frozen_snapshot', payload))


def encode_variant(tokenizer, snapshot, character_span):
    a, b = character_span
    messages = [dict(message) for message in snapshot['messages']]
    user_start = snapshot['history_character_span'][0] - len(HISTORY_HEADER)
    content = messages[1]['content']
    messages[1]['content'] = content[:a - user_start] + content[b - user_start:]
    prompt = render(tokenizer, messages)
    ids, offsets = encode(tokenizer, prompt)
    query = probe_start(offsets, snapshot['probe_character_start'] - (b - a))
    return {'prompt': prompt, 'input_ids': ids, 'query_start': query}


def read_reviews(path, ids):
    if path is None:
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == REVIEW_COLUMNS, 'Unexpected human review columns')
        reviews = {}
        seen = set()
        for row in reader:
            require(set(row) == set(REVIEW_COLUMNS) and all(isinstance(v, str) for v in row.values()),
                    'Invalid human review row')
            candidate_id = row.pop('candidate_id')
            require(candidate_id in ids and candidate_id not in seen, 'Unknown or duplicate review candidate')
            seen.add(candidate_id)
            row = {k: v.strip() for k, v in row.items()}
            if any(row.values()):
                require(row['label'] in LABELS and row['reviewer'] and row['evidence'],
                        f'Incomplete or unknown human label for {candidate_id}')
                reviews[candidate_id] = row
    return reviews


def candidate_for(source, snapshot, review):
    require(source['snapshot_id'] == snapshot['snapshot_id'] and source['task_id'] == snapshot['task_id'],
            'Graph candidate belongs to a different history')
    require(source['selection_uses_future_after_checkpoint'] is False,
            'Candidate selection used information after the checkpoint')
    unit = next((u for u in snapshot['units'] if u['unit_id'] == source['unit_id']), None)
    require(unit is not None, 'Graph candidate unit missing from snapshot')
    require(source['text'] == unit['text'] and source['content_type'] == unit['content_type']
            and source['source'] == {k: unit[k] for k in (
                'trace_path', 'trace_line', 'field_pointer', 'character_span', 'text_sha256',
            )}, 'Graph candidate text, type, or source changed')
    require(source['text_cluster_sha256'] == digest(canonical(unit['text'])),
            'Graph candidate text cluster hash mismatch')
    return {
        'candidate_id': source['candidate_id'], 'unit_id': unit['unit_id'],
        'role': unit['role'], 'content_type': unit['content_type'],
        'graph_category': source['category'], 'text_cluster_sha256': source['text_cluster_sha256'],
        'human_review': review,
    }


def build(bundle, dataset, output, config, tokenizer, tokenizer_evidence, review_file=None):
    validate_config(config)
    candidate_file = within(dataset, 'candidates.jsonl')
    candidate_bytes = candidate_file.read_bytes()
    candidates = [json.loads(row) for row in candidate_bytes.decode('utf-8').split('\n') if row.strip()]
    ids = {row['candidate_id'] for row in candidates}
    require(bool(candidates) and len(ids) == len(candidates)
            and all(isinstance(i, str) and i and i[0].isalnum() for i in ids),
            'Candidate IDs must be unique and nonempty')
    source_hash_file = within(dataset, 'source_hashes.json')
    source_hash_bytes = source_hash_file.read_bytes()
    source_hashes = json.loads(source_hash_bytes)
    reviews = read_reviews(review_file, ids)

    snapshots = {}
    records = {}
    cases = []
    encode_text = partial(encode_variant, tokenizer)
    for row in candidates:
        sid = row['snapshot_id']
        if sid not in snapshots:
            path = within(dataset, f'snapshots/{sid}.json')
            snapshots[sid] = graph_snapshot(bundle, path, source_hashes,
                                            config, tokenizer, tokenizer_evidence)
        snapshot = snapshots[sid]
        candidate = candidate_for(row, snapshot, reviews.get(row['candidate_id']))
        snapshot_path = f'snapshots/{snapshot["sha256"]}.json'
        records[snapshot_path] = snapshot
        plans = {}
        for mode in MODES:
            plan = make_deletion_plan(snapshot, candidate, mode, encode_text)
            path = f'plans/{plan["sha256"]}.json'
            records[path] = plan
            plans[mode] = path
        cases.append({'candidate_id': row['candidate_id'], 'snapshot': snapshot_path, 'plans': plans})

    # The manifest is written last. A verifier cannot mistake an interrupted
    # output directory for a complete dataset.
    output.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        write_json(output / name, record)
    review_path = output / 'human-review.csv'
    with review_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, lineterminator='\n')
        writer.writeheader()
        for row in candidates:
            cid = row['candidate_id']
            writer.writerow({'candidate_id': cid, **reviews.get(cid, {})})
    manifest = seal('prepared_dataset', {
        'graph_candidates_file_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
        'graph_source_hashes_file_sha256': hashlib.sha256(source_hash_bytes).hexdigest(),
        'files': {name: file_hash(output / name) for name in [*records, review_path.name]},
        'cases': cases,
    })
    write_json(output / 'manifest.json', manifest)
    return verify(output)


def read_prepared(output, bundle=None, capture_files=()):
    """Read verified snapshots and both plans without re-opening their files."""
    manifest = read_json(output / 'manifest.json')
    verify_seal(manifest, 'prepared_dataset')
    require(bool(manifest['cases']) and 'human-review.csv' in manifest['files'],
            'Prepared dataset is incomplete')
    for relative, expected in manifest['files'].items():
        require(file_hash(within(output, relative)) == expected, f'Artifact checksum mismatch for {relative}')
    snapshots = {}
    verified_traces = {}
    seen = set()
    eligible = Counter()
    excluded = {mode: Counter() for mode in MODES}
    reviewed = 0
    cases = []
    for case in manifest['cases']:
        require(case['candidate_id'] not in seen, 'Duplicate candidate in prepared dataset')
        seen.add(case['candidate_id'])
        relative = case['snapshot']
        require(relative in manifest['files'], 'Snapshot is missing from the checksum manifest')
        if relative not in snapshots:
            snapshot = validate_snapshot(read_json(within(output, relative)))
            snapshots[relative] = snapshot
            if bundle is not None:
                source = snapshot['source']
                trace = source['trace_path']
                if trace not in verified_traces:
                    verified_traces[trace] = file_hash(within(bundle, trace))
                require(verified_traces[trace] == source['file_sha256'], f'Original trace changed for {trace}')
        snapshot = snapshots[relative]
        require(set(case['plans']) == set(MODES), 'Each case must contain both deletion modes')
        shared_candidate = None
        plans = {}
        for mode, path in case['plans'].items():
            require(path in manifest['files'], 'Plan is missing from the checksum manifest')
            plan = validate_plan(snapshot, read_json(within(output, path)))
            plans[mode] = plan
            candidate = plan['candidate']
            require(mode == plan['mode'] and candidate['candidate_id'] == case['candidate_id'],
                    'Case, mode, and plan do not agree')
            if shared_candidate is None:
                shared_candidate = candidate
            require(candidate == shared_candidate, 'Text and KV plans do not share the same candidate')
            if plan['status'] == 'eligible':
                eligible[mode] += 1
            else:
                excluded[mode][plan['reason']] += 1
        reviewed += shared_candidate['human_review'] is not None
        cases.append((snapshot, plans))

    by_sha = {s['sha256']: s for s in snapshots.values()}
    for path in capture_files:
        capture = read_json(path)
        if 'worker' in capture:
            capture = capture['worker']['capture']
        require(isinstance(capture, dict), f'No native capture recorded in {path}')
        identity = capture['snapshot_sha256']
        require(identity in by_sha, 'Capture does not belong to this dataset')
        validate_runtime_capture(by_sha[identity], capture)
    summary = {
        'manifest_sha256': manifest['sha256'], 'snapshots': len(snapshots), 'candidates': len(seen),
        'reviewed': reviewed, 'eligible_by_mode': {mode: eligible[mode] for mode in MODES},
        'excluded_by_mode': {mode: dict(excluded[mode]) for mode in MODES},
        'runtime_manifests_validated': len(capture_files),
    }
    return summary, cases


def verify(output, bundle=None, capture_files=()):
    return read_prepared(output, bundle, capture_files)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    build_parser = command.add_parser('build')
    for name in ('bundle', 'dataset', 'output', 'config', 'tokenizer'):
        build_parser.add_argument(f'--{name}', type=Path, required=True)
    build_parser.add_argument('--reviews', type=Path)
    verify_parser = command.add_parser('verify')
    verify_parser.add_argument('--output', type=Path, required=True)
    verify_parser.add_argument('--bundle', type=Path)
    verify_parser.add_argument('--capture', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            config = read_json(args.config)
            validate_config(config)
            tokenizer, evidence = load_tokenizer(args.tokenizer)
            result = build(args.bundle, args.dataset, args.output, config, tokenizer, evidence, args.reviews)
        else:
            result = verify(args.output, args.bundle, args.capture)
    except (OSError, ImportError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
