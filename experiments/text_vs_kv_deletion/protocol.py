"""Shared CPU protocol for a frozen replay and its paired deletion plans."""

import json
import re

from experiments.duokv_observation.core import resolve
from experiments.graph_cleanup.protocol import POLICY, PROBES, digest, history, token_mapping

SCHEMA_VERSION = 1
STRICT = 'strict_alignment'
PRACTICAL = 'practical_deletion'
MODES = (STRICT, PRACTICAL)
LABELS = (
    'Explicit Duplicate', 'Superseded Information', 'Obsolete Information',
    'Critical Information',
)
HISTORY_HEADER = 'QUOTED HISTORY:\n'
HISTORY_END = '\nEND QUOTED HISTORY\n\n'
QUESTIONS = 'FIXED QUESTIONS:\n' + json.dumps(PROBES) + (
    '\nReturn one JSON object with exactly these six keys. Each value must be a '
    'concise string of at most 30 words, including supporting unit-ID suffixes '
    '(m#:u#) when available. Use unknown if unsupported.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return digest(json.dumps(value, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(',', ':')))


def sha256_hex(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def seal(kind, payload):
    record = {k: v for k, v in payload.items() if k != 'sha256'}
    record.update(kind=kind, schema_version=SCHEMA_VERSION)
    return {**record, 'sha256': fingerprint(record)}


def verify_seal(record, kind):
    payload = {k: v for k, v in record.items() if k != 'sha256'}
    require(record.get('kind') == kind and type(record.get('schema_version')) is int
            and record['schema_version'] == SCHEMA_VERSION, f'Unsupported {kind} schema')
    require(record.get('sha256') == fingerprint(payload), f'{kind} checksum mismatch')


def span(value, limit, name, *, empty=False):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) is int for x in value), f'Invalid {name}')
    a, b = value
    require(0 <= a <= b <= limit and (empty or a < b), f'{name} out of bounds or empty')
    return a, b


def valid_ids(ids):
    return isinstance(ids, list) and bool(ids) and all(type(i) is int and i >= 0 for i in ids)


def validate_encoding(prompt, ids, offsets):
    require(valid_ids(ids) and len(offsets) == len(ids),
            'Missing input IDs or incomplete offset mapping')
    for a_b in offsets:
        span(a_b, len(prompt), 'token offset', empty=True)


def messages_for(rendered_history):
    return [
        {'role': 'system', 'content': POLICY},
        {'role': 'user', 'content': HISTORY_HEADER + rendered_history + HISTORY_END + QUESTIONS},
    ]


def history_coordinates(prompt, messages, rendered_history):
    user = messages[1]['content']
    position = prompt.find(user)
    require(position >= 0 and prompt.find(user, position + 1) < 0,
            'Chat template must embed the complete user message exactly once')
    begin = position + len(HISTORY_HEADER)
    end = begin + len(rendered_history)
    return [begin, end], end + len(HISTORY_END)


def probe_start(offsets, boundary):
    start = next((i for i, (a, b) in enumerate(offsets) if a < b and b > boundary), None)
    require(start is not None and start > 0, 'Missing history or fixed Probe tokens')
    return start


def validate_config(config):
    for name in ('model', 'tokenizer'):
        item = config[name]
        require(isinstance(item['id'], str) and item['id'].strip(), f'Missing {name} ID')
        require(isinstance(item['revision'], str)
                and re.fullmatch(r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})', item['revision']),
                f'{name} revision must be a commit or an exact manifest SHA-256')
    require(isinstance(config['vllm_version'], str) and config['vllm_version'].strip(),
            'Missing vLLM version')
    for name in ('kernel_page_tokens', 'attention_layers', 'recurrent_layers'):
        require(type(config[name]) is int and config[name] > 0, f'Invalid {name}')
    require(config['recurrent_layout'] == 'qwen_gdn', 'Expected the Qwen GDN replay layout')
    params = config['inference_params']
    require(params.get('enable_prefix_caching') is False, 'Replay requires APC off')
    require(type(params['seed']) is int and type(params['max_tokens']) is int
            and params['max_tokens'] > 0 and type(params['temperature']) in (int, float)
            and 0 <= params['temperature'] < float('inf'), 'Invalid sampling settings')


def verify_unit_source(unit, record):
    require(record['record_type'] == 'message', 'Unit source is not a message')
    require(unit['role'] == record['role'], 'Unit role differs from its source message')
    pointer = unit['field_pointer']
    require(isinstance(pointer, str) and pointer.startswith('/'), 'Invalid source JSON Pointer')
    try:
        field = resolve(record, pointer)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid source field for {unit["unit_id"]}') from error
    require(isinstance(field, str), 'Source field must be text')
    a, b = span(unit['character_span'], len(field), 'source character span')
    require(field[a:b] == unit['text'] and digest(unit['text']) == unit['text_sha256'],
            f'Source content or hash mismatch for {unit["unit_id"]}')


def validate_snapshot(snapshot):
    verify_seal(snapshot, 'frozen_snapshot')
    validate_config(snapshot['config'])
    require(snapshot['scope'] == 'graph_cleanup_rerendered_trace', 'Unexpected snapshot scope')
    require(snapshot['runtime_state'] == {
        'physical_page_mapping': None, 'attention_kv_sha256': None,
        'recurrent_state': {'layout': snapshot['config']['recurrent_layout'], 'captured': False},
    }, 'Attach runtime evidence separately from the immutable CPU snapshot')
    require(sha256_hex(snapshot['source']['file_sha256'])
            and sha256_hex(snapshot['source']['graph_snapshot_sha256']),
            'Missing original trace or Graph snapshot hash')
    require(sha256_hex(snapshot['tokenizer_evidence']['backend_sha256'])
            and sha256_hex(snapshot['tokenizer_evidence']['template_sha256']),
            'Missing tokenizer fingerprints')

    units = snapshot['units']
    unit_ids = {u['unit_id'] for u in units}
    require(bool(units) and len(unit_ids) == len(units), 'Empty or duplicate snapshot units')
    messages = snapshot['source_messages']
    lines = [item['trace_line'] for item in messages]
    require(all(type(i) is int and i > 0 for i in lines) and lines == sorted(set(lines)),
            'Original message lines must be unique and ordered')
    by_line = {item['trace_line']: item['record'] for item in messages}
    require(set(by_line) == {unit['trace_line'] for unit in units}, 'Incomplete source messages')
    for unit in units:
        require(unit['step'] <= snapshot['checkpoint_step'], 'Unit lies after frozen checkpoint')
        require(unit['trace_path'] == snapshot['source']['trace_path'], 'Unexpected trace path')
        verify_unit_source(unit, by_line[unit['trace_line']])

    rendered_history, body_spans = history(units)
    require(snapshot['messages'] == messages_for(rendered_history), 'Replay messages changed')
    prompt = snapshot['prompt']
    history_span, qchar = history_coordinates(prompt, snapshot['messages'], rendered_history)
    require(history_span == snapshot['history_character_span'] and qchar == snapshot['probe_character_start'],
            'Incorrect history or Probe character range')
    require(snapshot['messages'][0]['content'] in prompt[:history_span[0]],
            'Missing rendered system message')
    offsets = snapshot['offsets']
    validate_encoding(prompt, snapshot['input_ids'], offsets)
    require(probe_start(offsets, qchar) == snapshot['query_start'], 'Incorrect Probe token boundary')
    absolute_spans = {uid: [a + history_span[0], b + history_span[0]] for uid, (a, b) in body_spans.items()}
    require(snapshot['unit_character_spans'] == absolute_spans, 'Incorrect unit character mapping')
    require(snapshot['unit_token_mapping'] == token_mapping(offsets, absolute_spans,
            snapshot['config']['kernel_page_tokens']), 'Incorrect unit token or logical page mapping')
    return snapshot


def _geometry(snapshot, unit_id, mode):
    require(mode in MODES, 'Unknown deletion mode')
    require(unit_id in snapshot['unit_character_spans'], 'Candidate unit missing from snapshot')
    prompt = snapshot['prompt']
    offsets = snapshot['offsets']
    a, b = snapshot['unit_character_spans'][unit_id]
    mapping = snapshot['unit_token_mapping'][unit_id]
    target = mapping['token_indices']
    require(bool(target) and target == list(range(target[0], target[-1] + 1))
            and target[-1] < snapshot['query_start'], 'Candidate token mapping is incomplete or outside history')
    covered = {i for j in target for i in range(max(a, offsets[j][0]), min(b, offsets[j][1]))}
    require(all(prompt[i].isspace() or i in covered for i in range(a, b)),
            'Candidate has characters with no matching token offset')
    page = snapshot['config']['kernel_page_tokens']
    blocks = mapping['logical_blocks']
    first, end = blocks[0] * page, (blocks[-1] + 1) * page
    h0, h1 = snapshot['history_character_span']
    reason = None
    if end > snapshot['query_start']:
        reason = 'page_crosses_probe_or_incomplete_tail'
    elif not all(h0 <= start < stop <= h1 for start, stop in offsets[first:end]):
        reason = 'page_crosses_protected_prompt_or_missing_offset'
    kv_eligible = reason is None
    text_span = None
    if kv_eligible and mode == PRACTICAL:
        text_span = [a, b]
    if kv_eligible and mode == STRICT:
        text_span = [min(x for x, _ in offsets[first:end]), max(y for _, y in offsets[first:end])]
        touched = [i for i, (x, y) in enumerate(offsets)
                   if x < y and x < text_span[1] and text_span[0] < y]
        if touched != list(range(first, end)):
            reason, text_span = 'page_boundary_splits_character_offsets', None
    excluded = end - first
    collateral = excluded - len(target)
    history_tokens = sum(1 for x, y in offsets if x < y and x < h1 and h0 < y)
    return {
        'candidate_character_span': [a, b],
        'token_start': target[0],
        'token_end': target[-1] + 1,
        'boundary_tokens': [i for i in target if offsets[i][0] < a or offsets[i][1] > b],
        'logical_blocks': blocks,
        'logical_token_span': [first, end],
        'text_character_span': text_span,
        'kv_eligible': kv_eligible,
        'geometry_reason': reason,
        'history_tokens': history_tokens,
        'planned_excluded_tokens': excluded,
        'collateral_tokens': collateral,
        'collateral_unit_ids': mapping['collateral_unit_ids'],
        'deletion_ratio': excluded / history_tokens if kv_eligible else None,
        'collateral_deletion_ratio': collateral / excluded if kv_eligible else None,
    }


def _pair_reason(snapshot, geometry, mode, text):
    if geometry['text_character_span'] is None:
        require(text is None, 'Text variant provided for a blocked edit')
        return geometry['geometry_reason']
    a, b = geometry['text_character_span']
    ids = text['input_ids']
    query = text['query_start']
    original = snapshot['input_ids']
    require(valid_ids(ids) and type(query) is int and 0 < query < len(ids),
            'Invalid reduced input IDs or Probe boundary')
    require(text['prompt_sha256'] == digest(snapshot['prompt'][:a] + snapshot['prompt'][b:]),
            'Text edit changed characters outside the deletion')
    require(text['removed_prompt_tokens'] == len(original) - len(ids),
            'Incorrect text reduction count')
    if ids[query:] != original[snapshot['query_start']:]:
        return 'probe_ids_changed_after_retokenization'
    if mode == STRICT:
        start, end = geometry['logical_token_span']
        if ids != original[:start] + original[end:]:
            return 'retokenization_changed_retained_ids'
    return None


def make_deletion_plan(snapshot, candidate, mode, encode_variant):
    geometry = _geometry(snapshot, candidate['unit_id'], mode)
    text = None
    if geometry['text_character_span'] is not None:
        variant = encode_variant(snapshot, geometry['text_character_span'])
        text = {
            'prompt_sha256': digest(variant['prompt']),
            'input_ids': variant['input_ids'],
            'query_start': variant['query_start'],
            'removed_prompt_tokens': len(snapshot['input_ids']) - len(variant['input_ids']),
        }
    reason = _pair_reason(snapshot, geometry, mode, text)
    return seal('deletion_plan', {
        'snapshot_sha256': snapshot['sha256'],
        'candidate': candidate,
        'mode': mode,
        'geometry': geometry,
        'text': text,
        'status': 'ineligible' if reason else 'eligible',
        'reason': reason,
    })


def validate_plan(snapshot, plan):
    verify_seal(plan, 'deletion_plan')
    require(plan['snapshot_sha256'] == snapshot['sha256'], 'Plan points to a different snapshot')
    candidate = plan['candidate']
    geometry = _geometry(snapshot, candidate['unit_id'], plan['mode'])
    unit = next(u for u in snapshot['units'] if u['unit_id'] == candidate['unit_id'])
    require(unit['role'] == candidate['role'] and unit['content_type'] == candidate['content_type'],
            'Candidate role or content type differs from its unit')
    review = candidate['human_review']
    require(review is None or (review['label'] in LABELS and review['reviewer'] and review['evidence']),
            'Invalid human review evidence')
    require(geometry == plan['geometry'], 'Plan geometry differs from frozen input')
    reason = _pair_reason(snapshot, geometry, plan['mode'], plan['text'])
    require(plan['reason'] == reason and plan['status'] == ('ineligible' if reason else 'eligible'),
            'Plan status does not match its mapping and token evidence')
    return plan


def validate_runtime_capture(snapshot, capture):
    """Check the shape and binding of an externally recorded native capture."""
    require(capture['snapshot_sha256'] == snapshot['sha256'], 'Capture points to a different snapshot')
    n = snapshot['query_start']
    config = snapshot['config']
    page = config['kernel_page_tokens']
    require(isinstance(capture['cache_instance'], str) and capture['cache_instance']
            and type(capture['request_generation']) is int
            and capture['request_generation'] >= 0, 'Missing capture owner or request generation')
    require(capture['history_tokens'] == n and capture['kernel_page_tokens'] == page,
            'Native capture boundary or geometry mismatch')
    require(capture['config_sha256'] == fingerprint(config)
            and capture['tokenizer_backend_sha256'] == snapshot['tokenizer_evidence']['backend_sha256'],
            'Capture model, tokenizer, or runtime configuration mismatch')
    require(len(capture['attention']) == config['attention_layers'], 'Incomplete Attention capture')

    page_tables = {}
    attention_hashes = {}
    for layer, data in capture['attention'].items():
        slots = data['slot_mapping']
        capacity = data['cache_page_count']
        require(type(capacity) is int and capacity > 0 and len(slots) == n
                and all(type(slot) is int and 0 <= slot < capacity * page for slot in slots),
                f'Incomplete or out-of-bounds native slots for {layer}')
        pages = [slot // page for slot in slots[::page]]
        require(len(set(pages)) == len(pages) and all(
            slot == pages[i // page] * page + i % page for i, slot in enumerate(slots)
        ), f'Inconsistent native page offsets for {layer}')
        require(sha256_hex(data['kv_sha256']), f'Missing historical KV hash for {layer}')
        page_tables[layer] = pages
        attention_hashes[layer] = data['kv_sha256']

    state = capture['recurrent_state']
    require(state['layout'] == config['recurrent_layout'] and state['boundary_tokens'] == n,
            'Recurrent state is from a different boundary or layout')
    conv, ssm = state['conv_sha256'], state['ssm_sha256']
    require(set(conv) == set(ssm) and len(conv) == config['recurrent_layers']
            and all(sha256_hex(h) for h in (*conv.values(), *ssm.values(), state['accepted_counter_sha256'])),
            'Incomplete recurrent state fingerprints')
    return {
        'physical_page_mapping': page_tables,
        'attention_kv_sha256': attention_hashes,
        'recurrent_state': state,
    }


def allocation_slots(blocks, block_size, length, page, capacity):
    """Expand allocator blocks into the token slots seen by the Attention kernel."""
    require(all(type(value) is int and value > 0 for value in (block_size, length, page, capacity))
            and block_size % page == 0, 'Invalid allocator or kernel geometry')
    count = (length + block_size - 1) // block_size
    owned = blocks[:count]
    require(len(owned) == count and len(set(owned)) == count
            and all(type(block) is int and block > 0 for block in owned),
            'Invalid native allocation for warm prefix')
    slots = [owned[i // block_size] * block_size + i % block_size for i in range(length)]
    require(all(slot < capacity * page for slot in slots), 'Warm slot exceeds the native cache')
    return slots
