"""One L2 experiment using the completed S4 implementation without source edits."""
from __future__ import annotations
import copy
import json
import logging
import os
from pathlib import Path
import sys
from html import escape
from urllib.parse import urlsplit, parse_qs

ROOT = Path('/Users/sree/aperture')
sys.path.insert(0, str(ROOT))
import run_s4_targeted as frozen
import targeted_retrieval as retrieval
from provenance_gate import Delivery, check_claims, publish_result, digest

TRACE_ID = 'val:smol+gaia_gpt-4o-mini_smoagents_23_9c7c5be9'
STUDY_ID = 'l2-frozen-mandatory-expansion-20261011-v1'
OUT = ROOT / 'reports' / STUDY_ID
S4 = ROOT / 'reports/s4-mandatory-expansion-spans-20261011-v3/comparison.json'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def current_hashes(names):
    return {name: digest((ROOT / name).read_text()) for name in names}


def summarize_delivery(doc, events):
    sizes = {e['step']: len(e['content']) for e in events}
    best = {}
    chunks = []
    for c in doc['chunks']:
        count = len(c['text'])
        best[c['step']] = max(best.get(c['step'], 0), count)
        chunks.append({'event_id': c['step'], 'author': c['agent_name'],
                       'delivered_characters': count, 'source_characters': c['source_chars'],
                       'omitted_characters': c['source_chars'] - count,
                       'clipped': c['truncated'], 'chunk_id': c['chunk_id']})
    return {'delivery_id': doc['_id'], 'run_id': doc['run_id'], 'round': doc['round'],
            'evidence_characters': doc['evidence_chars'], 'evidence_tokens': doc['evidence_tokens'],
            'input_estimate_with_reserve': doc['context_tokens_with_framing_reserve'],
            'event_ids_present': sorted(best),
            'complete_event_ids': sorted(s for s, n in best.items() if n == sizes[s]),
            'clipped_event_ids': sorted(s for s, n in best.items() if n < sizes[s]),
            'entirely_omitted_event_ids': sorted(set(sizes) - set(best)),
            'unique_source_characters_delivered': sum(best.values()),
            'unique_source_characters_omitted': sum(sizes.values()) - sum(best.values()),
            'chunks': chunks}


def report(bundle, calls):
    def table(headers, rows):
        return '<div class="scroll"><table><tr>' + ''.join('<th>' + escape(str(x)) + '</th>' for x in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</table></div>'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aperture L2 frozen comparison</title>',
             '<style>body{font:16px/1.5 system-ui,sans-serif;color:#182938;background:#f7fafb;margin:0}main{max-width:1200px;margin:auto;padding:32px 24px}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d9e1e6;text-align:left;vertical-align:top}th{background:#e6f0f3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{margin:14px 0}code{overflow-wrap:anywhere}</style><main>',
             '<p>APERTURE · L2 ONLY · 11 OCTOBER 2026</p><h1>Frozen implementation comparison</h1><p><code>' + TRACE_ID + '</code></p>',
             '<p>Unmodified completed S4 implementation: gpt-4o-mini, temperature 0, JSON output, 2,000 completion tokens maximum. Identical span citation protocol and host provenance gate. Short context: 180 characters per event. Targeted: mandatory first expansion when clipped events are visible, investigator-selected events, at most two rounds and five events per round, 4,000 characters per retrieved event. Input budget: 12,000 tokens, evidence subbudget: 10,000 tokens. No prompt tuning or event-selection heuristics.</p>',
             '<p>Ground truth was read only after every investigator arm finished. Provenance establishes delivered evidence, not correct fault attribution. No failed arm is repaired or rerun.</p>']
    rows = []
    for run in bundle['runs']:
        ev = next(e for e in bundle['evaluations'] if e['arm'] == run['arm'])
        use = run.get('usage', {})
        gate = run.get('gate', {})
        rows.append([run.get('context_label', run['arm']), run.get('candidate_predicted_agents'), ev.get('candidate_f1'), ev.get('candidate_exact_match'), run['status'], gate.get('errors', []) + gate.get('protocol_errors', []), use.get('prompt_tokens'), use.get('completion_tokens'), use.get('total_tokens'), run.get('call_count')])
    parts.append(table(['Arm', 'Predictions', 'Candidate F1', 'Exact', 'Gate/status', 'Rejection reasons', 'Input', 'Output', 'All-call tokens', 'Calls'], rows))
    parts.append('<p>Actual faulty agents (evaluation only): ' + escape(', '.join(bundle['actual_agents'])) + '</p>')
    parts.append('<h2>Model-selected retrieval and actual delivery</h2>')
    parts.append(table(['Round', 'Selected events', 'Actually expanded events', 'Details'], [[r['round'], r['requested_steps'], r['actually_delivered_steps'], json.dumps(r['log'])] for r in bundle['retrievals']]))
    parts.append('<h2>Every call, including selection and provisional diagnoses</h2>')
    parts.append(table(['Arm', 'Round', 'Status', 'Input', 'Output', 'Total', 'Seconds', 'Requested events'], [[c['arm'], c['round'], c['status'], c.get('usage', {}).get('prompt_tokens'), c.get('usage', {}).get('completion_tokens'), c.get('usage', {}).get('total_tokens'), c.get('elapsed_seconds'), c.get('candidate', {}).get('retrieve_steps', [])] for c in calls]))
    parts.append('<pre>' + escape(json.dumps(bundle['accounting'], indent=2)) + '</pre>')
    parts.append('<h2>Truncation after budget enforcement</h2>')
    run_names = {r['_id']: r.get('context_label', r['arm']) for r in bundle['runs']}
    parts.append(table(['Arm', 'Call round', 'Evidence chars', 'Unique source chars delivered', 'Unique source chars omitted', 'Complete events', 'Clipped events', 'Entirely omitted'], [[run_names[d['run_id']], d['round'], d['evidence_characters'], d['unique_source_characters_delivered'], d['unique_source_characters_omitted'], d['complete_event_ids'], d['clipped_event_ids'], d['entirely_omitted_event_ids']] for d in bundle['truncation']]))
    parts.append('<h2>Final candidate and gate audit</h2>')
    for run in bundle['runs']:
        final = next((c for c in calls if c['_id'] == run.get('final_call_id')), {})
        parts.append('<details open><summary>' + escape(run.get('context_label', run['arm'])) + '</summary><pre>' + escape(json.dumps({'candidate': final.get('candidate'), 'gate': run.get('gate'), 'error': run.get('error_class')}, indent=2, ensure_ascii=False)) + '</pre></details>')
    parts.append('<h2>Freeze, tests, and Atlas verification</h2><pre>' + escape(json.dumps({'freeze': bundle['freeze'], 'tests': {'passed': bundle['gate_tests']['passed'], 'tests_run': bundle['gate_tests']['tests_run']}, 'persistence': bundle['persistence_verification'], 'limitations': bundle['limitations']}, indent=2)) + '</pre>')
    parts.append('<p>Audit files: <a href="comparison.json">comparison</a> · <a href="deliveries.json">exact delivered evidence and prompts</a> · <a href="calls.json">all calls</a> · <a href="retrievals.json">retrievals</a> · <a href="configuration.json">configuration</a> · <a href="freeze.json">freeze manifest</a> · <a href="gate-tests.json">tests</a> · <a href="run_l2_frozen.py">experiment driver</a>.</p></main></html>')
    return '\n'.join(parts)


def main():
    s4 = json.loads(S4.read_text())
    expected = s4['decoder_recovery']['recovery_code_sha256']
    before = current_hashes(expected)
    assert before == expected, 'Frozen implementation differs from completed S4'
    extras = current_hashes(['dashboard.py', 'requirements.txt'])
    assert retrieval.PROTOCOL_VERSION == s4['config']['protocol']
    assert frozen.GATE_VERSION == s4['config']['gate_version']
    assert digest(retrieval.SYSTEM_PROMPT) == s4['config']['system_prompt_sha256']
    assert retrieval.DEFAULT_POLICY == s4['config']['policy']
    assert retrieval.MAX_ROUNDS == s4['config']['max_retrieval_rounds']
    assert retrieval.MAX_CONTEXT_TOKENS == s4['config']['context_token_budget']
    assert retrieval.MAX_EVIDENCE_TOKENS == s4['config']['evidence_token_budget']
    tests = frozen.run_tests()
    assert tests['passed'], 'Frozen tests failed'
    print(f"Frozen S4 hashes match; {tests['tests_run']} tests passed.", flush=True)
    # Only experiment routing changes. No implementation, prompt, or policy edits.
    frozen.TRACE_ID, frozen.STUDY_ID, frozen.OUT = TRACE_ID, STUDY_ID, OUT
    frozen.load_dotenv(ROOT / '.env', override=True)
    uri = os.environ['MONGODB_URI']
    parsed = urlsplit(uri)
    assert parsed.scheme == 'mongodb+srv' and (parsed.hostname or '').endswith('.mongodb.net')
    options = {k.lower(): v for k, v in parse_qs(parsed.query).items()}
    assert not any(v.lower() == 'true' for k in ('tlsinsecure', 'tlsallowinvalidcertificates', 'tlsallowinvalidhostnames') for v in options.get(k, []))
    assert not os.getenv('OPENAI_BASE_URL') or urlsplit(os.environ['OPENAI_BASE_URL']).hostname == 'api.openai.com'
    def connect():
        return frozen.MongoClient(uri, tlsCAFile=frozen.certifi.where(), serverSelectionTimeoutMS=30000, connectTimeoutMS=10000, socketTimeoutMS=45000)
    with connect() as client:
        client.admin.command('ping')
        db = client[frozen.harness.DB_NAME]
        assert not db.citation_studies.find_one({'_id': STUDY_ID}), 'Experiment already exists; never duplicate paid calls'
        trace = db.traces.find_one({'_id': TRACE_ID}, {'_id': 1, 'task': 1, 'task_type': 1, 'history': 1, 'split': 1})
        assert trace and trace['split'] == 'val', 'Requested L2 validation trace missing'
        events = [frozen.harness.event_details(e, i+1) for i, e in enumerate(trace['history'])]
        assert all(type(e['step']) is int for e in events) and len({e['step'] for e in events}) == len(events)
        known = sorted({e['agent_name'] for e in events if e['agent_name'].lower() not in {'user', 'system', 'unknown'}})
        source = {'trace_id': TRACE_ID, 'task': trace['task'], 'task_type': trace['task_type'], 'events': events}
        config = copy.deepcopy(s4['config'])
        config.update(trace_id=TRACE_ID, source_sha256=digest(frozen.stable(source)), known_agents=known,
                      code_sha256=before, experiment_id=STUDY_ID)
        frozen_fields = set(s4['config']) - {'trace_id', 'source_sha256', 'known_agents', 'code_sha256'}
        assert all(config[k] == s4['config'][k] for k in frozen_fields)
        run_ids = [frozen.cache_key(config, arm) for arm in frozen.ARMS]
        assert not db.citation_runs.count_documents({'_id': {'$in': run_ids}}), 'Fresh calls required'
        freeze = {'reference_study': s4['study_id'], 'code_sha256_before': before,
                  'additional_hashes_before': extras, 'matches_completed_s4': True,
                  'driver_sha256': digest(Path(__file__).read_text()),
                  'only_runtime_overrides': ['TRACE_ID', 'STUDY_ID', 'OUT'],
                  'only_config_differences': ['trace_id', 'source_sha256', 'known_agents', 'code_sha256 (completed S4 decoder recovery)', 'experiment_id (fresh-cache namespace)']}
        db.citation_studies.insert_one({'_id': STUDY_ID, 'status': 'running', 'created_at': frozen.now(), 'config': config, 'max_total_model_calls': 5, 'freeze': freeze})
        db.citation_sources.insert_one({'_id': STUDY_ID, **source})
        db.citation_policies.insert_one({'_id': STUDY_ID, 'policy': config['policy']})
        db.citation_tests.insert_one({'_id': STUDY_ID, **tests})
        write('configuration.json', config)
        write('gate-tests.json', tests)
        write('freeze.json', freeze)
        (OUT / 'run_l2_frozen.py').write_text(Path(__file__).read_text())
        print(json.dumps({'trace_id': TRACE_ID, 'events': len(events), 'source_characters': sum(len(e['content']) for e in events), 'labels_read': False, 'fresh': True}), flush=True)
        runs, failures = [], []
        with frozen.harness.model_client().with_options(timeout=90, max_retries=0) as api:
            for arm in frozen.ARMS:
                try:
                    runs.append(frozen.run_arm(db, api, trace, events, config, arm))
                except Exception as exc:
                    failed = {'arm': arm, 'error_class': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None), 'raw_error_redacted': True, 'at': frozen.now()}
                    failures.append(failed)
                    run_id = frozen.cache_key(config, arm)
                    db.citation_runs.update_one({'_id': run_id}, {'$set': {'status': 'error', **failed}})
                    runs.append(db.citation_runs.find_one({'_id': run_id}))
                    print(json.dumps({'failed_attempt': failed}), flush=True)
        # L2 labels are fetched only after all arms and model calls have ended.
        label = db.labels.find_one({'_id': TRACE_ID})
        assert label and isinstance(label['faulty_agents'], list)
        actual = sorted(frozen.harness.normalize_names(label['faulty_agents']))
        evaluations = []
        for run in runs:
            predicted = run.get('candidate_predicted_agents')
            f1, exact = frozen.harness.metrics(predicted, actual) if predicted is not None else (None, None)
            accepted = run.get('gate', {}).get('accepted', False)
            ev = {'_id': run['_id'], 'study_id': STUDY_ID, 'trace_id': TRACE_ID, 'arm': run['arm'], 'actual_agents': actual,
                  'candidate_predicted_agents': predicted, 'candidate_f1': f1, 'candidate_exact_match': exact,
                  'published_f1': f1 if accepted else None, 'published_exact_match': exact if accepted else None,
                  'false_positives': sorted(set(predicted) - set(actual)) if predicted is not None else None,
                  'false_negatives': sorted(set(actual) - set(predicted)) if predicted is not None else None}
            db.citation_evaluations.insert_one(ev)
            evaluations.append(ev)
        calls = list(db.citation_calls.find({'study_id': STUDY_ID}))
        deliveries = list(db.citation_deliveries.find({'study_id': STUDY_ID}))
        retrievals = list(db.citation_retrievals.find({'study_id': STUDY_ID}))
        assert len(calls) <= 5
        by_delivery = {d['_id']: d for d in deliveries}
        for call in calls:
            if call['status'] != 'complete':
                continue
            doc = by_delivery[call['delivery_id']]
            assert doc['messages'][-1]['content'].endswith(doc['evidence_text'])
            assert digest(frozen.stable(doc['messages'])) == doc['messages_sha256']
            assert check_claims(call['candidate'], Delivery.from_document(doc), known) == call['gate']
            assert call['usage']['prompt_tokens'] <= retrieval.MAX_CONTEXT_TOKENS
        metered = [c for c in calls if 'usage' in c]
        failed_runs = {r['_id'] for r in runs if r['status'] != 'accepted'}
        accepted_calls = [c for c in metered if c['run_id'] not in failed_runs]
        failed_calls = [c for c in metered if c['run_id'] in failed_runs]
        accounting = {'total_api_attempts': len(calls), 'metered_calls': len(metered),
                      'all_calls_usage': frozen.usage_totals(metered),
                      'accepted_arm_calls': len(accepted_calls), 'accepted_arm_usage': frozen.usage_totals(accepted_calls),
                      'failed_or_rejected_arm_calls': len(failed_calls), 'failed_or_rejected_arm_usage': frozen.usage_totals(failed_calls),
                      'unmetered_api_attempts': [c['_id'] for c in calls if 'usage' not in c],
                      'automatic_retries': 0, 'experiment_reruns': 0,
                      'cache_reused': any(r.get('cache_used', False) for r in runs),
                      'failures': failures, 'prior_S4_calls_included': False}
        freeze.update(code_sha256_after=current_hashes(expected), additional_hashes_after=current_hashes(extras))
        freeze['unchanged_after_comparison'] = freeze['code_sha256_after'] == before and freeze['additional_hashes_after'] == extras
        assert freeze['unchanged_after_comparison']
        bundle = {'study_id': STUDY_ID, 'trace_id': TRACE_ID, 'config': config, 'freeze': freeze,
                  'source_summary': {'events': len(events), 'source_characters': sum(len(e['content']) for e in events)},
                  'runs': runs, 'evaluations': evaluations, 'actual_agents': actual, 'accounting': accounting,
                  'retrievals': retrievals, 'truncation': [summarize_delivery(d, events) for d in deliveries],
                  'gate_tests': tests, 'labels_read_after_all_calls': True,
                  'limitations': ['One L2 case with one fresh session per arm; no general accuracy conclusion.',
                                  'Provenance acceptance does not establish attribution correctness.',
                                  'Mandatory first expansion is fixed behavior, not learned retrieval.',
                                  'Token totals include every selection, provisional diagnosis, and final diagnosis call.',
                                  'Unknown usage on a failed API attempt would be reported separately, never counted as zero.'],
                  'persistence_verification': {'status': 'pending'}}
        for name, value in [('calls.json', calls), ('deliveries.json', deliveries), ('retrievals.json', retrievals), ('comparison.json', bundle), ('freeze.json', freeze)]:
            write(name, value)
        db.citation_studies.update_one({'_id': STUDY_ID}, {'$set': {'status': 'completed_with_failures' if failures else 'complete', 'completed_at': frozen.now(), 'accounting': accounting, 'freeze': freeze}})
    # Independent connection checks saved evidence and gate results without API calls.
    with connect() as client:
        db = client[frozen.harness.DB_NAME]
        counts = {'citation_runs': 3, 'citation_calls': len(calls), 'citation_deliveries': len(deliveries), 'citation_retrievals': len(retrievals), 'citation_evaluations': 3}
        for name, n in counts.items():
            assert db[name].count_documents({'study_id': STUDY_ID}) == n
        for run in runs:
            if not run.get('final_call_id'):
                continue
            final = db.citation_calls.find_one({'_id': run['final_call_id']})
            delivery = Delivery.from_document(db.citation_deliveries.find_one({'_id': final['delivery_id']}))
            gate = publish_result(db.citation_results, run['_id'], final['candidate'], delivery, known,
                                  first_expansion_required=run['mandatory_first_expansion'], actually_expanded=run['actually_expanded_steps'])
            assert gate == run['gate']
        accepted = sum(r.get('gate', {}).get('accepted', False) for r in runs)
        assert db.citation_results.count_documents({'_id': {'$in': run_ids}}) == accepted
        assert db.citation_sources.find_one({'_id': STUDY_ID}) == {'_id': STUDY_ID, **source}
        assert db.citation_tests.find_one({'_id': STUDY_ID}) == {'_id': STUDY_ID, **tests}
        bundle['persistence_verification'] = {'status': 'verified', 'at': frozen.now(), 'database': frozen.harness.DB_NAME, **counts, 'accepted_published_results': accepted, 'tests': tests['tests_run']}
        html = report(bundle, calls)
        db.citation_reports.insert_one({'_id': STUDY_ID, **bundle, 'html': html, 'html_sha256': digest(html)})
        assert db.citation_reports.find_one({'_id': STUDY_ID}) == {'_id': STUDY_ID, **bundle, 'html': html, 'html_sha256': digest(html)}
    write('comparison.json', bundle)
    (OUT / 'report.html').write_text(html)
    print(json.dumps({'complete': True, 'report': str(OUT / 'report.html'), 'actual_agents': actual, 'evaluations': evaluations, 'accounting': accounting, 'persistence': bundle['persistence_verification']}, indent=2), flush=True)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_class': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None), 'raw_error_redacted': True}), flush=True)
        raise SystemExit(1) from None
