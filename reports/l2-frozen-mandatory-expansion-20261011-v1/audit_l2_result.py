"""Post-run audit/report only. No API client, investigator calls, or source edits."""
import json
import logging
from pathlib import Path
from html import escape
import run_l2_frozen as driver

frozen = driver.frozen
tr = driver.retrieval
ROOT, OUT = driver.ROOT, driver.OUT


def main():
    bundle = json.loads((OUT / 'comparison.json').read_text())
    calls = json.loads((OUT / 'calls.json').read_text())
    deliveries = json.loads((OUT / 'deliveries.json').read_text())
    run = next(r for r in bundle['runs'] if r['arm'] == 'short_targeted')
    assert run['status'] == 'error' and run['error_class'] == 'ValueError'
    initial = next(d for d in deliveries if d['run_id'] == run['_id'])
    request = bundle['retrievals'][0]
    frozen.load_dotenv(ROOT / '.env', override=True)
    def connect():
        return frozen.MongoClient(driver.os.environ['MONGODB_URI'], tlsCAFile=frozen.certifi.where(), serverSelectionTimeoutMS=30000)
    with connect() as client:
        db = client[frozen.harness.DB_NAME]
        source = db.citation_sources.find_one({'_id': driver.STUDY_ID})
        assert driver.digest(frozen.stable({k:v for k,v in source.items() if k != '_id'})) == bundle['config']['source_sha256']
        trace = {'task': source['task'], 'task_type': source['task_type']}
        prepared, log = tr.deliver(run['session_id'], driver.TRACE_ID, source['events'], 'retrieve',
            previous=driver.Delivery.from_document(initial).chunks, requested_steps=request['requested_steps'],
            policy=bundle['config']['policy'], model=bundle['config']['model_settings']['model'])
        assert log == request['log']
        messages = tr.messages(trace, prepared, 1, set(request['requested_steps']), bundle['config']['policy'], False)
        estimate = tr.context_token_estimate(messages, bundle['config']['model_settings']['model'])
        try:
            tr.enforce_context_budget(messages, bundle['config']['model_settings']['model'])
        except ValueError as exc:
            reason = str(exc)
        else:
            raise AssertionError('Original context failure did not reproduce')
        arm_calls = [c for c in calls if c['run_id'] == run['_id']]
        assert len(arm_calls) == 1 and arm_calls[0]['candidate']['fault_claims'] == []
        assert db.citation_calls.count_documents({'run_id': run['_id']}) == 1
        assert db.citation_deliveries.count_documents({'run_id': run['_id']}) == 1
        audit = {'stage': 'input_budget_enforcement_before_second_call', 'reason': reason,
            'input_tokens_with_framing_reserve': estimate, 'input_token_budget': tr.MAX_CONTEXT_TOKENS,
            'prepared_evidence_tokens': tr.token_count(prepared.evidence_text), 'evidence_token_budget': tr.MAX_EVIDENCE_TOKENS,
            'selected_events': request['requested_steps'], 'prepared_expansions': log,
            'expanded_events_actually_received_by_investigator': [],
            'only_received_evidence': initial['_id'], 'second_api_call_made': False,
            'second_delivery_registered': False, 'final_diagnosis_produced': False,
            'final_provenance_gate_reached': False,
            'selection_gate_note': 'Selection schema passed with zero fault claims; this is not an accepted final diagnosis.',
            'frozen_log_note': 'The frozen retrieval log labels prepared chunks as delivered. Transport audit confirms no expansion reached the investigator because the next request failed its input-budget check.',
            'cause': 'Evidence fit the 10,000-token subbudget, but instructions, event overview, citation framing and reserve pushed the complete input above 12,000 tokens.',
            'audit_method': 'Reconstructed exact prepared context from persisted source, session, initial evidence and model selection; no model calls or source edits.',
            'failed_arm_usage': frozen.usage_totals(arm_calls), 'failure_reproduced_read_only': True}
        assert estimate == 14030
        # Store unreceived material separately, never in the received-evidence collection.
        blocked = {'study_id': driver.STUDY_ID, 'trace_id': driver.TRACE_ID, 'status': 'prepared_but_never_sent',
                   'not_received_evidence': True, 'failure_analysis': audit,
                   'unreceived_evidence_text': prepared.evidence_text, 'unreceived_messages': messages,
                   'unreceived_chunks': prepared.document()['chunks']}
        driver.write('blocked-context.json', blocked)
        driver.write('comparison-initial.json', bundle)
        run['raw_execution_run'] = dict(run)
        run.update(context_label='short + mandatory first expansion', usage=audit['failed_arm_usage'],
                   call_count=1, retrieval_rounds=1, requested_steps=[request['requested_steps']],
                   actually_delivered_steps=[], actually_expanded_steps=[], candidate_predicted_agents=None,
                   total_execution_seconds=arm_calls[0]['elapsed_seconds'], failure_analysis=audit,
                   gate_status='not reached: context budget blocked final diagnosis')
        request['transport_audit'] = {'expanded_events_received_by_investigator': [], 'status': 'blocked_before_next_api_call',
            'original_actually_delivered_steps_means': 'prepared locally only'}
        bundle.update(failure_analysis=audit, blocked_context_file='blocked-context.json', report_audit_at=frozen.now())
        bundle['accounting']['failures'][0]['reason'] = reason
        bundle['accounting']['failed_attempt_stage'] = audit['stage']
        bundle['accounting']['unsent_requests_after_budget_failure'] = 1
        assert driver.current_hashes(bundle['freeze']['code_sha256_before']) == bundle['freeze']['code_sha256_before']
        assert driver.current_hashes(bundle['freeze']['additional_hashes_before']) == bundle['freeze']['additional_hashes_before']
        db.citation_runs.update_one({'_id':run['_id']}, {'$set':{'failure_analysis':audit, 'usage':audit['failed_arm_usage'], 'call_count':1}})
        db.citation_retrievals.update_one({'_id':request['_id']}, {'$set':{'transport_audit':request['transport_audit']}})
        db.citation_studies.update_one({'_id':driver.STUDY_ID}, {'$set':{'failure_analysis':audit, 'accounting':bundle['accounting']}})
        # Frozen renderer is S4-specific, so render only this experiment's report.
        html = driver.report(bundle, calls)
        html = html.replace('<h2>Model-selected retrieval and actual delivery</h2>',
            '<h2>Model-selected retrieval and delivery audit</h2><p><strong>No expanded events reached the investigator.</strong> Events 5, 8, 11, 25 and 39 were prepared locally, but the next input required 14,030 tokens against a 12,000-token budget. The request was blocked before sending. The original frozen retrieval log uses “delivered” for this preparation stage; it does not establish API delivery.</p>')
        html = html.replace('<th>Actually expanded events</th>', '<th>Prepared locally (NOT received)</th>')
        html = html.replace('<h2>Final candidate and gate audit</h2>',
            '<h2>Targeted failure audit</h2><p>The targeted arm produced no final diagnosis, so F1 and exact match are unavailable and the final provenance gate was not reached. Its selection response passed schema validation with zero claims. The accepted baseline diagnoses had valid citations but incorrect attribution.</p><pre>' + escape(json.dumps(audit, indent=2)) + '</pre><p><a href="blocked-context.json">Prepared but unsent context audit</a> is separate from the received-evidence log.</p><h2>Final candidate and gate audit</h2>')
        db.citation_reports.replace_one({'_id':driver.STUDY_ID}, {'_id':driver.STUDY_ID, **bundle,
            'html':html, 'html_sha256':driver.digest(html), 'blocked_context_audit':blocked})
    with connect() as client:
        db = client[frozen.harness.DB_NAME]
        saved = db.citation_reports.find_one({'_id':driver.STUDY_ID})
        assert saved == {'_id':driver.STUDY_ID, **bundle, 'html':html, 'html_sha256':driver.digest(html), 'blocked_context_audit':blocked}
        assert db.citation_calls.count_documents({'study_id':driver.STUDY_ID}) == 3
        assert db.citation_deliveries.count_documents({'study_id':driver.STUDY_ID}) == 3
        assert db.citation_results.count_documents({'_id':{'$in':[r['_id'] for r in bundle['runs']]}}) == 2
        assert db.citation_runs.find_one({'_id':run['_id']})['failure_analysis'] == audit
        assert db.citation_retrievals.find_one({'_id':request['_id']})['transport_audit'] == request['transport_audit']
    driver.write('comparison.json', bundle)
    driver.write('retrievals.json', bundle['retrievals'])
    driver.write('failure-analysis.json', audit)
    (OUT/'audit_l2_result.py').write_text(Path(__file__).read_text())
    (OUT/'report.html').write_text(html)
    print(json.dumps({'audit_saved_and_verified':True, 'implementation_unchanged':True, 'no_new_calls':True,
        'input_tokens':estimate, 'budget':tr.MAX_CONTEXT_TOKENS, 'selected':request['requested_steps'],
        'expanded_events_received':[], 'total_tokens':bundle['accounting']['all_calls_usage'],
        'failed_arm_tokens':audit['failed_arm_usage'], 'report':str(OUT/'report.html')},indent=2))


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'audit_failed','error_class':type(exc).__name__,'raw_error_redacted':True}))
        raise SystemExit(1) from None
