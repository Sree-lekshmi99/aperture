"""S4-only cited comparison. Maximum five model calls; no automatic retries.

python -B run_s4_targeted.py --check-only
python -B -u run_s4_targeted.py --run
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from html import escape
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlsplit
from types import SimpleNamespace
import uuid

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

import aperture as harness
from provenance_gate import Delivery, GATE_VERSION, check_claims, digest, publish_result
from targeted_retrieval import (DEFAULT_POLICY, MAX_ROUNDS, PROTOCOL_VERSION,
                                SYSTEM_PROMPT, MAX_CONTEXT_TOKENS, MAX_EVIDENCE_TOKENS,
                                deliver, messages, validate_policy, token_count,
                                first_expansion_required, enforce_context_budget, selection_controls)
from test_targeted_retrieval import run_tests

ROOT = Path(__file__).resolve().parent
TRACE_ID = 'val:SciBench_gpt-4o-mini_llm_debate_29_4444d10c'
STUDY_ID = 's4-mandatory-expansion-spans-20261011-v3'
PRIOR_ATTEMPT_ID = 's4-mandatory-expansion-spans-20261011-v2'
ARMS = ('full', 'short', 'short_targeted')
OUT = ROOT / 'reports' / STUDY_ID


def now():
    return datetime.now(timezone.utc).isoformat()


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def parse_candidate(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (TypeError, ValueError):
        return {}


def cache_key(config, arm):
    return f"{PROTOCOL_VERSION}:{arm}:{digest(stable(config))}"


def decode_response(response, selection_required):
    choice=response.choices[0]
    tool_calls=getattr(choice.message,'tool_calls',None) or []
    tool_log=[{'id':t.id,'name':t.function.name,'arguments':t.function.arguments} for t in tool_calls]
    raw=choice.message.content or ''
    if selection_required:
        if choice.finish_reason not in ('tool_calls','stop') or len(tool_calls)!=1 or tool_calls[0].function.name!='select_events':
            return raw,{'invalid_required_selection':True},tool_log
        args=parse_candidate(tool_calls[0].function.arguments)
        if not isinstance(args,dict) or set(args)!={'event_ids'} or not isinstance(args['event_ids'],list) or not args['event_ids']:
            return tool_calls[0].function.arguments,{'invalid_required_selection':True},tool_log
        return tool_calls[0].function.arguments,{'fault_claims':[],'retrieve_steps':args['event_ids']},tool_log
    return raw,parse_candidate(raw) if choice.finish_reason=='stop' else {'invalid_finish_reason':choice.finish_reason},tool_log


def usage_totals(calls):
    return {key: sum(c['usage'][key] for c in calls)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def decode_audit():
    """Audit earlier failures once-decoded, without repairing or normalizing quotes."""
    old = ROOT/'reports'/'s4-targeted-citations-20261011-v1'
    if not (old/'calls.json').exists():
        return {'status':'old_audit_unavailable'}
    calls = json.loads((old/'calls.json').read_text())
    deliveries = json.loads((old/'deliveries.json').read_text())
    records=[]
    for call in calls:
        delivery=next(d for d in deliveries if d['_id']==call['delivery_id'])
        for claim in call['candidate']['fault_claims']:
            for citation in claim['citations']:
                chunk=next(c for c in delivery['chunks'] if c['chunk_id']==citation['chunk_id'])
                records.append({'arm':call['arm'],'quote_type':type(citation['quote']).__name__,
                    'delivered_text_type':type(chunk['text']).__name__,
                    'decoded_exact_match':citation['quote'] in chunk['text'],
                    'contains_form_feed':'\f' in citation['quote'],
                    'json_roundtrip_preserves_delivered_text':json.loads(json.dumps(chunk['text']))==chunk['text']})
    return {'status':'verified','matching':'decoded strings; no fuzzy matching or extra unescaping',
            'findings':records}


def bounded_request(requested, available, expanded, remaining, policy):
    if not 0 < remaining <= MAX_ROUNDS or not isinstance(requested, list) or not requested:
        raise ValueError('No retrieval budget/request')
    if any(type(n) is not int for n in requested):
        raise ValueError('Step IDs must be integers')
    if len(requested) > policy['max_steps_per_round'] or len(requested) != len(set(requested)):
        raise ValueError('Invalid request count')
    if not set(requested) <= set(available) or set(requested) & set(expanded):
        raise ValueError('Unknown or already expanded step')
    return requested


def recover_completed_selector(db, stored_config, current_config):
    """Re-decode one completed tool response, preserving original evidence and API usage."""
    prior={k:v for k,v in stored_config.items() if k!='code_sha256'}
    current={k:v for k,v in current_config.items() if k!='code_sha256'}
    if prior!=current:
        raise RuntimeError('Recovery cannot change prompts, policy, source, gate version, or model settings')
    changed={name for name,value in current_config['code_sha256'].items() if stored_config['code_sha256'].get(name)!=value}
    if not changed <= {'run_s4_targeted.py','test_targeted_retrieval.py'}:
        raise RuntimeError('Only the response adapter and its regression tests may change during recovery')
    run_id=cache_key(stored_config,'short_targeted')
    run=db.citation_runs.find_one({'_id':run_id})
    if not run or run['status']!='rejected' or run['call_count']!=1:
        raise RuntimeError('Recovery requires one rejected initial selection call')
    call=db.citation_calls.find_one({'_id':run['final_call_id']})
    if call['status']!='complete' or call['round']!=0 or call['finish_reason']!='stop':
        raise RuntimeError('No completed stop-marked selector to recover')
    if db.citation_results.find_one({'_id':run_id}):
        raise RuntimeError('Never revise a published diagnosis')
    tools=[SimpleNamespace(id=t['id'],function=SimpleNamespace(name=t['name'],arguments=t['arguments'])) for t in call['tool_calls']]
    response=SimpleNamespace(choices=[SimpleNamespace(finish_reason=call['finish_reason'],
        message=SimpleNamespace(content=None,tool_calls=tools))])
    raw,candidate,_=decode_response(response,True)
    doc=db.citation_deliveries.find_one({'_id':call['delivery_id']})
    received=Delivery.from_document(doc)
    bounded_request(candidate.get('retrieve_steps'),[c.step for c in received.chunks if c.truncated],[],MAX_ROUNDS,stored_config['policy'])
    recovery={'reason':'Valid forced tool call returned finish_reason=stop; re-decoded without a model call.',
              'at':now(),'original_candidate':call['candidate'],'original_gate':call['gate'],
              'original_protocol_errors':call['protocol_errors'],
              'original_code_sha256':stored_config['code_sha256'],'recovery_code_sha256':current_config['code_sha256']}
    call.update(candidate=candidate,raw_response=raw,gate=check_claims(candidate,received,stored_config['known_agents']),
                protocol_errors=[],decoder_recovery=recovery)
    db.citation_calls.replace_one({'_id':call['_id']},call)
    run.update(status='running',decoder_recovery=recovery)
    db.citation_runs.replace_one({'_id':run_id},run)
    db.citation_studies.update_one({'_id':STUDY_ID},{'$set':{'status':'running','decoder_recovery':recovery}})
    print('Recovered saved model-selected events; no API call repeated.',flush=True)
    return stored_config


def run_arm(db, api, trace, events, config, arm):
    run_id = cache_key(config, arm)
    run = db.citation_runs.find_one({'_id': run_id})
    if run and run['status'] in ('accepted', 'rejected'):
        # Only this versioned protocol's audit data can be reused, and it is re-gated.
        final = db.citation_calls.find_one({'_id': run['final_call_id']})
        doc = db.citation_deliveries.find_one({'_id': final['delivery_id']})
        gate = publish_result(db.citation_results, run_id, final['candidate'],
                              Delivery.from_document(doc), config['known_agents'],
                              first_expansion_required=run['mandatory_first_expansion'],
                              actually_expanded=run['actually_expanded_steps'])
        if gate != run['gate']:
            raise RuntimeError('Cached gate result changed; refusing reuse')
        return {**run, 'cache_used': True}
    if not run:
        run = {'_id': run_id, 'study_id': STUDY_ID, 'trace_id': TRACE_ID, 'arm': arm,
               'session_id': uuid.uuid4().hex, 'status': 'running', 'created_at': now(),
               'config_hash': digest(stable(config)), 'cache_used': False}
        db.citation_runs.insert_one(run)
    session = run['session_id']
    policy = config['policy']
    budget = MAX_ROUNDS if arm == 'short_targeted' else 0
    model = config['model_settings']['model']
    delivery, initial_log = deliver(session, TRACE_ID, events, 'full' if arm == 'full' else 'short', policy=policy,model=model)
    initial_clipped = len(delivery.chunks) != len(events) or any(c.truncated for c in delivery.chunks)
    context_label = ('capped-large' if initial_clipped else 'full context') if arm == 'full' else ('short + mandatory first expansion' if arm == 'short_targeted' else 'short context')
    required = first_expansion_required(arm,delivery)
    expanded, actually_expanded, calls, retrievals = set(), set(), [], []
    for round_number in range(budget + 1):
        call_id = f'{run_id}:call{round_number}'
        delivery_id = f'{run_id}:delivery{round_number}'
        selection_required = required and not actually_expanded
        request_messages = messages(trace, delivery, budget - round_number, expanded, policy, selection_required)
        controls=selection_controls(delivery,policy) if selection_required else {}
        estimated_tokens = enforce_context_budget(request_messages,model,controls)
        doc = {**delivery.document(), '_id': delivery_id, 'study_id': STUDY_ID, 'run_id': run_id,
               'round': round_number, 'messages': request_messages,
               'messages_sha256': digest(stable(request_messages)),
               'initial_truncation_log': initial_log if round_number == 0 else [],
               'evidence_chars': len(delivery.evidence_text),
               'evidence_tokens':token_count(delivery.evidence_text,model),
               'context_tokens_with_framing_reserve':estimated_tokens,
               'context_token_budget':MAX_CONTEXT_TOKENS,
               'span_registration_stage':'after_final_clipping_and_input_budget_enforcement',
               'selection_required':selection_required}
        doc['request_controls']=controls
        prior = db.citation_deliveries.find_one({'_id': delivery_id})
        if prior is not None and prior != doc:
            raise RuntimeError('Saved evidence differs from reconstructed delivery')
        db.citation_deliveries.replace_one({'_id': delivery_id}, doc, upsert=True)
        call = db.citation_calls.find_one({'_id': call_id})
        if call and call['status'] != 'complete':
            raise RuntimeError('Earlier API attempt has no completed result; refusing duplicate call')
        if not call:
            if db.citation_calls.count_documents({'study_id': STUDY_ID}) >= 5:
                raise RuntimeError('Five-call comparison budget exhausted')
            call = {'_id': call_id, 'study_id': STUDY_ID, 'run_id': run_id, 'arm': arm,
                    'round': round_number, 'delivery_id': delivery_id, 'status': 'started',
                    'started_at': now(), 'request_settings': config['model_settings']}
            db.citation_calls.insert_one(call)
            print(f'CALL {arm} round={round_number}; evidence_chars={len(delivery.evidence_text)}', flush=True)
            started = time.monotonic()
            try:
                response = api.chat.completions.create(messages=request_messages, **config['model_settings'], **controls)
                elapsed = time.monotonic() - started
                raw,candidate,tool_log=decode_response(response,selection_required)
                usage = response.usage.model_dump()
                if not all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                    raise RuntimeError('Missing API token accounting')
                if usage['prompt_tokens'] > MAX_CONTEXT_TOKENS:
                    raise RuntimeError('Actual API input exceeded context-token budget')
                # Resolve spans ONLY from the received-evidence log saved for this API call.
                received = Delivery.from_document(db.citation_deliveries.find_one({'_id':delivery_id}))
                assert received.session_id == session and received.trace_id == TRACE_ID
                gate = check_claims(candidate, received, config['known_agents'])
                protocol_errors=[]
                if selection_required and (not isinstance(candidate,dict) or candidate.get('fault_claims') != [] or not candidate.get('retrieve_steps')):
                    protocol_errors.append('mandatory_first_expansion_selection_required')
                call.update(status='complete', completed_at=now(), raw_response=raw, candidate=candidate,
                            usage=usage, elapsed_seconds=round(elapsed, 3), gate=gate,
                            protocol_errors=protocol_errors,
                            tool_calls=tool_log,request_controls=controls,
                            returned_model=response.model, response_id=response.id,
                            system_fingerprint=response.system_fingerprint,
                            finish_reason=response.choices[0].finish_reason)
                # Local recovery log precedes database persistence; never silently retry a paid call.
                save(f"{arm}-call-{round_number}.json", call)
                db.citation_calls.replace_one({'_id': call_id}, call)
            except Exception as exc:
                db.citation_calls.update_one({'_id': call_id}, {'$set': {
                    'status': 'error', 'error_class': type(exc).__name__,
                    'http_status': getattr(exc, 'status_code', None), 'failed_at': now()}})
                raise
        candidate = call['candidate']
        if call['gate'] != check_claims(candidate, delivery, config['known_agents']):
            raise RuntimeError('Call gate mismatch')
        calls.append(call)
        if call.get('protocol_errors'):
            break
        requested = candidate.get('retrieve_steps', []) if isinstance(candidate, dict) else []
        if requested:
            try:
                available = [c.step for c in delivery.chunks if c.truncated]
                bounded_request(requested, available, expanded, budget-round_number, policy)
            except ValueError:
                break
            next_delivery, log = deliver(session, TRACE_ID, events, 'retrieve',
                                         previous=delivery.chunks, requested_steps=requested, policy=policy,model=model)
            delivered_steps=[x['step'] for x in log if x['status']=='delivered' and x['expanded_beyond_overview']]
            retrieval = {'_id': f'{run_id}:retrieval{round_number+1}', 'study_id': STUDY_ID,
                         'run_id': run_id, 'round': round_number+1, 'requested_steps': requested,
                         'selected_by': 'investigator_response', 'request_call_id': call_id,
                         'actually_delivered_steps':delivered_steps,
                         'log': log, 'before_chars': len(delivery.evidence_text),
                         'after_chars': len(next_delivery.evidence_text), 'retrieval_model_tokens': 0}
            db.citation_retrievals.replace_one({'_id': retrieval['_id']}, retrieval, upsert=True)
            retrievals.append(retrieval)
            print(f'RETRIEVE {arm}: model requested steps {requested}', flush=True)
            expanded.update(requested)
            actually_expanded.update(delivered_steps)
            delivery = next_delivery
            continue
        break
    # Mandatory final gate immediately precedes every accepted diagnosis write.
    received = Delivery.from_document(db.citation_deliveries.find_one({'_id':calls[-1]['delivery_id']}))
    gate = publish_result(db.citation_results, run_id, candidate, received, config['known_agents'],
                          first_expansion_required=required, actually_expanded=actually_expanded)
    predicted = sorted({c['agent_name'] for c in candidate.get('fault_claims', [])
                        if isinstance(c, dict) and isinstance(c.get('agent_name'), str)}) if isinstance(candidate, dict) and isinstance(candidate.get('fault_claims'), list) else []
    run.update(status='accepted' if gate['accepted'] else 'rejected', completed_at=now(),
               gate=gate, final_call_id=calls[-1]['_id'], candidate_predicted_agents=predicted,
               published_predicted_agents=predicted if gate['accepted'] else None,
               usage=usage_totals(calls), call_count=len(calls), retrieval_rounds=len(retrievals),
               requested_steps=[r['requested_steps'] for r in retrievals],
               actually_delivered_steps=[r['actually_delivered_steps'] for r in retrievals],
               actually_expanded_steps=sorted(actually_expanded), mandatory_first_expansion=required,
               context_label=context_label, initial_context_clipped=initial_clipped,
               retrieval_label='mandatory first expansion' if arm=='short_targeted' else 'fixed context',
               total_execution_seconds=round(sum(c['elapsed_seconds'] for c in calls), 3),
               context_chars_by_call=[db.citation_deliveries.find_one({'_id': c['delivery_id']})['evidence_chars'] for c in calls],
               returned_models=sorted({c['returned_model'] for c in calls}))
    db.citation_runs.replace_one({'_id': run_id}, run)
    print(json.dumps({'arm': arm, 'gate': run['status'], 'predictions': predicted,
                      'tokens_all_calls': run['usage'], 'retrieval_rounds': len(retrievals)}), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--policy-file', type=Path)
    parser.add_argument('--recover-completed-selector', action='store_true')
    args = parser.parse_args()
    tests = run_tests()
    decoding=decode_audit()
    if not tests['passed']:
        raise RuntimeError('Gate tests failed')
    policy = validate_policy(json.loads(args.policy_file.read_text()) if args.policy_file else DEFAULT_POLICY)
    print(f"Gate tests: {tests['tests_run']} passed.", flush=True)
    if not args.run:
        return
    load_dotenv(ROOT / '.env', override=True)
    uri = os.environ['MONGODB_URI']
    parsed = urlsplit(uri)
    if parsed.scheme != 'mongodb+srv' or not (parsed.hostname or '').endswith('.mongodb.net'):
        raise RuntimeError('Use the configured official Atlas sandbox SRV URI')
    options = {k.lower(): v for k,v in parse_qs(parsed.query).items()}
    if any(v.lower() == 'true' for k in ('tlsinsecure','tlsallowinvalidcertificates','tlsallowinvalidhostnames') for v in options.get(k, [])):
        raise RuntimeError('TLS verification is required')
    if os.getenv('OPENAI_BASE_URL') and urlsplit(os.environ['OPENAI_BASE_URL']).hostname != 'api.openai.com':
        raise RuntimeError('Unexpected investigator endpoint')
    def connect():
        return MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000,
                           connectTimeoutMS=10000, socketTimeoutMS=45000)
    with connect() as client:
        client.admin.command('ping')
        db = client[harness.DB_NAME]
        # No label read here. Only this S4 validation record is loaded.
        trace = db.traces.find_one({'_id': TRACE_ID}, {'_id':1,'task':1,'task_type':1,'history':1,'split':1})
        if not trace or trace['split'] != 'val':
            raise RuntimeError('S4 validation trace missing')
        events = [harness.event_details(e,i+1) for i,e in enumerate(trace['history'])]
        assert all(type(e['step']) is int for e in events) and len({e['step'] for e in events}) == len(events)
        known = sorted({e['agent_name'] for e in events if e['agent_name'].lower() not in {'user','system','unknown'}})
        source = {'trace_id':TRACE_ID, 'task':trace['task'], 'task_type':trace['task_type'], 'events':events}
        config = {
            'protocol':PROTOCOL_VERSION, 'gate_version':GATE_VERSION, 'trace_id':TRACE_ID,
            'source_sha256':digest(stable(source)), 'known_agents':known, 'policy':policy,
            'max_retrieval_rounds':MAX_ROUNDS, 'system_prompt_sha256':digest(SYSTEM_PROMPT),
            'retrieval_label':'mandatory first expansion', 'max_events_per_round':5,
            'initial_selection_control':'forced strict select_events function; IDs chosen by investigator',
            'context_token_budget':MAX_CONTEXT_TOKENS,'evidence_token_budget':MAX_EVIDENCE_TOKENS,
            'token_budget_note':'tiktoken for model text, plus 256 framing tokens; actual API input usage checked after every call',
            'model_settings':{'model':os.getenv('OPENAI_MODEL','gpt-4o-mini'), 'temperature':0,
                              'response_format':{'type':'json_object'}, 'max_completion_tokens':2000},
            'code_sha256': {name: digest((ROOT/name).read_text()) for name in
                           ['aperture.py','provenance_gate.py','targeted_retrieval.py','run_s4_targeted.py','test_targeted_retrieval.py']},
        }
        old = db.citation_studies.find_one({'_id':STUDY_ID})
        if old and old['config'] != config:
            if args.recover_completed_selector:
                config=recover_completed_selector(db,old['config'],config)
            else:
                raise RuntimeError('Study configuration changed; refusing to mix calls')
        if not old:
            db.citation_studies.insert_one({'_id':STUDY_ID,'status':'running','created_at':now(),
                                           'config':config,'max_total_model_calls':5})
        db.citation_sources.replace_one({'_id':STUDY_ID}, {'_id':STUDY_ID,**source}, upsert=True)
        db.citation_policies.replace_one({'_id':STUDY_ID}, {'_id':STUDY_ID,'policy':policy}, upsert=True)
        db.citation_tests.replace_one({'_id':STUDY_ID}, {'_id':STUDY_ID,**tests,'decoding_audit':decoding}, upsert=True)
        save('configuration.json',config)
        save('gate-tests.json',tests)
        save('decoding-audit.json',decoding)
        with harness.model_client().with_options(timeout=90, max_retries=0) as api:
            runs = [run_arm(db,api,trace,events,config,arm) for arm in ARMS]
        # All model calls have finished. Only now load evaluator-only labels.
        label = db.labels.find_one({'_id':TRACE_ID})
        actual = sorted(harness.normalize_names(label['faulty_agents']))
        evaluations = []
        for run in runs:
            f1, exact = harness.metrics(run['candidate_predicted_agents'], actual)
            evaluation = {'_id':run['_id'],'study_id':STUDY_ID,'trace_id':TRACE_ID,'arm':run['arm'],
                          'actual_agents':actual,'candidate_predicted_agents':run['candidate_predicted_agents'],
                          'candidate_f1':f1,'candidate_exact_match':exact,
                          'published_f1':f1 if run['gate']['accepted'] else None,
                          'published_exact_match':exact if run['gate']['accepted'] else None,
                          'false_positives':sorted(set(run['candidate_predicted_agents'])-set(actual)),
                          'false_negatives':sorted(set(actual)-set(run['candidate_predicted_agents']))}
            db.citation_evaluations.replace_one({'_id':evaluation['_id']},evaluation,upsert=True)
            evaluations.append(evaluation)
        calls = list(db.citation_calls.find({'study_id':STUDY_ID}))
        prior_calls=list(db.citation_calls.find({'study_id':PRIOR_ATTEMPT_ID}))
        prior_usage=usage_totals(prior_calls)
        deliveries = list(db.citation_deliveries.find({'study_id':STUDY_ID}))
        retrievals = list(db.citation_retrievals.find({'study_id':STUDY_ID}))
        assert len(calls) <= 5 and all(c['status']=='complete' for c in calls)
        assert len({c['returned_model'] for c in calls}) == 1
        by_delivery = {d['_id']:d for d in deliveries}
        for call in calls:
            d = by_delivery[call['delivery_id']]
            assert d['messages'][-1]['content'].endswith(d['evidence_text'])
            assert digest(stable(d['messages'])) == d['messages_sha256']
            assert check_claims(call['candidate'],Delivery.from_document(d),known) == call['gate']
            assert call['usage']['prompt_tokens'] <= MAX_CONTEXT_TOKENS
        # Post-run reporting probes only; these are NEVER used by retrieval or model prompts.
        probes = [(10,r'\frac{1}{12}'),(12,r'\frac{1}{4}')]
        visibility = []
        for run in runs:
            checked = []
            for step,formula in probes:
                source_formula=next(line for e in events if e['step']==step for line in e['content'].splitlines() if formula in line)
                rounds = sorted(d['round'] for d in deliveries if d['run_id']==run['_id'] and
                                any(c['step']==step and any(source_formula in s['text'] for s in c['spans']) for c in d['chunks']))
                checked.append({'step':step,'formula':source_formula,'delivered':bool(rounds),'first_call_round':rounds[0] if rounds else None})
            visibility.append({'arm':run['arm'],'formulas':checked})
        bundle = {'study_id':STUDY_ID,'trace_id':TRACE_ID,'config':config,'runs':runs,'evaluations':evaluations,
                  'formula_visibility':visibility,'gate_tests':tests,'decoding_audit':decoding,'all_calls_usage':usage_totals(calls),
                  'retrievals':retrievals,
                  'decoder_recovery':db.citation_studies.find_one({'_id':STUDY_ID}).get('decoder_recovery'),
                  'prior_attempt':{'study_id':PRIOR_ATTEMPT_ID,'calls':len(prior_calls),'usage':prior_usage,
                                   'reason':'JSON-only mandatory selection was ignored; host rejected premature diagnosis. Preserved, not reused.'},
                  'request_total_usage_including_prior_attempt':usage_totals(prior_calls+calls),
                  'request_total_calls_including_prior_attempt':len(prior_calls)+len(calls),
                  'total_model_calls':len(calls),'retrieval_model_calls':0,
                  'limitations':['Single S4 example, one session per arm; no general accuracy conclusion.',
                    'The gate verifies citation provenance, not whether a quotation logically proves the claim.',
                    'Mandatory first expansion is a fixed protocol rule, not learned retrieval. The investigator chooses events without label access.',
                    'All round inputs and outputs count toward usage, including provisional and rejected claims.',
                    'Prompt and evidence framing now require citations; these results are not identical-prompt reruns of the old baseline.'],
                  'persistence_verification':{'status':'pending'}}
        for filename,value in [('calls.json',calls),('deliveries.json',deliveries),('retrievals.json',retrievals),('comparison.json',bundle)]:
            save(filename,value)
        db.citation_studies.update_one({'_id':STUDY_ID},{'$set':{'status':'complete','completed_at':now(),
            'total_model_calls':len(calls),'all_calls_usage':usage_totals(calls),'formula_visibility':visibility}})
    # Independent connection verifies audit trail and accepted-only publication before report creation.
    with connect() as client:
        db = client[harness.DB_NAME]
        for collection, expected in [('citation_calls',len(calls)),('citation_deliveries',len(deliveries)),
                                     ('citation_retrievals',len(retrievals)),('citation_runs',3),('citation_evaluations',3)]:
            assert db[collection].count_documents({'study_id':STUDY_ID}) == expected
        accepted = sum(r['gate']['accepted'] for r in runs)
        ids=[r['_id'] for r in runs]
        assert db.citation_results.count_documents({'_id':{'$in':ids}}) == accepted
        for run in runs:
            final=db.citation_calls.find_one({'_id':run['final_call_id']})
            d=Delivery.from_document(db.citation_deliveries.find_one({'_id':final['delivery_id']}))
            fresh_gate=publish_result(db.citation_results,run['_id'],final['candidate'],d,known,
                first_expansion_required=run['mandatory_first_expansion'],actually_expanded=run['actually_expanded_steps'])
            assert fresh_gate == run['gate']
        bundle['persistence_verification']={'status':'verified','verified_at':now(),'database':harness.DB_NAME,
            'runs':3,'calls':len(calls),'deliveries':len(deliveries),'retrievals':len(retrievals),
            'evaluations':3,'accepted_published_results':accepted,'gate_tests':tests['tests_run']}
        db.citation_reports.replace_one({'_id':STUDY_ID},{'_id':STUDY_ID,**bundle},upsert=True)
        assert db.citation_reports.find_one({'_id':STUDY_ID}) == {'_id':STUDY_ID,**bundle}
    save('comparison.json',bundle)
    render_report(bundle,calls)
    print(json.dumps({'complete':True,'report':str(OUT/'report.html'),'calls':len(calls),
                      'usage':bundle['all_calls_usage'],'verification':bundle['persistence_verification']}),flush=True)


def render_report(bundle,calls):
    def table(headers,rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aperture S4 targeted retrieval</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;color:#182938;background:#f7fafb;margin:0}main{max-width:1220px;margin:auto;padding:36px 24px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:12px;border-bottom:1px solid #d9e1e6;text-align:left;vertical-align:top}th{background:#e6f0f3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border-left:3px solid #19868b}code{overflow-wrap:anywhere}details{padding:14px;background:#fff;margin:12px 0}h2{margin-top:32px}</style><main>',
        '<p>APERTURE · S4 CITED COMPARISON · 11 OCTOBER 2026</p><h1>Mandatory first expansion with exact span citations</h1>',
        '<p><code>'+TRACE_ID+'</code></p>',
        '<p>Same model, temperature 0, JSON response format, 2,000-token output cap, system prompt and citation requirements in all arms. Short context keeps 180 characters per event. Mandatory first expansion requires an investigator-selected expansion before targeted diagnosis when the overview is clipped. This is not learned retrieval. Limits: five events per round, two rounds; 12,000 input tokens including instructions and a 256-token framing reserve, of which at most 10,000 are evidence. The gate has no policy switch.</p>',
        '<p>The model returns chunk_id + span_id. Exact quotations are attached by application code from the received-evidence log registered after clipping and token-budget enforcement. Cross-author evidence is allowed. Provenance does not establish attribution correctness.</p>']
    rows=[]
    for r in bundle['runs']:
        e=next(x for x in bundle['evaluations'] if x['arm']==r['arm'])
        v=next(x for x in bundle['formula_visibility'] if x['arm']==r['arm'])
        rows.append([r['context_label'],', '.join(r['candidate_predicted_agents']) or 'none',e['candidate_f1'],e['candidate_exact_match'],r['status'],r['usage']['prompt_tokens'],r['usage']['completion_tokens'],r['usage']['total_tokens'],r['call_count'],r['retrieval_rounds'],str(r['requested_steps']),str(r['actually_delivered_steps']),'; '.join(f"step {x['step']}: {'yes, round '+str(x['first_call_round']) if x['delivered'] else 'no'}" for x in v['formulas'])])
    parts.append(table(['Context','Candidate predictions','F1','Exact','Gate','Input tokens','Output tokens','Total tokens','Calls','Retrieval rounds','Requested events','Actually expanded events','Omitted formulas delivered'],rows))
    parts.append('<p>Actual faulty agents (evaluation only): '+escape(', '.join(bundle['evaluations'][0]['actual_agents']))+'. A rejected candidate is retained for audit and scoring only; it is not published as an accepted diagnosis.</p>')
    parts.append('<h2>All calls, including retrieval decisions</h2>')
    parts.append(table(['Arm','Round','Input','Output','Total','Seconds','Provisional gate','Requested steps'],[[c['arm'],c['round'],c['usage']['prompt_tokens'],c['usage']['completion_tokens'],c['usage']['total_tokens'],c['elapsed_seconds'],c['gate']['accepted'],c['candidate'].get('retrieve_steps',[]) if isinstance(c['candidate'],dict) else []] for c in calls]))
    parts.append('<p>Total usage across every call: '+escape(str(bundle['all_calls_usage']))+'. Database retrieval itself makes zero model calls.</p>')
    parts.append('<p>Initial selection uses a forced strict select_events tool call; it uses the same model, temperature and output limit. All final diagnoses use identical span citations and the same provenance gate.</p>')
    parts.append('<p>Earlier implementation attempt retained for transparency: '+escape(str(bundle['prior_attempt']))+'. Total for this request, including that attempt: '+escape(str(bundle['request_total_usage_including_prior_attempt']))+' across '+str(bundle['request_total_calls_including_prior_attempt'])+' calls.</p>')
    if bundle.get('decoder_recovery'):
        parts.append('<p>Decoder recovery: '+escape(bundle['decoder_recovery']['reason'])+' Baseline and selection responses were preserved; no duplicate API calls were made for them.</p>')
    parts.append('<h2>Evidence actually delivered after clipping</h2>')
    parts.append(table(['Round','Requested events','Delivered expansions','Delivery details'],[[r['round'],r['requested_steps'],r['actually_delivered_steps'],json.dumps(r['log'])] for r in bundle['retrievals']]))
    parts.append('<h2>Final claims and citation audits</h2>')
    for r in bundle['runs']:
        final=next(c for c in calls if c['_id']==r['final_call_id'])
        parts.append('<details open><summary>'+escape(r['context_label']+' — '+r['status'])+'</summary><pre>'+escape(json.dumps(final['candidate'],indent=2,ensure_ascii=False))+'</pre><pre>'+escape(json.dumps(r['gate'],indent=2,ensure_ascii=False))+'</pre></details>')
    parts.append('<h2>Gate tests</h2>')
    parts.append(table(['Test','Result'],[[r['name'].split('.')[-1],r['status']] for r in bundle['gate_tests']['results']]))
    parts.append('<h2>Decoded-string verification</h2><pre>'+escape(json.dumps(bundle['decoding_audit'],indent=2))+'</pre>')
    parts.append('<h2>Limits and persistence</h2><ul>'+''.join('<li>'+escape(s)+'</li>' for s in bundle['limitations'])+'</ul>')
    parts.append('<pre>'+escape(json.dumps(bundle['persistence_verification'],indent=2))+'</pre>')
    parts.append('<p>Exact prompts and post-truncation chunks: <a href="deliveries.json">deliveries.json</a>. Other audit files: <a href="calls.json">calls</a> · <a href="retrievals.json">retrievals</a> · <a href="comparison.json">comparison</a> · <a href="gate-tests.json">tests</a> · <a href="configuration.json">settings and code hashes</a>.</p>')
    parts.append('<p>API usage fields follow the <a href="https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create">official OpenAI API reference</a>. Application cache keys include the protocol, source, settings, policy, prompt and code versions; the old uncited cache is never read.</p></main></html>')
    (OUT/'report.html').write_text('\n'.join(parts))


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_class':type(exc).__name__,
                          'http_status':getattr(exc,'status_code',None),'raw_error_redacted':True}),flush=True)
        raise SystemExit(1) from None
