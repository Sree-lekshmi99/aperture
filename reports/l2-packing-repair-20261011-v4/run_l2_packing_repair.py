"""One targeted L2 run; prior baselines and failed targeted run are references only."""
import json,logging,os,sys,unittest
from pathlib import Path
from html import escape

ROOT=Path('/Users/sree/aperture')
sys.path.insert(0,str(ROOT))
import run_s4_targeted as runner
import targeted_retrieval as tr
import test_context_packing as packing_tests
from provenance_gate import Delivery,check_claims,publish_result,digest

TRACE_ID='val:smol+gaia_gpt-4o-mini_smoagents_23_9c7c5be9'
STUDY_ID='l2-packing-repair-20261011-v4'
OUT=ROOT/'reports'/STUDY_ID
PRIOR='l2-frozen-mandatory-expansion-20261011-v1'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def render(bundle,calls):
    def table(headers,rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aperture L2 packing repair</title>',
        '<style>body{font:16px/1.5 system-ui;background:#f7fafb;color:#182938;margin:0}main{max-width:1200px;margin:auto;padding:32px 24px}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;border-bottom:1px solid #ccd9df;text-align:left;vertical-align:top}th{background:#e6f0f3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}code{overflow-wrap:anywhere}</style><main><p>APERTURE · L2 ONLY · 11 OCTOBER 2026</p><h1>Targeted context packing repair</h1><p><code>'+TRACE_ID+'</code></p>',
        '<p>One new targeted run. Baseline rows and the previous failed targeted row are preserved historical results, not new calls. Model gpt-4o-mini, temperature 0, 2,000 output-token cap, original diagnosis instructions and provenance gate unchanged. Limits remain two rounds, five events per round, 4,000 characters per retrieved event, 12,000 total input tokens and 10,000 evidence tokens.</p>',
        '<p>The next request uses a compact timeline and one current chunk per event. Selected evidence replaces its preview; unselected previews are retained and counted. Requests are fitted in selection order, with partial/deferred status logged. All instructions, IDs, evidence, message/tool framing and retained messages are counted before sending. No expansion without additional source text can satisfy the mandatory-first-expansion rule.</p>']
    parts.append(table(['Version / arm','Prediction','Candidate F1','Exact','Gate/status','All-call tokens','New calls this request'],bundle['comparison_rows']))
    parts.append('<p>Actual faulty agents, evaluated after inference: '+escape(', '.join(bundle['actual_agents']))+'. Provenance verifies source references, not attribution correctness.</p>')
    parts.append('<h2>New calls</h2>')
    parts.append(table(['Round','Input','Output','Total','Requested events','Status'],[[c['round'],c.get('usage',{}).get('prompt_tokens'),c.get('usage',{}).get('completion_tokens'),c.get('usage',{}).get('total_tokens'),c.get('candidate',{}).get('retrieve_steps',[]),c['status']] for c in calls]))
    parts.append('<pre>'+escape(json.dumps(bundle['accounting'],indent=2))+'</pre><h2>Retrieval and delivery</h2>')
    for r in bundle['retrievals']:
        parts.append('<pre>'+escape(json.dumps({k:v for k,v in r.items() if k not in ('_id','run_id','request_call_id')},indent=2))+'</pre>')
    parts.append('<h2>Final candidate and gate</h2><pre>'+escape(json.dumps({'candidate':bundle.get('final_candidate'),'gate':bundle['run'].get('gate'),'failure':bundle.get('failure')},indent=2,ensure_ascii=False))+'</pre>')
    parts.append('<h2>Local reproduction and regressions</h2><pre>'+escape(json.dumps({'tests':bundle['tests'],'local_packing_replays':bundle['local_packing_replays']},indent=2))+'</pre>')
    parts.append('<h2>Preservation and persistence</h2><pre>'+escape(json.dumps({'manifest':bundle['implementation_manifest'],'verification':bundle['persistence_verification']},indent=2))+'</pre>')
    parts.append('<p>Audit: <a href="comparison.json">comparison</a> · <a href="calls.json">calls</a> · <a href="deliveries.json">exact delivered evidence</a> · <a href="retrievals.json">retrievals</a> · <a href="configuration.json">settings</a> · <a href="local-tests.json">tests and packing replays</a> · <a href="../'+PRIOR+'/report.html">previous frozen comparison</a>.</p><p>Single validation case. No broader experiment or dashboard changes.</p></main></html>')
    return '\n'.join(parts)


def main():
    prior=json.loads((ROOT/'reports'/PRIOR/'comparison.json').read_text())
    manifest=json.loads((OUT/'implementation-manifest.json').read_text())
    assert {n:digest((ROOT/n).read_text()) for n in manifest['after']}==manifest['after']
    assert digest(tr.SYSTEM_PROMPT)==prior['config']['system_prompt_sha256']
    assert manifest['unchanged_gate'] and manifest['unchanged_harness_and_scoring']
    base_tests=runner.run_tests()
    result=unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(packing_tests.PackingTests).run(result)
    tests={'passed':base_tests['passed'] and result.wasSuccessful(),'tests_run':base_tests['tests_run']+result.testsRun,
           'provenance_and_runner':base_tests,'packing_tests':{'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}}
    assert tests['passed']
    local=[]
    for study in [PRIOR,'s4-mandatory-expansion-spans-20261011-v3']:
        b,trace,events,delivery,selected=packing_tests.fixture(study)
        packed,log,audit=tr.pack_expansion(trace,delivery.session_id,delivery.trace_id,events,delivery.chunks,selected,1,set())
        local.append({'source_study':study,'trace_id':b['trace_id'],'selected_saved_fixture':selected,'packing':audit,'events':log,'model_calls':0})
    save('local-tests.json',{'tests':tests,'local_packing_replays':local,'original_L2_overflow_tokens':14030})
    print(json.dumps({'local_tests':tests['tests_run'],'passed':True,'packing_replays':local}),flush=True)
    runner.TRACE_ID,runner.STUDY_ID,runner.OUT=TRACE_ID,STUDY_ID,OUT
    runner.load_dotenv(ROOT/'.env',override=True)
    uri=os.environ['MONGODB_URI']
    assert runner.urlsplit(uri).scheme=='mongodb+srv' and runner.urlsplit(uri).hostname.endswith('.mongodb.net')
    options={k.lower():v for k,v in runner.parse_qs(runner.urlsplit(uri).query).items()}
    assert not any(v.lower()=='true' for k in ('tlsinsecure','tlsallowinvalidcertificates','tlsallowinvalidhostnames') for v in options.get(k,[]))
    assert not os.getenv('OPENAI_BASE_URL') or runner.urlsplit(os.environ['OPENAI_BASE_URL']).hostname=='api.openai.com'
    def connect(): return runner.MongoClient(uri,tlsCAFile=runner.certifi.where(),serverSelectionTimeoutMS=30000,connectTimeoutMS=10000,socketTimeoutMS=45000)
    with connect() as client:
        client.admin.command('ping'); db=client[runner.harness.DB_NAME]
        assert not db.citation_studies.find_one({'_id':STUDY_ID}), 'Never repeat this targeted run'
        prior_db_runs=list(db.citation_runs.find({'study_id':PRIOR}))
        prior_run_hashes={r['_id']:digest(runner.stable(r)) for r in prior_db_runs}
        trace=db.traces.find_one({'_id':TRACE_ID},{'_id':1,'task':1,'task_type':1,'history':1,'split':1})
        assert trace and trace['split']=='val'
        events=[runner.harness.event_details(e,i+1) for i,e in enumerate(trace['history'])]
        source={'trace_id':TRACE_ID,'task':trace['task'],'task_type':trace['task_type'],'events':events}
        assert digest(runner.stable(source))==prior['config']['source_sha256']
        config={**prior['config'],'protocol':tr.PROTOCOL_VERSION,'code_sha256':manifest['after'],
                'experiment_id':STUDY_ID,'packing':'compact timeline; replace previews; deterministic prefixes in selection order',
                'token_budget_note':'serialized full message/tool/response-format payload plus the existing 256-token reserve; verified before sending and against actual API usage'}
        assert config['model_settings']==prior['config']['model_settings'] and config['policy']==prior['config']['policy']
        assert config['gate_version']==runner.GATE_VERSION
        run_id=runner.cache_key(config,'short_targeted')
        assert not db.citation_runs.find_one({'_id':run_id})
        db.citation_studies.insert_one({'_id':STUDY_ID,'status':'running','config':config,'created_at':runner.now(),
            'max_total_model_calls':3,'baseline_reference_study':PRIOR,'implementation_manifest':manifest})
        db.citation_sources.insert_one({'_id':STUDY_ID,**source})
        db.citation_policies.insert_one({'_id':STUDY_ID,'policy':config['policy']})
        db.citation_tests.insert_one({'_id':STUDY_ID,**tests,'local_packing_replays':local,'original_L2_overflow_tokens':14030})
        save('configuration.json',config)
        failure=None
        try:
            with runner.harness.model_client().with_options(timeout=90,max_retries=0) as api:
                run=runner.run_arm(db,api,trace,events,config,'short_targeted')
        except Exception as exc:
            failure={'error_class':type(exc).__name__,'http_status':getattr(exc,'status_code',None),'raw_error_redacted':True}
            db.citation_runs.update_one({'_id':run_id},{'$set':{'status':'error','failure':failure}})
            run=db.citation_runs.find_one({'_id':run_id})
        calls=list(db.citation_calls.find({'study_id':STUDY_ID}))
        deliveries=list(db.citation_deliveries.find({'study_id':STUDY_ID}))
        retrievals=list(db.citation_retrievals.find({'study_id':STUDY_ID}))
        assert len(calls)<=3
        # All new API calls finished; now, and only now, fetch evaluator labels.
        label=db.labels.find_one({'_id':TRACE_ID})
        actual=sorted(runner.harness.normalize_names(label['faulty_agents']))
        predictions=run.get('candidate_predicted_agents')
        f1,exact=runner.harness.metrics(predictions,actual) if predictions is not None else (None,None)
        ev={'_id':run_id,'study_id':STUDY_ID,'trace_id':TRACE_ID,'arm':'short_targeted','actual_agents':actual,
            'candidate_predicted_agents':predictions,'candidate_f1':f1,'candidate_exact_match':exact,
            'published_f1':f1 if run.get('gate',{}).get('accepted') else None,
            'published_exact_match':exact if run.get('gate',{}).get('accepted') else None}
        db.citation_evaluations.insert_one(ev)
        metered=[c for c in calls if 'usage' in c]
        usage=runner.usage_totals(metered)
        old_usage=prior['accounting']['all_calls_usage']
        accounting={'new_calls':len(calls),'new_usage':usage,'unmetered_new_attempts':[c['_id'] for c in calls if 'usage' not in c],
            'previous_baseline_calls':2,'previous_baseline_usage':prior['accounting']['accepted_arm_usage'],
            'previous_failed_targeted_calls':1,'previous_failed_targeted_usage':prior['accounting']['failed_or_rejected_arm_usage'],
            'all_L2_versions_calls':len(calls)+prior['accounting']['total_api_attempts'],
            'all_L2_versions_usage':{k:usage[k]+old_usage[k] for k in usage},'automatic_retries':0,'new_targeted_runs':1,
            'new_failed_attempts':failure,'new_gate_rejected':run.get('status')=='rejected'}
        rows=[]
        for old_run in prior['runs']:
            old_ev=next(e for e in prior['evaluations'] if e['arm']==old_run['arm'])
            rows.append(['v3 / '+old_run.get('context_label',old_run['arm']),old_run.get('candidate_predicted_agents'),old_ev['candidate_f1'],old_ev['candidate_exact_match'],old_run['status'],old_run.get('usage',{}).get('total_tokens'),0])
        rows.append(['v4 packing / targeted',predictions,f1,exact,run['status'],usage['total_tokens'],len(calls)])
        final=next((c for c in calls if c['_id']==run.get('final_call_id')),None)
        bundle={'study_id':STUDY_ID,'trace_id':TRACE_ID,'config':config,'run':run,'evaluation':ev,'actual_agents':actual,
            'accounting':accounting,'retrievals':retrievals,'failure':failure,'final_candidate':final['candidate'] if final else None,
            'comparison_rows':rows,'baseline_reference_study':PRIOR,'prior_run_hashes':prior_run_hashes,
            'tests':tests,'local_packing_replays':local,'implementation_manifest':manifest,'labels_read_after_all_calls':True,
            'persistence_verification':{'status':'pending'}}
        db.citation_studies.update_one({'_id':STUDY_ID},{'$set':{'status':'complete' if not failure else 'completed_with_failure','completed_at':runner.now(),'accounting':accounting}})
        for name,value in [('calls.json',calls),('deliveries.json',deliveries),('retrievals.json',retrievals),('comparison.json',bundle)]:save(name,value)
    with connect() as client:
        db=client[runner.harness.DB_NAME]
        assert {r['_id']:digest(runner.stable(r)) for r in db.citation_runs.find({'study_id':PRIOR})}==prior_run_hashes
        assert {n:digest((ROOT/n).read_text()) for n in manifest['after']}==manifest['after']
        for call in calls:
            if call['status']!='complete':continue
            doc=db.citation_deliveries.find_one({'_id':call['delivery_id']})
            d=Delivery.from_document(doc)
            assert check_claims(call['candidate'],d,config['known_agents'])==call['gate']
            assert tr.enforce_context_budget(doc['messages'],config['model_settings']['model'],doc['request_controls'])<=tr.MAX_CONTEXT_TOKENS
            assert call['usage']['prompt_tokens']<=tr.MAX_CONTEXT_TOKENS
            assert len(d.chunks)==len({c.step for c in d.chunks})
        if final:
            d=Delivery.from_document(db.citation_deliveries.find_one({'_id':final['delivery_id']}))
            assert publish_result(db.citation_results,run_id,final['candidate'],d,config['known_agents'],
                first_expansion_required=run['mandatory_first_expansion'],actually_expanded=run['actually_expanded_steps'])==run['gate']
        counts={'citation_runs':1,'citation_calls':len(calls),'citation_deliveries':len(deliveries),'citation_retrievals':len(retrievals),'citation_evaluations':1}
        for name,count in counts.items(): assert db[name].count_documents({'study_id':STUDY_ID})==count
        bundle['persistence_verification']={'status':'verified','at':runner.now(),'database':runner.harness.DB_NAME,
            **counts,'previous_runs_unchanged':True,'implementation_matches_manifest':True}
        html=render(bundle,calls)
        report_doc={'_id':STUDY_ID,**bundle,'html':html,'html_sha256':digest(html)}
        db.citation_reports.insert_one(report_doc)
        assert db.citation_reports.find_one({'_id':STUDY_ID})==report_doc
    save('comparison.json',bundle)
    (OUT/'report.html').write_text(html)
    (OUT/'run_l2_packing_repair.py').write_text(Path(__file__).read_text())
    print(json.dumps({'complete':True,'evaluation':ev,'accounting':accounting,'gate':run.get('gate'),'retrievals':retrievals,'report':str(OUT/'report.html')},indent=2),flush=True)


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_class':type(exc).__name__,'raw_error_redacted':True}),flush=True)
        raise SystemExit(1) from None
