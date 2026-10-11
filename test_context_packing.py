"""Local packing regressions. Saved model selections are test fixtures only."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import targeted_retrieval as tr
import frozen_targeted_retrieval as old
from provenance_gate import Delivery, check_claims
import aperture

ROOT = Path('/Users/sree/aperture')


def fixture(study):
    folder = ROOT/'reports'/study
    bundle = json.loads((folder/'comparison.json').read_text())
    docs = json.loads((folder/'deliveries.json').read_text())
    run = next(r for r in bundle['runs'] if r['arm'] == 'short_targeted')
    initial = next(d for d in docs if d['run_id'] == run['_id'] and d['round'] == 0)
    request = bundle['retrievals'][0]['requested_steps']
    with (ROOT/'data/val.jsonl').open() as source_file:
        rows = source_file.readlines()
    for line in rows:
        row = json.loads(line)
        if 'val:'+row['id'] == bundle['trace_id']:
            data = aperture.as_dict(row['input'])
            trace = {'task': data['query'], 'task_type': aperture.as_dict(row['metadata'])['task_type']}
            events = [aperture.event_details(e, i+1) for i,e in enumerate(data['conversation_history'])]
            return bundle, trace, events, Delivery.from_document(initial), request
    raise AssertionError('Fixture missing')


class PackingTests(unittest.TestCase):
    def replay(self, study):
        b, trace, events, prior, requested = fixture(study)
        packed, log, audit = tr.pack_expansion(trace, prior.session_id, prior.trace_id, events, prior.chunks,
                                              requested, 1, set())
        msgs = tr.messages(trace, packed, 1, set(requested), tr.DEFAULT_POLICY, compact_timeline=True)
        self.assertEqual(tr.enforce_context_budget(msgs), audit['input_tokens_with_framing_reserve'])
        self.assertLessEqual(audit['evidence_tokens'], tr.MAX_EVIDENCE_TOKENS)
        self.assertEqual(len(packed.chunks), len({c.step for c in packed.chunks}))
        self.assertTrue(any(x['additional_chars'] > 0 for x in log))
        return b, events, prior, requested, packed, log, audit

    def test_l2_original_request_reproduces_overflow(self):
        p = ROOT/'reports/l2-frozen-mandatory-expansion-20261011-v1/blocked-context.json'
        request = json.loads(p.read_text())['unreceived_messages']
        self.assertEqual(old.context_token_estimate(request), 14030)
        with self.assertRaises(ValueError): old.enforce_context_budget(request)

    def test_l2_replacement_fits_and_no_stale_preview_citation(self):
        b,events,prior,requested,packed,log,audit = self.replay('l2-frozen-mandatory-expansion-20261011-v1')
        expanded = next(x['step'] for x in log if x['additional_chars'] > 0)
        original = next(c for c in prior.chunks if c.step == expanded)
        final = next(c for c in packed.chunks if c.step == expanded)
        self.assertNotIn(original.chunk_id, [c.chunk_id for c in packed.chunks])
        candidate = {'fault_claims':[{'agent_name':final.agent_name,'claim':'Test claim',
                     'citations':[{'chunk_id':final.chunk_id,'span_id':final.spans[0].span_id}]}], 'retrieve_steps':[]}
        self.assertTrue(check_claims(candidate,packed,b['config']['known_agents'])['accepted'])
        candidate['fault_claims'][0]['citations'] = [{'chunk_id':original.chunk_id,'span_id':original.spans[0].span_id}]
        self.assertFalse(check_claims(candidate,packed,b['config']['known_agents'])['accepted'])

    def test_s4_evidence_and_both_reported_formulas_survive(self):
        b,events,prior,requested,packed,log,audit = self.replay('s4-mandatory-expansion-spans-20261011-v3')
        self.assertEqual(audit['prepared_steps'],requested)
        probes = next(v for v in b['formula_visibility'] if v['arm']=='short_targeted')['formulas']
        for probe in probes:
            chunk = next(c for c in packed.chunks if c.step==probe['step'])
            self.assertTrue(any(probe['formula'] in s.text for s in chunk.spans))

    def test_deterministic_selection_order_and_partial_fitting(self):
        events = [{'step':n,'agent_name':'Worker','content':('text-%d ! , 漢字 ; '%n)*1000} for n in range(71,74)]
        trace = {'task':'Inspect','task_type':'unit'}
        prior,_ = tr.deliver('test','test:packing',events,'short')
        requested = [73,71,72]
        base = tr.context_token_estimate(tr.messages(trace,prior,1,set(requested),tr.DEFAULT_POLICY,compact_timeline=True))
        with patch.object(tr,'MAX_CONTEXT_TOKENS',base+600):
            first = tr.pack_expansion(trace,'test','test:packing',events,prior.chunks,requested,1,set())
            second = tr.pack_expansion(trace,'test','test:packing',events,prior.chunks,requested,1,set())
        self.assertEqual(first,second)
        self.assertEqual([x['step'] for x in first[1]],requested)
        self.assertTrue(any(x.get('packing_status')=='partial' for x in first[1]))
        self.assertTrue(first[2]['prepared_steps'])
        self.assertLessEqual(first[2]['input_tokens_with_framing_reserve'],base+600)

    def test_no_extra_source_text_reports_budget_exhaustion(self):
        events = [{'step':98,'agent_name':'Worker','content':'visible. '*100}]
        prior,_ = tr.deliver('test','test:packing',events,'short')
        trace = {'task':'task '*20000,'task_type':'unit'}
        result,log,audit = tr.pack_expansion(trace,'test','test:packing',events,prior.chunks,[98],1,set())
        self.assertEqual(audit['status'],'budget_exhausted')
        self.assertEqual(audit['prepared_steps'],[])
        self.assertEqual(log[0]['status'],'deferred')
        self.assertEqual(result,prior)

    def test_all_retained_messages_and_tool_schema_counted(self):
        small = [{'role':'system','content':'instructions'},{'role':'user','content':'evidence'}]
        extended = small + [{'role':'assistant','content':'history '*15000}]
        controls = {'tools':[{'schema':'tool '*15000}]}
        self.assertGreater(tr.context_token_estimate(extended),tr.MAX_CONTEXT_TOKENS)
        self.assertGreater(tr.context_token_estimate(small,controls=controls),tr.MAX_CONTEXT_TOKENS)
        with self.assertRaises(ValueError): tr.enforce_context_budget(extended)
        with self.assertRaises(ValueError): tr.enforce_context_budget(small,controls=controls)


if __name__=='__main__': unittest.main(verbosity=2)
