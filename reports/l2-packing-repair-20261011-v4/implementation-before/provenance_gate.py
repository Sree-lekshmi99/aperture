"""Host-enforced provenance checks. No editable policy or model can disable this gate."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import re

GATE_VERSION = 'decoded-delivered-spans-v2'


def digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Span:
    span_id: str
    start: int
    end: int
    text: str


def make_spans(chunk_id, text):
    """Partition already-clipped, decoded text. Never look up unseen source text."""
    spans = []
    boundaries = [0] + [m.end() for m in re.finditer(r'\n[ \t]*\n', text)] + [len(text)]
    for start, end in zip(boundaries, boundaries[1:]):
        for offset in range(start, end, 600):
            stop = min(offset + 600, end)
            quote = text[offset:stop]
            if quote.strip():
                spans.append(Span(f'{chunk_id}:p{len(spans)+1}', offset, stop, quote))
    return tuple(spans)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    session_id: str
    trace_id: str
    step: int
    agent_name: str
    text: str
    source_chars: int
    source_sha256: str
    truncated: bool
    spans: tuple[Span, ...]


def make_chunk(session_id, trace_id, event, text):
    chunk_id = f"{session_id}:s{event['step']}:{digest(text)[:12]}"
    return Chunk(chunk_id, session_id,
                 trace_id, event['step'], event['agent_name'], text,
                 len(event['content']), digest(event['content']), text != event['content'],
                 make_spans(chunk_id, text))


def render_chunks(chunks):
    return '\n\n'.join(
        f'[CHUNK {c.chunk_id} | event_id={c.step} | author={c.agent_name} | clipped={str(c.truncated).lower()}]\n'
        + ''.join(f'[SPAN {s.span_id}]\n{s.text}\n[END SPAN]\n' for s in c.spans)
        + (f'[{c.source_chars-len(c.text):,} characters omitted]\n' if c.truncated else '')
        + '[END CHUNK]' for c in chunks)


@dataclass(frozen=True)
class Delivery:
    session_id: str
    trace_id: str
    chunks: tuple[Chunk, ...]
    evidence_text: str

    def document(self):
        doc = asdict(self)
        doc['chunks'] = [{**asdict(c), 'spans': [asdict(s) for s in c.spans]} for c in self.chunks]
        doc['evidence_sha256'] = digest(self.evidence_text)
        return doc

    @classmethod
    def from_document(cls, doc):
        if digest(doc['evidence_text']) != doc['evidence_sha256']:
            raise ValueError('Delivery digest mismatch')
        chunks = tuple(Chunk(**{**c, 'spans': tuple(Span(**s) for s in c['spans'])}) for c in doc['chunks'])
        return cls(doc['session_id'], doc['trace_id'], chunks, doc['evidence_text'])


def check_claims(candidate, delivery, known_agents):
    """Validate every citation against only the current, post-truncation delivery."""
    errors, checked, resolved_claims = [], [], []
    if render_chunks(delivery.chunks) != delivery.evidence_text:
        errors.append('delivery_text_mismatch')
    chunks = {}
    for c in delivery.chunks:
        if c.session_id != delivery.session_id or c.trace_id != delivery.trace_id:
            errors.append('delivery_scope_mismatch')
        if c.chunk_id != f'{delivery.session_id}:s{c.step}:{digest(c.text)[:12]}':
            errors.append('chunk_identity_mismatch')
        if c.chunk_id in chunks:
            errors.append('duplicate_chunk_id')
        if c.spans != make_spans(c.chunk_id, c.text):
            errors.append('delivered_span_integrity_mismatch')
        chunks[c.chunk_id] = c
    if not isinstance(candidate, dict) or set(candidate) != {'fault_claims', 'retrieve_steps'}:
        errors.append('response_schema')
        candidate = {}
    claims = candidate.get('fault_claims')
    requests = candidate.get('retrieve_steps')
    if not isinstance(claims, list) or not isinstance(requests, list):
        errors.append('response_lists_required')
        claims = []
    for index, claim in enumerate(claims):
        prefix = f'claim_{index}'
        if not isinstance(claim, dict) or set(claim) != {'agent_name', 'claim', 'citations'}:
            errors.append(prefix + ':schema')
            continue
        agent = claim['agent_name']
        if not isinstance(agent, str) or agent not in known_agents:
            errors.append(prefix + ':unknown_agent')
        if not isinstance(claim['claim'], str) or not claim['claim'].strip():
            errors.append(prefix + ':empty_claim')
        citations = claim['citations']
        if not isinstance(citations, list) or not citations:
            errors.append(prefix + ':citations_required')
            continue
        resolved = []
        for n, citation in enumerate(citations):
            reason = None
            chunk = None
            span = None
            if not isinstance(citation, dict) or set(citation) != {'chunk_id', 'span_id'}:
                reason = 'citation_schema'
            elif not isinstance(citation['chunk_id'], str):
                reason = 'invalid_chunk_id'
            elif not (chunk := chunks.get(citation['chunk_id'])):
                reason = 'unknown_or_undelivered_chunk'
            elif not isinstance(citation['span_id'], str):
                reason = 'invalid_span_id'
            elif not (span := next((s for s in chunk.spans if s.span_id == citation['span_id']), None)):
                reason = 'unknown_or_undelivered_span'
            elif not span.text.strip():
                reason = 'empty_quote'
            elif (span.text != chunk.text[span.start:span.end] or span.text not in delivery.evidence_text):
                reason = 'quote_not_verbatim_in_delivered_chunk'
            if reason:
                errors.append(f'{prefix}:citation_{n}:{reason}')
            else:
                resolved.append({**citation, 'quote': span.text, 'start': span.start, 'end': span.end,
                                 'source_author': chunk.agent_name})
            checked.append({'claim_index': index, 'citation_index': n,
                            'accepted': reason is None, 'reason': reason})
        resolved_claims.append({**claim, 'citations': resolved})
    return {'version': GATE_VERSION, 'accepted': not errors, 'errors': errors,
            'claims_checked': len(claims), 'citations_checked': checked,
            'session_id': delivery.session_id, 'evidence_sha256': digest(delivery.evidence_text),
            'resolved_claims': resolved_claims}


def publish_result(collection, result_id, candidate, delivery, known_agents, *,
                   first_expansion_required=False, actually_expanded=()):
    """The sole publication path: fail closed and publish no rejected fault claims."""
    gate = check_claims(candidate, delivery, known_agents)
    gate['provenance_accepted'] = gate['accepted']
    gate['protocol_errors'] = []
    if first_expansion_required and not actually_expanded:
        gate['accepted'] = False
        gate['protocol_errors'].append('mandatory_first_expansion_not_delivered')
    if isinstance(candidate, dict) and candidate.get('retrieve_steps'):
        gate['accepted'] = False
        gate['protocol_errors'].append('retrieval_not_final')
    if gate['accepted']:
        collection.replace_one({'_id': result_id}, {
            '_id': result_id, 'session_id': delivery.session_id, 'trace_id': delivery.trace_id,
            'gate': gate, 'fault_claims': gate['resolved_claims'],
            'predicted_agents': sorted({c['agent_name'] for c in candidate['fault_claims']}),
        }, upsert=True)
    return gate
