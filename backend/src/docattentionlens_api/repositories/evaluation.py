"""Checksummed immutable study/review records and durable repetition membership."""
from hashlib import sha256
import json
import re
from psycopg import errors
from docattentionlens.datasets.adapter import canonical
from docattentionlens.evaluation.contracts import StudyRecord, EvaluationEvidence
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens_api.errors import fail


class EvaluationRepository:
    def __init__(self, jobs):
        self.jobs = jobs
        self.root = jobs.settings.artifact_root / 'evaluation'

    def path(self, identifier):
        if not re.fullmatch(r'(study|evaluation)_[a-f0-9]{32}', identifier):
            fail('not_found', 'Unknown evaluation record', 404)
        return self.root / (identifier + '.json')

    def write(self, record):
        content = record.model_dump(mode='json', exclude={'content_sha256'})
        digest = sha256(canonical(content)).hexdigest()
        content['content_sha256'] = digest
        identifier = record.study_id if isinstance(record, StudyRecord) else record.evidence_id
        CaptureStore.immutable(self.path(identifier), canonical(content))
        return digest

    def read(self, identifier, expected, model):
        try:
            content = json.loads(self.path(identifier).read_bytes())
            digest = sha256(canonical({k: v for k, v in content.items() if k != 'content_sha256'})).hexdigest()
            record = model.model_validate(content)
            identity = record.study_id if model is StudyRecord else record.evidence_id
            if digest != expected or content['content_sha256'] != expected or identity != identifier:
                raise ValueError('Evaluation integrity differs')
            return record
        except (OSError, ValueError, KeyError):
            fail('artifact_incomplete', 'Evaluation record failed integrity verification', 409)

    def get(self, identifier):
        with self.jobs.connect() as conn:
            row = conn.execute('SELECT * FROM evaluation_studies WHERE study_id=%s', (identifier,)).fetchone()
        if not row:
            fail('not_found', 'Unknown evaluation study', 404)
        return self.read(identifier, row['content_hash'], StudyRecord)

    def admit(self, record):
        digest = self.write(record)
        with self.jobs.connect() as conn:
            conn.execute('INSERT INTO evaluation_studies(study_id,name,content_hash) VALUES (%s,%s,%s)',
                         (record.study_id, record.specification.name, digest))
        return self.get(record.study_id)

    def list(self, limit, before):
        with self.jobs.connect() as conn:
            rows = conn.execute('SELECT study_id FROM evaluation_studies WHERE (%s::text IS NULL OR '
                                '(created_at,study_id)<(SELECT created_at,study_id FROM evaluation_studies WHERE study_id=%s)) '
                                'ORDER BY created_at DESC,study_id DESC LIMIT %s', (before, before, limit)).fetchall()
        return [self.get(r['study_id']) for r in rows]

    def runs(self, identifier):
        with self.jobs.connect() as conn:
            return conn.execute('SELECT r.*,j.state,j.output_resource_id FROM evaluation_runs r LEFT JOIN jobs j '
                                'ON r.job_id=j.job_id WHERE r.study_id=%s ORDER BY r.protocol_id,r.repetition',
                                (identifier,)).fetchall()

    def assign(self, study_id, protocol_id, repetition, job_id=None, result_id=None, retrospective=False):
        with self.jobs.connect() as conn:
            try:
                row = conn.execute('INSERT INTO evaluation_runs(study_id,protocol_id,repetition,job_id,result_id,retrospective) '
                                   'VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(study_id,protocol_id,repetition) DO NOTHING RETURNING *',
                                   (study_id, protocol_id, repetition, job_id, result_id, retrospective)).fetchone()
            except errors.UniqueViolation:
                fail('duplicate_observation', 'The same result cannot count as multiple repetitions', 409)
            if row is None:
                row = conn.execute('SELECT * FROM evaluation_runs WHERE study_id=%s AND protocol_id=%s AND repetition=%s',
                                   (study_id, protocol_id, repetition)).fetchone()
                if (row['job_id'], row['result_id']) != (job_id, result_id):
                    fail('repetition_conflict', 'This repetition already names another execution', 409)
        return row

    def evidence(self, study_id):
        with self.jobs.connect() as conn:
            rows = conn.execute('SELECT evidence_id,content_hash FROM evaluation_evidence WHERE study_id=%s '
                                'ORDER BY created_at,evidence_id', (study_id,)).fetchall()
        return [self.read(row['evidence_id'], row['content_hash'], EvaluationEvidence) for row in rows]

    def add_evidence(self, record):
        digest = self.write(record)
        payload = record.payload
        trial = record.record_kind == 'utility_trial'
        with self.jobs.connect() as conn:
            if trial:
                pair = '|'.join((record.study_id, payload.protocol_id, payload.participant, payload.task))
                conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (pair,))
                other = conn.execute('SELECT trial_order FROM evaluation_evidence WHERE study_id=%s AND protocol_id=%s '
                                     'AND participant=%s AND task=%s AND condition<>%s',
                                     (record.study_id, payload.protocol_id, payload.participant, payload.task, payload.condition)).fetchone()
                if other and other['trial_order'] == payload.order:
                    fail('invalid_order', 'Paired conditions must have distinct presentation orders', 422)
            try:
                conn.execute('INSERT INTO evaluation_evidence(evidence_id,study_id,record_kind,content_hash,result_id,reviewer,'
                             'protocol_id,participant,task,condition,trial_order) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                             (record.evidence_id, record.study_id, record.record_kind, digest,
                              None if trial else payload.result_id, None if trial else payload.reviewer,
                              payload.protocol_id if trial else None, payload.participant if trial else None,
                              payload.task if trial else None, payload.condition if trial else None,
                              payload.order if trial else None))
            except errors.UniqueViolation:
                fail('duplicate_evidence', 'This reviewer or participant condition already has an immutable record', 409)
        return self.read(record.evidence_id, digest, EvaluationEvidence)
