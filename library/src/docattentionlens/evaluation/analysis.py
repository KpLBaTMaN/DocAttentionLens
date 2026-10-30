"""Descriptive evaluation, never a surrogate for independent research evidence."""
from collections import defaultdict
from statistics import mean, median
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def assess_answer(answer, reference, rubric, judgements=()):
    if rubric == 'human_adjudication_v1':
        verdicts = {item['verdict'] for item in judgements}
        if not verdicts:
            return {'verdict': 'pending', 'basis': 'human adjudication required', 'review_count': 0, 'disagreement': False}
        verdict = next(iter(verdicts)) if len(verdicts) == 1 else 'uncertain'
        return {'verdict': verdict, 'basis': 'submitted human judgements',
                'review_count': len(judgements), 'disagreement': len(verdicts) > 1}
    expected, actual = normalized(reference), normalized(answer)
    if rubric == 'boolean_strict_v1':
        # A trailing full stop is permitted. Explanations and aliases are not.
        expected, actual = expected.removesuffix('.'), actual.removesuffix('.')
        if expected not in ('true', 'false'):
            raise ValueError('Boolean rubric requires a true/false reference')
        matched = actual == expected
    elif rubric == 'normalized_exact_v1':
        if not expected:
            raise ValueError('Exact rubric requires a nonempty reference')
        matched = actual == expected
    else:
        raise ValueError('Unsupported answer rubric')
    return {'verdict': 'correct' if matched else 'incorrect',
            'basis': rubric, 'interpretation': 'rubric agreement, not semantic correctness'}


def assess_against_accepted(answer, accepted):
    """normalized_exact_v1 against each accepted answer of an extractive question; correct when any agrees."""
    if not accepted:
        raise ValueError('Exact rubric requires at least one accepted answer')
    matched = next((a for a in accepted if assess_answer(answer, a, 'normalized_exact_v1')['verdict'] == 'correct'), None)
    return {'verdict': 'correct' if matched is not None else 'incorrect', 'basis': 'normalized_exact_v1',
            'matched_answer': matched, 'accepted_answers': list(accepted),
            'interpretation': 'rubric agreement with an accepted answer, not semantic correctness'}


def answer_summary(rows):
    correct = sum(r['verdict'] == 'correct' for r in rows)
    scored = sum(r['verdict'] in ('correct', 'incorrect') for r in rows)
    return {'correct': correct, 'scored': scored,
            'pending': sum(r['verdict'] == 'pending' for r in rows),
            'uncertain': sum(r['verdict'] == 'uncertain' for r in rows),
            'agreement_rate': correct / scored if scored else None}


def result_measurements(result, frozen, reviews):
    rows = []
    for arm, outcome in zip(result.arms, result.outcomes, strict=True):
        if arm.name != outcome['arm']:
            raise ValueError('Result arm alignment differs')
        row = {'arm': arm.name, 'answer': outcome.get('answer')}
        if 'answer' in outcome:
            selected = [j for review in reviews for j in review['judgements'] if j['arm'] == arm.name]
            row['answer_assessment'] = assess_answer(outcome['answer'], frozen['reference_answer'],
                                                      frozen['rubric'], selected)
        else:
            row['answer_assessment'] = {'verdict': 'unavailable', 'basis': 'retrieval-only stage'}
        scores = outcome.get('scores')
        candidates = {frozen['changed_page_id'] if pid == arm.page.page_id else pid: score
                      for pid, score in (scores or {}).items()}
        support = set(frozen['support_page_ids'])
        eligible = support & set(candidates)
        row['designated_support_in_candidate_scope'] = len(eligible)
        row['designated_support_total'] = len(support)
        if scores is not None:
            ordered = sorted(candidates, key=lambda pid: (-candidates[pid], frozen['page_numbers'][pid], pid))
            top = set(ordered[:2])
            row['support_recall_at_2_within_candidates'] = len(top & eligible) / len(eligible) if eligible else None
            ranks = [ordered.index(pid) + 1 for pid in eligible]
            row['reciprocal_rank_within_candidates'] = 1 / min(ranks) if ranks else None
            row['changed_page_score'] = candidates.get(frozen['changed_page_id'])
        if 'supplied_page_ids' in outcome:
            supplied = {frozen['changed_page_id'] if pid == arm.page.page_id else pid
                        for pid in outcome['supplied_page_ids']}
            row['context_support_page_recall'] = len(supplied & support) / len(support) if support else None
            row['context_scope'] = 'page inclusion only; not element grounding or readability'
        rows.append(row)
    return rows


def repeat_summary(measurements):
    """Repetitions describe one frozen case, not independent sampled questions."""
    by_arm = defaultdict(list)
    for repetition in measurements:
        for arm in repetition:
            by_arm[arm['arm']].append(arm)
    result = []
    for arm, rows in by_arm.items():
        scores = [r['changed_page_score'] for r in rows if 'changed_page_score' in r]
        answers = [r['answer'] for r in rows if r['answer'] is not None]
        result.append({'arm': arm, 'repetitions_observed': len(rows),
                       'answer_text_consistent': len(set(answers)) == 1 if answers else None,
                       'changed_page_score_mean': mean(scores) if scores else None,
                       'changed_page_score_min': min(scores) if scores else None,
                       'changed_page_score_max': max(scores) if scores else None,
                       'answer_assessment': answer_summary([r['answer_assessment'] for r in rows
                                                            if r['answer_assessment']['verdict'] != 'unavailable'])})
    return result


def utility_summary(trials):
    groups = defaultdict(dict)
    for trial in trials:
        key = (trial['participant'], trial['protocol_id'], trial['task'])
        if trial['condition'] in groups[key]:
            raise ValueError('Duplicate participant/case/task/condition')
        groups[key][trial['condition']] = trial
    pairs = [g for g in groups.values() if len(g) == 2]
    differences = [p['docattentionlens']['elapsed_seconds'] - p['baseline']['elapsed_seconds'] for p in pairs]
    conditions = {}
    for condition in ('baseline', 'docattentionlens'):
        rows = [p[condition] for p in pairs]
        conditions[condition] = {'trials': len(rows), 'completed': sum(r['completed'] for r in rows)}
        for field in ('diagnostic_correct', 'reproduction_success', 'unsupported_causal_assertion'):
            measured = [r[field] for r in rows if r[field] is not None]
            conditions[condition][field] = {'positive': sum(measured), 'measured': len(measured)}
    return {'recorded_trials': len(trials), 'complete_pairs': len(pairs),
            'unpaired_groups': len(groups) - len(pairs),
            'participants': len({r['participant'] for r in trials}),
            'baseline_first_pairs': sum(p['baseline']['order'] == 1 for p in pairs),
            'linked_first_pairs': sum(p['docattentionlens']['order'] == 1 for p in pairs),
            'paired_time_difference_median_seconds': median(differences) if differences else None,
            'conditions': conditions,
            'interpretation': 'descriptive observer reports; no efficiency or population claim'}
