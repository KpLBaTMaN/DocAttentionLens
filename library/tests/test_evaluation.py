"""Known evaluation fixtures, not measurements from a model or participants."""
import pytest
from docattentionlens.evaluation.analysis import (
    assess_answer, answer_summary, utility_summary, result_measurements, repeat_summary,
)
from test_experiments import fixture
from docattentionlens.experiments.relayout import build_draft


def test_rubrics_keep_uncertainty_and_paraphrases_explicit():
    assert assess_answer(' TRUE. ', 'True', 'boolean_strict_v1')['verdict'] == 'correct'
    assert assess_answer('True because it increased', 'True', 'boolean_strict_v1')['verdict'] == 'incorrect'
    assert assess_answer('Yes', 'True', 'boolean_strict_v1')['verdict'] == 'incorrect'
    with pytest.raises(ValueError):
        assess_answer('False', 'Unknown', 'boolean_strict_v1')
    assert assess_answer('Ａ B\n C', 'a b c', 'normalized_exact_v1')['verdict'] == 'correct'
    assert assess_answer('£2 million', '2 million', 'normalized_exact_v1')['verdict'] == 'incorrect'
    assert assess_answer('anything', 'reference', 'human_adjudication_v1')['verdict'] == 'pending'
    disagreement = assess_answer('text', 'reference', 'human_adjudication_v1',
                                 [{'verdict': 'correct'}, {'verdict': 'incorrect'}])
    assert disagreement['disagreement'] and disagreement['verdict'] == 'uncertain'
    summary = answer_summary([{'verdict': x} for x in ['correct', 'incorrect', 'pending', 'uncertain']])
    assert summary == {'correct': 1, 'scored': 2, 'pending': 1, 'uncertain': 1, 'agreement_rate': .5}
    assert answer_summary([])['agreement_rate'] is None


def test_retrieval_candidate_denominator_and_context_use_explicit_lineage():
    req, case, page, raw, elements = fixture()
    result, _ = build_draft(req, case, page, raw, elements)
    result.outcomes = [{'arm': a.name, 'answer': 'False', 'scores': {a.page.page_id: -2, 'other': -3},
                        'supplied_page_ids': [a.page.page_id]} for a in result.arms]
    frozen = {'rubric': 'boolean_strict_v1', 'reference_answer': 'False',
              'changed_page_id': page.page_id, 'support_page_ids': [page.page_id, 'outside'],
              'page_numbers': {page.page_id: 1, 'other': 2}}
    rows = result_measurements(result, frozen, [])
    assert all(r['support_recall_at_2_within_candidates'] == 1 for r in rows)
    assert all(r['context_support_page_recall'] == .5 for r in rows)
    assert all(r['designated_support_total'] == 2 for r in rows)
    assert all(r['designated_support_in_candidate_scope'] == 1 for r in rows)
    summaries = repeat_summary([rows, rows])
    assert all(r['repetitions_observed'] == 2 and r['changed_page_score_min'] == -2 for r in summaries)
    frozen['support_page_ids'] = ['outside']
    assert result_measurements(result, frozen, [])[0]['support_recall_at_2_within_candidates'] is None


def test_utility_pairs_do_not_count_missing_measures_or_unpaired_trials():
    def trial(participant, condition, seconds, order):
        return {'participant': participant, 'protocol_id': 'p', 'task': 'reproduce_investigation',
                'condition': condition, 'elapsed_seconds': seconds, 'order': order,
                'completed': True, 'diagnostic_correct': None, 'reproduction_success': True,
                'unsupported_causal_assertion': False}
    baseline = trial('p1', 'baseline', 100, 1)
    linked = trial('p1', 'docattentionlens', 80, 2)
    unpaired = trial('p2', 'docattentionlens', 12, 1)
    summary = utility_summary([baseline, linked, unpaired])
    assert summary['complete_pairs'] == 1 and summary['unpaired_groups'] == 1
    assert summary['paired_time_difference_median_seconds'] == -20
    assert summary['conditions']['baseline']['diagnostic_correct']['measured'] == 0
    assert utility_summary([unpaired])['paired_time_difference_median_seconds'] is None
    with pytest.raises(ValueError):
        utility_summary([baseline, baseline])
