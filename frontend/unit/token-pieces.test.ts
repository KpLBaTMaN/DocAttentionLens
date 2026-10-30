import { describe, expect, it } from 'vitest';
import { decodePiece, displayPiece } from '../src/features/document/TokenStrip';
import { defaultSource, lastAnswerToken, lastQuestionWord } from '../src/features/document/useRetrieverAttention';

describe('readable token pieces', () => {
  it('decodes byte-level BPE pieces (Qwen) into their text', () => {
    expect(decodePiece('âĢĻs')).toBe('’s');       // E2 80 99: a right single quote
    expect(displayPiece('ĠMeeting')).toBe('Meeting'); // Ġ is a leading space
    expect(displayPiece('Ċ')).toBe('↵');
    expect(displayPiece('Ã©tÃ©')).toBe('été');
  });

  it('keeps SentencePiece, plain ASCII and incomplete byte runs as they are', () => {
    expect(displayPiece('▁patch')).toBe('patch');
    expect(decodePiece('café')).toBe('café');  // SentencePiece text: é alone is not valid UTF-8 bytes
    expect(decodePiece('âĢ')).toBe('âĢ');      // a character split across two tokens
    expect(decodePiece('<|im_start|>')).toBe('<|im_start|>');
  });
});

describe('default source for a Page + question reading', () => {
  const positions = [
    {index: 0, role: 'visual_patch'}, {index: 1, role: 'query_text', piece: 'ĠAnnual'},
    {index: 2, role: 'query_text', piece: 'ĠMeeting'}, {index: 3, role: 'query_text', piece: '?'}, {index: 4, role: 'visual_prompt', piece: '<|im_end|>'},
  ].map(p => ({...p, eligible: true}));
  const joint = {pass_kind: 'joint', causal: true, attention_mask: [1, 1, 1, 1, 1], positions};

  it('opens on the last question word, not its punctuation', () => {
    expect(defaultSource(joint)).toBe(2);
  });

  it('falls back to the last question token, then to the last active position', () => {
    expect(defaultSource({...joint, positions: positions.map(p => p.index === 1 || p.index === 2 ? {...p, piece: '…'} : p)})).toBe(3);
    expect(defaultSource({...joint, pass_kind: 'document'})).toBe(4);
  });
});

describe('Page + question with the reference answer', () => {
  const positions = [
    {index: 0, role: 'visual_patch'}, {index: 1, role: 'query_text', piece: 'ĠMeeting'}, {index: 2, role: 'query_text', piece: '?'},
    {index: 3, role: 'visual_prompt', piece: 'Answer'}, {index: 4, role: 'visual_prompt', piece: ':'},
    {index: 5, role: 'answer_text', piece: 'Ġ$'}, {index: 6, role: 'answer_text', piece: '4'}, {index: 7, role: 'answer_text', piece: '000'},
    {index: 8, role: 'visual_prompt', piece: '<|im_end|>'},
  ].map(p => ({...p, eligible: true}));
  const answered = {pass_kind: 'joint', causal: true, answer_source: 'reference', attention_mask: positions.map(() => 1), positions};

  it('reads "last answer token" at the position that predicts the final answer token', () => {
    expect(lastAnswerToken(answered)).toEqual({source: 6, target: positions[7]});
  });

  it('opens the answer reading on the last answer token, and the question-only reading on the last question word', () => {
    expect(defaultSource(answered)).toBe(6);
    expect(defaultSource({...answered, answer_source: null})).toBe(1);
    expect(lastQuestionWord(answered)).toBe(1);
  });

  it('has no last answer token without answer tokens', () => {
    expect(lastAnswerToken({...answered, positions: positions.filter(p => p.role !== 'answer_text')})).toBeUndefined();
  });
});
