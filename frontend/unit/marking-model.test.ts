import { describe, expect, it } from 'vitest';
import {
  addMark, automaticOrder, checks, commit, moveInOrder, normalise, pageMarks, positionsCovered, redo, removeMark, setReviewed,
  startHistory, undo, updateMark, xyCut, type Mark, type MarkBox, type Marking, type MarkingPage,
} from '../src/features/marking/markingModel';

const PAGE:MarkingPage = {page_id:'p1', page_number:1, width:640, height:640, image_url:'/p1.png', text_layer:true};
const OTHER:MarkingPage = {...PAGE, page_id:'p2', page_number:2};
const box = (x1:number, y1:number, x2:number, y2:number):MarkBox => ({x1, y1, x2, y2});
const mark = (key:string, bbox:MarkBox, page = 'p1'):Omit<Mark,'reading_order'> => ({key, page_id:page, element_type:'text', bbox, text:null, text_source:'none'});
const EMPTY:Marking = {marks:[], reviewed:[], manual:[]};
const keys = (state:Marking, page = 'p1') => pageMarks(state.marks, page).map(m => m.key);

describe('automatic reading order', () => {
  it('reads a full-width title, then each column top to bottom', () => {
    // Two columns whose paragraphs share a gap at y 300: columns must still be read one at a time.
    const items = [{key:'right-low', bbox:box(340, 320, 600, 500)}, {key:'left-high', bbox:box(40, 100, 300, 290)},
      {key:'title', bbox:box(40, 20, 600, 60)}, {key:'left-low', bbox:box(40, 320, 300, 500)}, {key:'right-high', bbox:box(340, 100, 600, 290)}];
    expect(xyCut(items).map(i => i.key)).toEqual(['title', 'left-high', 'left-low', 'right-high', 'right-low']);
  });

  it('falls back to top to bottom, then left to right, when no gap separates boxes', () => {
    const items = [{key:'b', bbox:box(50, 10, 200, 100)}, {key:'a', bbox:box(0, 0, 100, 60)}];
    expect(xyCut(items).map(i => i.key)).toEqual(['a', 'b']);
  });

  it('renumbers a page as boxes are added, moved and removed', () => {
    let state = addMark(EMPTY, mark('low', box(10, 400, 600, 450)));
    state = addMark(state, mark('high', box(10, 50, 600, 100)));
    state = addMark(state, mark('elsewhere', box(10, 10, 50, 50), 'p2'));
    expect(keys(state)).toEqual(['high', 'low']);
    expect(pageMarks(state.marks, 'p1').map(m => m.reading_order)).toEqual([1, 2]);
    state = updateMark(state, 'low', {bbox:box(10, 5, 600, 40)});
    expect(keys(state)).toEqual(['low', 'high']);
    state = removeMark(state, 'low');
    expect(pageMarks(state.marks, 'p1').map(m => [m.key, m.reading_order])).toEqual([['high', 1]]);
    expect(pageMarks(state.marks, 'p2')[0].reading_order).toBe(1);
  });
});

describe('order set by hand', () => {
  const three = [mark('a', box(10, 10, 600, 50)), mark('b', box(10, 100, 600, 150)), mark('c', box(10, 200, 600, 250))].reduce(addMark, EMPTY);

  it('keeps a moved order, appends new boxes and returns to automatic order on request', () => {
    let state = moveInOrder(three, 'c', 0);
    expect(keys(state)).toEqual(['c', 'a', 'b']);
    expect(state.manual).toEqual(['p1']);
    state = addMark(state, mark('d', box(10, 0, 600, 5)));
    expect(keys(state)).toEqual(['c', 'a', 'b', 'd']);
    state = updateMark(state, 'a', {bbox:box(10, 300, 600, 350)});
    expect(keys(state)).toEqual(['c', 'a', 'b', 'd']);
    state = automaticOrder(state, 'p1');
    expect(keys(state)).toEqual(['d', 'b', 'c', 'a']);
    expect(state.manual).toEqual([]);
  });

  it('clamps a move beyond either end', () => {
    expect(keys(moveInOrder(three, 'a', 9))).toEqual(['b', 'c', 'a']);
    expect(keys(moveInOrder(three, 'c', -2))).toEqual(['c', 'a', 'b']);
  });
});

describe('history', () => {
  it('undoes and redoes whole states and drops the redo branch on a new change', () => {
    let history = startHistory(EMPTY);
    history = commit(history, addMark(history.present, mark('a', box(10, 10, 100, 100))));
    history = commit(history, setReviewed(history.present, 'p1', true));
    expect(history.present.reviewed).toEqual(['p1']);
    history = undo(history);
    expect(history.present.reviewed).toEqual([]);
    expect(undo(undo(history)).present).toBe(EMPTY);
    history = redo(history);
    expect(history.present.reviewed).toEqual(['p1']);
    history = commit(undo(history), removeMark(history.present, 'a'));
    expect(redo(history)).toBe(history);
    expect(commit(history, history.present)).toBe(history);
  });
});

describe('checks', () => {
  it('blocks boxes without area or beyond the page and warns about overlaps, tiny boxes and empty pages', () => {
    let state = addMark(EMPTY, mark('a', box(10, 10, 300, 200)));
    state = addMark(state, mark('b', box(200, 150, 400, 300)));
    state = addMark(state, mark('tiny', box(500, 600, 510, 610)));
    const found = checks([...state.marks, {...state.marks[0], key:'flat', bbox:box(10, 400, 300, 401), reading_order:4},
      {...state.marks[0], key:'outside', bbox:box(600, 400, 700, 450), reading_order:5}], [PAGE, OTHER]);
    expect(found.filter(c => c.level === 'blocking').map(c => c.keys)).toEqual([['flat'], ['outside']]);
    expect(found.find(c => c.keys.includes('tiny'))?.message).toMatch(/smaller than one input position/);
    expect(found.find(c => c.keys.length === 2)?.message).toBe('Page 1: elements 1 and 2 overlap; attention on the shared area is split equally between them.');
    expect(found.find(c => c.pageId === 'p2')?.message).toBe('Page 2 has no elements; attention on it will count as background.');
  });

  it('measures boxes in positions of a 32 × 32 grid', () => {
    expect(positionsCovered(box(0, 0, 20, 20), PAGE)).toBe(1);
    expect(positionsCovered(box(0, 0, 640, 640), PAGE)).toBe(1024);
  });

  it('normalises drawn boxes to the page in any drag direction', () => {
    expect(normalise(box(700.04, 50.06, -5, 10), PAGE)).toEqual(box(0, 10, 640, 50.1));
  });
});
