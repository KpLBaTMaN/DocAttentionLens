// Elements a researcher marks on uploaded pages: pure edits, automatic reading order and checks.
// Boxes are page-image pixels with a top-left origin, like every other element box in DocAttentionLens.
import type { components } from '../../api/generated/schema';

export type Mark = components['schemas']['MarkedElement'];
export type MarkingPage = components['schemas']['MarkingPage'];
export type MarkingDraft = components['schemas']['MarkingDraft'];
export type ElementType = Mark['element_type'];
export type MarkBox = {x1:number;y1:number;x2:number;y2:number};

/** The benchmark vocabulary. Keys 1–9 and 0 choose a type; page header and footer are chosen from the list. */
export const ELEMENT_TYPES:{type:ElementType;label:string;key?:string;definition:string}[] = [
  {type:'title',label:'Title',key:'1',definition:'The document’s own title, usually once, on its first page.'},
  {type:'heading',label:'Heading',key:'2',definition:'A line that names the section below it. Not the document title.'},
  {type:'text',label:'Text',key:'3',definition:'A paragraph of running text. One box per paragraph.'},
  {type:'listitem',label:'List item',key:'4',definition:'One bullet or numbered item. One box per item.'},
  {type:'table',label:'Table',key:'5',definition:'Values in rows and columns, including the header row. Give its title its own Caption box.'},
  {type:'figure',label:'Figure',key:'6',definition:'A chart, diagram, photo or drawing, including the labels inside it.'},
  {type:'caption',label:'Caption',key:'7',definition:'The text that names or explains a figure or table.'},
  {type:'footnote',label:'Footnote',key:'8',definition:'A note at the foot of the page, usually small and numbered.'},
  {type:'formula',label:'Formula',key:'9',definition:'A displayed equation on its own line.'},
  {type:'pagenumber',label:'Page number',key:'0',definition:'The page’s number on its own.'},
  {type:'pageheader',label:'Page header',definition:'Running text repeated at the top of pages, such as a report name.'},
  {type:'pagefooter',label:'Page footer',definition:'Running text repeated at the foot of pages, other than the page number.'},
];
export const typeInfo = (type:ElementType) => ELEMENT_TYPES.find(t => t.type === type)!;

/** What the draft saves: the marks, pages marked reviewed and pages whose order was set by hand. */
export type Marking = {marks:Mark[]; reviewed:string[]; manual:string[]};

export function newKey() {
  return 'm' + crypto.randomUUID().replaceAll('-', '').slice(0, 16);
}

export function normalise(box:MarkBox, page:{width:number;height:number}):MarkBox {
  const round = (v:number) => Math.round(v * 10) / 10;
  return {x1:round(Math.max(0, Math.min(box.x1, box.x2))), y1:round(Math.max(0, Math.min(box.y1, box.y2))),
    x2:round(Math.min(page.width, Math.max(box.x1, box.x2))), y2:round(Math.min(page.height, Math.max(box.y1, box.y2)))};
}

export const pageMarks = (marks:readonly Mark[], pageId:string) =>
  marks.filter(m => m.page_id === pageId).sort((a, b) => a.reading_order - b.reading_order);

/** Reading order by recursive XY-cut at the widest empty band; a row band wins a tie. */
export function xyCut<T extends {bbox:MarkBox}>(items:readonly T[]):T[] {
  if (items.length < 2) return [...items];
  const widest = (lo:'x1'|'y1', hi:'x2'|'y2') => {
    const sorted = [...items].sort((a, b) => a.bbox[lo] - b.bbox[lo]);
    let end = sorted[0].bbox[hi], cut = 0, gap = -Infinity;
    sorted.forEach((item, i) => {
      // Overlaps of a pixel or two are drawing noise, not a shared band.
      if (i && item.bbox[lo] >= end - 2 && item.bbox[lo] - end > gap) { gap = item.bbox[lo] - end; cut = i; }
      end = Math.max(end, item.bbox[hi]);
    });
    return {sorted, cut, gap};
  };
  const rows = widest('y1', 'y2'), columns = widest('x1', 'x2');
  const best = rows.cut && (!columns.cut || rows.gap >= columns.gap) ? rows : columns.cut ? columns : undefined;
  if (!best) return [...items].sort((a, b) => a.bbox.y1 - b.bbox.y1 || a.bbox.x1 - b.bbox.x1);
  return [...xyCut(best.sorted.slice(0, best.cut)), ...xyCut(best.sorted.slice(best.cut))];
}

/** Renumber one page: automatic pages follow XY-cut; hand-ordered pages keep their order without gaps. */
function orderPage(state:Marking, pageId:string, sequence?:Mark[]):Marking {
  const marks = pageMarks(state.marks, pageId);
  const ordered = sequence ?? (state.manual.includes(pageId) ? marks : xyCut(marks));
  const position = new Map(ordered.map((m, i) => [m.key, i + 1]));
  return {...state, marks:state.marks.map(m => m.page_id === pageId ? {...m, reading_order:position.get(m.key)!} : m)};
}

export function addMark(state:Marking, mark:Omit<Mark,'reading_order'>):Marking {
  const order = pageMarks(state.marks, mark.page_id).length + 1;
  return orderPage({...state, marks:[...state.marks, {...mark, reading_order:order}]}, mark.page_id);
}

export function updateMark(state:Marking, key:string, patch:Partial<Omit<Mark,'key'|'page_id'|'reading_order'>>):Marking {
  const mark = state.marks.find(m => m.key === key);
  if (!mark) return state;
  const next = {...state, marks:state.marks.map(m => m.key === key ? {...m, ...patch} : m)};
  return patch.bbox ? orderPage(next, mark.page_id) : next;
}

export function removeMark(state:Marking, key:string):Marking {
  const mark = state.marks.find(m => m.key === key);
  if (!mark) return state;
  return orderPage({...state, marks:state.marks.filter(m => m.key !== key)}, mark.page_id, pageMarks(state.marks, mark.page_id).filter(m => m.key !== key));
}

/** Put a mark at a zero-based position in its page's order; the page's order is then the researcher's. */
export function moveInOrder(state:Marking, key:string, index:number):Marking {
  const mark = state.marks.find(m => m.key === key);
  if (!mark) return state;
  const rest = pageMarks(state.marks, mark.page_id).filter(m => m.key !== key);
  rest.splice(Math.max(0, Math.min(index, rest.length)), 0, mark);
  const manual = state.manual.includes(mark.page_id) ? state.manual : [...state.manual, mark.page_id];
  return orderPage({...state, manual}, mark.page_id, rest);
}

export function automaticOrder(state:Marking, pageId:string):Marking {
  return orderPage({...state, manual:state.manual.filter(id => id !== pageId)}, pageId);
}

export function setReviewed(state:Marking, pageId:string, reviewed:boolean):Marking {
  const rest = state.reviewed.filter(id => id !== pageId);
  return {...state, reviewed:reviewed ? [...rest, pageId] : rest};
}

export type History = {past:Marking[]; present:Marking; future:Marking[]};
export const startHistory = (present:Marking):History => ({past:[], present, future:[]});
export function commit(history:History, next:Marking):History {
  return next === history.present ? history : {past:[...history.past.slice(-199), history.present], present:next, future:[]};
}
export function undo(history:History):History {
  const previous = history.past.at(-1);
  return previous ? {past:history.past.slice(0, -1), present:previous, future:[history.present, ...history.future]} : history;
}
export function redo(history:History):History {
  const next = history.future[0];
  return next ? {past:[...history.past, history.present], present:next, future:history.future.slice(1)} : history;
}

/** ColPali's 32 × 32 grid, the coarsest input in DocAttentionLens: a box smaller than one cell is "under 1 position". */
export const GRID = 32;
export function positionsCovered(box:MarkBox, page:{width:number;height:number}) {
  return (box.x2 - box.x1) * (box.y2 - box.y1) / (page.width / GRID * page.height / GRID);
}

export type Check = {level:'blocking'|'warning'; pageId:string; keys:string[]; message:string};

/** Problems to resolve (blocking) or consider (warning) before saving a version. */
export function checks(marks:readonly Mark[], pages:readonly MarkingPage[]):Check[] {
  const found:Check[] = [];
  for (const page of pages) {
    const own = pageMarks(marks, page.page_id);
    const name = (m:Mark) => `${m.reading_order} (${typeInfo(m.element_type).label.toLowerCase()})`;
    for (const m of own) {
      const b = m.bbox;
      if (b.x2 - b.x1 < 2 || b.y2 - b.y1 < 2) found.push({level:'blocking', pageId:page.page_id, keys:[m.key], message:`Page ${page.page_number}: element ${name(m)} has no area.`});
      else if (b.x1 < 0 || b.y1 < 0 || b.x2 > page.width || b.y2 > page.height) found.push({level:'blocking', pageId:page.page_id, keys:[m.key], message:`Page ${page.page_number}: element ${name(m)} extends beyond the page.`});
      else if (positionsCovered(b, page) < 1) found.push({level:'warning', pageId:page.page_id, keys:[m.key], message:`Page ${page.page_number}: element ${name(m)} is smaller than one input position on a 32 × 32 grid; the element list will flag it as under 1 position.`});
    }
    own.forEach((a, i) => own.slice(i + 1).forEach(b => {
      const w = Math.min(a.bbox.x2, b.bbox.x2) - Math.max(a.bbox.x1, b.bbox.x1), h = Math.min(a.bbox.y2, b.bbox.y2) - Math.max(a.bbox.y1, b.bbox.y1);
      if (w > 1 && h > 1) found.push({level:'warning', pageId:page.page_id, keys:[a.key, b.key], message:`Page ${page.page_number}: elements ${a.reading_order} and ${b.reading_order} overlap; attention on the shared area is split equally between them.`});
    }));
    if (!own.length) found.push({level:'warning', pageId:page.page_id, keys:[], message:`Page ${page.page_number} has no elements; attention on it will count as background.`});
  }
  return found;
}
