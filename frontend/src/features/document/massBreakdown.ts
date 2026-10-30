/** Where a row's mass goes, grouped by the role of each target position. */
export type MassGroup = {
  key: string; label: string; mass: number; positions: number;
  /** The single target with the most mass in the group, when it is a token. */
  top?: {index: number; piece: string; mass: number};
};

type Target = {index: number; role: string; piece?: string; page_id?: string | null; bbox?: number[] | null};

// Generator roles and retriever encoding roles, in reading order: the page first, then the text around it.
const GROUPS: [key: string, label: string][] = [
  ['page', 'This page'], ['other_pages', 'Other supplied pages'], ['image', 'Page image'], ['document_text', 'Document text'],
  ['question', 'Question'], ['answer', 'Answer so far'], ['prompt', 'Prompt template'], ['special', 'Special tokens'],
  ['other', 'Prompt and special tokens'],
];

/** A page-image target: a generator image position, a retriever patch, or any boxed position that is not document text. */
export function isImagePosition(target: {role?: string; bbox?: number[] | null}) {
  return target.role === 'image' || target.role === 'visual_patch' || (!!target.bbox && target.role !== 'document_text');
}

function groupOf(target: Target, pageId: string | undefined, severalPages: boolean): string {
  if (isImagePosition(target))
    return severalPages ? (target.page_id === pageId ? 'page' : 'other_pages') : 'image';
  if (target.role === 'document_text') return 'document_text';
  if (target.role === 'query' || target.role === 'query_text') return 'question';
  if (target.role === 'generated') return 'answer';
  if (target.role === 'answer_text') return 'answer';
  if (target.role === 'prompt') return 'prompt';
  if (target.role === 'special') return 'special';
  return 'other';
}

/**
 * The row's mass by target role, largest first after the page groups. Rows with negative values (signed
 * differences) have no meaningful shares, so they get no breakdown. Targets past the row are not counted.
 */
export function massBreakdown(values: number[], targets: Target[], {pageId, severalPages = false, answerLabel}: {
  pageId?: string; severalPages?: boolean; answerLabel?: string;
} = {}): MassGroup[] {
  if (values.some(v => v < -1e-12)) return [];
  const groups = new Map<string, MassGroup>();
  for (const target of targets) {
    const mass = values[target.index];
    if (typeof mass !== 'number') continue;
    const key = groupOf(target, pageId, severalPages);
    const label = key === 'answer' && answerLabel ? answerLabel : GROUPS.find(([k]) => k === key)![1];
    const group = groups.get(key) ?? {key, label, mass: 0, positions: 0};
    group.mass += mass; group.positions += 1;
    const isToken = key !== 'page' && key !== 'other_pages' && key !== 'image';
    if (isToken && target.piece !== undefined && mass > (group.top?.mass ?? 0)) group.top = {index: target.index, piece: target.piece, mass};
    groups.set(key, group);
  }
  const order = (key: string) => GROUPS.findIndex(([k]) => k === key);
  const pages = new Set(['page', 'other_pages', 'image']);
  return [...groups.values()].sort((a, b) =>
    pages.has(a.key) || pages.has(b.key) ? order(a.key) - order(b.key) : b.mass - a.mass);
}
