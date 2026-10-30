/** Input structure labels, independent of the selected attention method. */
export const INPUT_GROUNDING = {
  table_grounding: {label: 'Table grounding', description: 'Cells, rows, columns and header paths'},
  document_layout: {label: 'Element layout grounding', description: 'Paragraphs, headings, figures and tables'},
} as const;

export type GroundingKind = keyof typeof INPUT_GROUNDING;
