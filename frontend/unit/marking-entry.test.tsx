// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { PageElements } from '../src/features/document/PageElements';
import { MarkingCard } from '../src/features/marking/MarkingCard';
import type { Document } from '../src/api/client';
import { PAGE_ID, elementRecords } from '../fixtures/attention';

afterEach(cleanup);
const upload:Document = {key:'document_u', document_id:'document_u', ref:{dataset_id:'dataset_u', dataset_version:'u', domain:'uploaded', document_id:'document_u'},
  native_id:'document_u', filename:'report.pdf', total_pages:2, page_ids:['page_a', 'page_b'], question_count:0, element_count:0,
  annotation_version:'unannotated', source_pdf_sha256:'', ocr_model:'none', ocr_status:'not_available'};

describe('ways into Mark elements', () => {
  it('offers marking from an uploaded page without elements, and only there', () => {
    const empty = render(<MemoryRouter><PageElements elements={[]} onSelect={() => {}} positions={[]} pageId={PAGE_ID} markHref="/documents/document_u/elements"/></MemoryRouter>);
    expect(empty.getByRole('link', {name:'Mark elements'}).getAttribute('href')).toBe('/documents/document_u/elements');
    cleanup();
    const marked = render(<MemoryRouter><PageElements elements={elementRecords} onSelect={() => {}} positions={[]} pageId={PAGE_ID} markHref="/documents/document_u/elements"/></MemoryRouter>);
    expect(marked.queryByTestId('mark-elements-link')).toBeNull();
  });

  it('shows in setup whether an upload has marked elements and returns there afterwards', () => {
    const before = render(<MemoryRouter><MarkingCard document={upload}/></MemoryRouter>);
    expect(before.getByText('Elements · not marked yet')).toBeTruthy();
    expect(before.getByRole('link', {name:'Mark elements'}).getAttribute('href')).toBe('/documents/document_u/elements?return=setup');
    expect(before.container.querySelectorAll('img')).toHaveLength(2);
    cleanup();
    const after = render(<MemoryRouter><MarkingCard document={{...upload, element_count:12, annotation_version:'3f9c2a1b'.padEnd(64, '0')}}/></MemoryRouter>);
    expect(after.getByText('Elements · 12 marked · version 3f9c2a1b')).toBeTruthy();
    expect(after.getByRole('link', {name:'Edit elements'})).toBeTruthy();
  });
});
