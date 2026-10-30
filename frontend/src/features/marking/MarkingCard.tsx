import { Link } from 'react-router-dom';
import type { Document } from '../../api/client';
import '../../styles/marking.css';

const SHOWN = 8;

/** Under an uploaded document in setup: whether its elements are marked, and the way to mark them. */
export function MarkingCard({document}:{document:Document}) {
  const marked = document.element_count > 0;
  const href = `/documents/${document.document_id}/elements?return=setup`;
  return <div className="docattentionlens-marking-card" data-testid="marking-card">
    <div className="docattentionlens-marking-card-pages" aria-hidden="true">{document.page_ids.slice(0, SHOWN).map((id, i) =>
      <figure key={id}><img src={`/api/v1/pages/${id}/image`} alt="" loading="lazy"/>{i + 1}</figure>)}
      {document.page_ids.length > SHOWN && <figure>+{document.page_ids.length - SHOWN}</figure>}</div>
    <div>
      <strong>{marked ? `Elements · ${document.element_count} marked · version ${document.annotation_version.slice(0, 8)}` : 'Elements · not marked yet'}</strong>
      <p>{marked ? 'Attention on these pages can be read by element as well as by position. Editing the elements saves a new version; questions you have already used keep the version they were asked about.'
        : 'Mark the headings, paragraphs, tables and figures on these pages to read attention by element as well as by position. Optional: without elements you still get attention by position.'}</p>
      <Link className="docattentionlens-button" to={href}>{marked ? 'Edit elements' : 'Mark elements'}</Link>
    </div>
  </div>;
}
