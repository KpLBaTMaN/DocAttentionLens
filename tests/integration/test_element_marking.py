"""Marking elements on uploaded pages, in disposable schemas; no model inference."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pymupdf

from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository


def upload_pages(client):
    """A one-page PDF with a text line and a PNG with one black square at (10,20)–(30,40)."""
    identifier = client.post('/api/v1/setup/uploads', json={'kind': 'documents'}).json()['upload_id']
    pdf = pymupdf.open(); pdf.new_page(width=200, height=300).insert_text((20, 30), 'Marked fixture line')
    image = Image.new('RGB', (56, 84), 'white'); ImageDraw.Draw(image).rectangle((10, 20, 29, 39), fill='black')
    raw = BytesIO(); image.save(raw, format='PNG')
    for name, body in [('fixture.pdf', pdf.tobytes()), ('fixture.png', raw.getvalue())]:
        assert client.put(f'/api/v1/setup/uploads/{identifier}/files', params={'filename': name}, content=body).status_code == 200
    document = client.post(f'/api/v1/setup/uploads/{identifier}/documents', json={'filenames': ['fixture.pdf', 'fixture.png']})
    assert document.status_code == 200, document.text
    return document.json()


def mark(key, page_id, element_type, bbox, order=1, **extra):
    return {'key': key, 'page_id': page_id, 'element_type': element_type,
            'bbox': dict(zip(('x1', 'y1', 'x2', 'y2'), bbox)), 'reading_order': order, **extra}


def test_marking_fits_saves_and_freezes_a_marked_version(database):
    with TestClient(create_app(database)) as client:
        document = upload_pages(client)
        base = document['document_id']
        pdf_page, png_page = document['page_ids']
        url = f'/api/v1/documents/{base}/marking'
        draft = client.get(url).json()
        assert draft['revision'] == 0 and draft['elements'] == [] and draft['frozen'] is None
        assert [(p['width'], p['height'], p['text_layer']) for p in draft['pages']] == [(400, 600, True), (56, 84, False)]

        # A loose box tightens to the ink inside it and reads the PDF words there.
        fitted = client.post(url + '/fit', json={'page_id': pdf_page, 'bbox': {'x1': 10, 'y1': 10, 'x2': 390, 'y2': 120}}).json()
        box = fitted['bbox']
        assert fitted['tightened'] and fitted['text'] == 'Marked fixture line' and fitted['text_source'] == 'pdf_text_layer'
        assert 10 < box['x1'] < 45 and 20 < box['y1'] < 60 and box['x2'] < 390 and box['y2'] < 70
        kept = client.post(url + '/fit', json={'page_id': pdf_page, 'bbox': {'x1': 10, 'y1': 10, 'x2': 390, 'y2': 120}, 'tighten': False}).json()
        assert kept['bbox'] == {'x1': 10, 'y1': 10, 'x2': 390, 'y2': 120} and not kept['tightened'] and kept['text'] == 'Marked fixture line'
        square = client.post(url + '/fit', json={'page_id': png_page, 'bbox': {'x1': 2, 'y1': 2, 'x2': 50, 'y2': 70}}).json()
        assert square['bbox'] == {'x1': 7, 'y1': 17, 'x2': 33, 'y2': 43} and square['text'] is None and square['text_source'] == 'none'
        blank = client.post(url + '/fit', json={'page_id': png_page, 'bbox': {'x1': 40, 'y1': 50, 'x2': 50, 'y2': 70}}).json()
        assert not blank['tightened'] and blank['bbox'] == {'x1': 40, 'y1': 50, 'x2': 50, 'y2': 70}
        assert client.post(url + '/fit', json={'page_id': 'page_elsewhere', 'bbox': {'x1': 0, 'y1': 0, 'x2': 5, 'y2': 5}}).status_code == 422

        elements = [mark('a', pdf_page, 'heading', box.values(), text=fitted['text'], text_source='pdf_text_layer'),
                    mark('b', png_page, 'figure', square['bbox'].values())]
        saved = client.put(url, json={'expected_revision': 0, 'elements': elements, 'reviewed_page_ids': [pdf_page]})
        assert saved.status_code == 200, saved.text
        assert saved.json()['revision'] == 1 and saved.json()['reviewed_page_ids'] == [pdf_page]
        assert client.put(url, json={'expected_revision': 0, 'elements': elements}).status_code == 409
        for invalid in [[mark('a', pdf_page, 'text', (0, 0, 500, 10))], [mark('a', pdf_page, 'text', (5, 5, 6, 30))],
                        [mark('a', pdf_page, 'text', (0, 0, 20, 20), order=2)],
                        [mark('a', pdf_page, 'text', (0, 0, 20, 20)), mark('a', png_page, 'text', (0, 0, 20, 20))]]:
            assert client.put(url, json={'expected_revision': 1, 'elements': invalid}).status_code == 422
        assert client.put(url, json={'expected_revision': 1, 'elements': [mark('a', pdf_page, 'sidebar', (0, 0, 20, 20))]}).status_code == 422

        assert client.post(url + '/freeze', json={'expected_revision': 2}).status_code == 409
        frozen = client.post(url + '/freeze', json={'expected_revision': 1})
        assert frozen.status_code == 201, frozen.text
        marked = frozen.json()['document']
        assert marked['document_id'] != base and marked['element_count'] == 2 and marked['ocr_status'] == 'pdf_text_layer'
        assert frozen.json()['draft']['frozen'] == {'document_id': marked['document_id'], 'annotation_version': marked['annotation_version'], 'element_count': 2}
        first, second = marked['page_ids']
        listed = client.get(f'/api/v1/pages/{first}/elements').json()['items']
        assert [(e['element_type'], e['element_category'], e['native_id'], e['reading_order'], e['ocr']) for e in listed] == \
            [('heading', 'TEXT', 1, 1, {'status': 'present', 'text': 'Marked fixture line'})]
        figure = client.get(f'/api/v1/pages/{second}/elements').json()['items'][0]
        assert (figure['element_category'], figure['ocr']['status'], figure['display_bbox']) == ('IMAGE', 'missing', square['bbox'])
        # The marked version serves the upload's verified pixels.
        assert client.get(f'/api/v1/pages/{second}/image').content == client.get(f'/api/v1/pages/{png_page}/image').content
        summary = client.get(f"/api/v1/datasets/{marked['ref']['dataset_id']}/summary").json()
        assert summary['adapter_version'] == 'uploaded_page_elements_v1' and summary['source_manifest']['base_document_id'] == base

        # The same elements give the same version; a marked version opens its upload's draft.
        again = client.post(url + '/freeze', json={'expected_revision': 2}).json()
        assert again['document']['document_id'] == marked['document_id'] and again['draft']['revision'] == 3
        reopened = client.get(f"/api/v1/documents/{marked['document_id']}/marking").json()
        assert reopened['document_id'] == base and reopened['revision'] == 3 and len(reopened['elements']) == 2

        case = client.post('/api/v1/setup/cases', json={'document_id': marked['document_id'], 'query': 'What is the heading?'}).json()
        assert case['annotation_version'] == marked['annotation_version'] and case['page_ids'] == marked['page_ids']

        # Without a draft, opening a marked version starts from its elements.
        with JobRepository(database).connect() as conn:
            conn.execute("DELETE FROM element_marking_drafts")
        seeded = client.get(f"/api/v1/documents/{marked['document_id']}/marking").json()
        assert seeded['revision'] == 0 and [e['key'] for e in seeded['elements']] == ['a', 'b']
        assert seeded['frozen']['document_id'] == marked['document_id']
        assert client.get(url).json()['elements'] == []


def test_marking_rejects_documents_that_are_not_uploads(registered):
    with TestClient(create_app(registered)) as client:
        dataset = client.get('/api/v1/datasets').json()['items'][0]['dataset_id']
        document = client.get('/api/v1/documents', params={'dataset_id': dataset, 'limit': 1}).json()['items'][0]
        response = client.get(f"/api/v1/documents/{document['document_id']}/marking")
        assert response.status_code == 422 and response.json()['error']['code'] == 'unsupported_document'
