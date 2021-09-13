import pytest
from jina import Document, DocumentArray, Flow

from executor import Doc2QueryExecutor

NUM_QUESTIONS = 10


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1',
                     content='Jina is a neural search framework that allows anyone to build SOTA & scalable deep learning search applications in minutes.'),
            Document(id='doc2', content=' neural search is deep neural network-powered information retrieval'),
        ]
    )


def test_doc2query(docs):
    doc2query = Doc2QueryExecutor(num_questions=NUM_QUESTIONS)
    doc2query.doc2query(docs)
    for d in docs:
        assert len(d.chunks) == 10


def test_flow(docs):
    f = Flow().add(uses=Doc2QueryExecutor)

    with f:
        resp = f.post(
            on='/',
            inputs=docs,
            return_results=True,
        )
        print(f'{resp}')
        assert len(resp[0].docs[0].chunks) == 10
