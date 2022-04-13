import pytest
from docarray import Document, DocumentArray
from jina import Flow

from executor import Doc2QueryExecutor

NUM_QUESTIONS = 10


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(
                id='doc1',
                text='Jina is a neural search framework that allows anyone to build SOTA & scalable deep learning search applications in minutes.',
            ),
            Document(
                id='doc2',
                text=' neural search is deep neural network-powered information retrieval',
            ),
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
        result_docs = f.post(
            on='/',
            inputs=docs,
        )
        assert len(result_docs[0].chunks) == 10
