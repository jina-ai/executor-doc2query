# Doc2QueryExecutor

**Doc2QueryExecutor** is an executor which generates "questions" from sentences and add them into chunks.


## Usage 

```python
from jina import Flow, DocumentArray, Document

document_array= DocumentArray(
        [
            Document(id='doc1',content='Hello world'),
            Document(id='doc2',content='Neural search is deep neural network-powered information retrieval'),
        ])

f = Flow().add(uses='jinahub://Doc2QueryExecutor', uses_with={'num_questions': 10})

with f:
    resp = f.post(on='/', inputs=document_array, return_results=True)
    print(f'{resp[0].data.docs[0].chunks}')

```


## Reference
- See the [Document Expansion by Query Prediction](https://github.com/castorini/docTTTTTquery)
