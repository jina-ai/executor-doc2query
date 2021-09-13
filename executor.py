from typing import Iterable

from jina import DocumentArray, Executor, requests, Document
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Doc2QueryExecutor(Executor):
    def __init__(self,
                 num_questions: int = 10,
                 default_traversal_paths: Iterable[str] = ('r',),
                 **kwargs):
        super().__init__(**kwargs)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self._model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self._model.to(self._device)
        self._num_questions = num_questions
        self._default_traversal_paths = default_traversal_paths

    @requests
    def doc2query(self, docs: DocumentArray, **kwargs):
        for d in docs.traverse_flat(self._default_traversal_paths):
            input_ids = self._tokenizer.encode(d.content, return_tensors='pt').to(self._device)

            outputs = self._model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_k=self._num_questions,
                num_return_sequences=self._num_questions)

            for o in outputs:
                d.chunks.append(Document(content=self._tokenizer.decode(o, skip_special_tokens=True)))
