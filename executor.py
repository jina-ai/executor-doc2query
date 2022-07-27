from typing import Dict
from warnings import warn

import torch
from docarray import Document, DocumentArray
from jina import Executor, requests
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Doc2QueryExecutor(Executor):
    """
    An executor which generates "questions" from sentences and add them into chunks.

    """

    def __init__(self, num_questions: int = 10, access_paths: str = '@r', **kwargs):
        """
        :param num_questions: the number of questions to generate
        :param access_paths: the traverse path on docs, e.g. '@r', '@c'
        """

        if("traversal_paths" in kwargs.keys()):
            warn("'traversal_paths' is deprecated, please use 'access_paths'")

        super().__init__(**kwargs)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = T5Tokenizer.from_pretrained(
            'castorini/doc2query-t5-base-msmarco'
        )
        self._model = T5ForConditionalGeneration.from_pretrained(
            'castorini/doc2query-t5-base-msmarco'
        )
        self._model.to(self._device)
        self._num_questions = num_questions
        self._access_paths = access_paths

    @requests
    def doc2query(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        Generates queries for documents stores them into chunks

        :param docs: document array containing text that we will generate questions for
        :param parameters: dict with optional parameters
        :param **kwargs: keyword arguments
        """
        access_paths = parameters.get('access_paths', self._access_paths)
        for d in docs[access_paths]:
            input_ids = self._tokenizer.encode(d.text, return_tensors='pt').to(
                self._device
            )

            outputs = self._model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_k=self._num_questions,
                num_return_sequences=self._num_questions,
            )

            for o in outputs:
                d.chunks.append(
                    Document(text=self._tokenizer.decode(o, skip_special_tokens=True))
                )
