import os
from urllib.parse import urljoin

import requests
from typing import List
from langchain_core.embeddings import Embeddings

class LoraXAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_url: str, model: str):
        self.api_key = 'Bearer ' + api_key
        self.api_url = urljoin(os.path.join(api_url, ''),'embeddings')
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers={'Authorization': self.api_key},
            json={
                "model": self.model,
                "input": texts,
            },
        )
        response = response.json()
        response_list = []
        if 'data' in response:
            for resp in response['data']:
                if 'embedding' in resp:
                    response_list.append(resp['embedding'])
        return response_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
        #return self.embed_documents([text])['data'][0]['embedding']
        #return self.embed_documents([text])[0]