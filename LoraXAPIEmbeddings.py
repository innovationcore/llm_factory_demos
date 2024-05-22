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
        response_list = []
        count = 0
        max_size = 10

        while count <= len(texts):
            if len(texts) < max_size:
                max_size = len(texts)

            canidate_text = texts[:max_size]
            del texts[:max_size]
            count += max_size

            response = requests.post(
                self.api_url,
                headers={'Authorization': self.api_key},
                json={
                    "model": self.model,
                    "input": canidate_text,
                },
            )
            response = response.json()

            if 'data' in response:
                for resp in response['data']:
                    if 'embedding' in resp:
                        emb = resp['embedding']
                        response_list.append(emb)
                    else:
                        print('why is embedding not in:', resp)
            else:
                print('WHY IS DATA NOT IN: ', response)

        return response_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

