import os
from urllib.parse import urljoin

import requests
from typing import List
from langchain_core.embeddings import Embeddings

class LoraXAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_url: str, model="", max_batch_size=10):
        self.api_key = 'Bearer ' + api_key
        self.api_url = urljoin(os.path.join(api_url, ''),'embeddings')
        self.model = model
        self.max_batch_size = max_batch_size

    def query_data(self, session, request_list, response_list):
        #print('rl:', request_list)
        response = session.post(
            self.api_url,
            headers={'Authorization': self.api_key},
            json={
                "model": self.model,
                "input": request_list,
            },
        )
        #print('response:', response.text)
        response = response.json()
        if 'data' in response:
            for resp in response['data']:
                if 'embedding' in resp:
                    emb = resp['embedding']
                    #print('emb:', emb)
                    response_list.append(emb)
                else:
                    print('why is embedding not in:', resp)
        else:
            print('WHY IS DATA NOT IN: ', response)

        request_list.clear()
        return response_list

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        max_size = self.max_batch_size
        offset = 0
        response_list = []

        session = requests.Session()

        while len(texts) != len(response_list):
            remaining = len(texts) - len(response_list)
            if max_size > remaining:
                max_size = remaining

            request_list = texts[offset:offset + max_size]
            offset += max_size
            response_list = self.query_data(session, request_list, response_list)

        session.close()

        return response_list

    def embed_query(self, text: str) -> List[float]:
        response = self.embed_documents([text])
        if len(response) > 0:
            return response[0]
        else:
            print('embed_query response is None')



