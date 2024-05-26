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
        return request_list, response_list


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response_list = []

        session = requests.Session()
        remaining = len(texts)
        request_list = []
        count = 0
        for text in texts:

            last_batch = False
            if self.max_batch_size > remaining:
                self.max_batch_size = remaining
                if(self.max_batch_size == remaining):
                    #print('last batch')
                    last_batch = True
                #print('changing max size: ', max_size, 'request_list:', len(request_list))

            if len(request_list) == self.max_batch_size:

                request_list, response_list = self.query_data(session, request_list, response_list)


            remaining = (len(texts) - len(response_list))
            count += 1
            '''
            print('request_list:', len(request_list))
            print('response_list:', len(response_list))
            print('text:', len(texts))
            print('remaining:', remaining)
            print('count:', count)
            print('max_size:', max_size)

            print('---')
            '''

            if remaining != 0:
                request_list.append(text)

            if last_batch:
                request_list, response_list = self.query_data(session, request_list, response_list)

        '''
        print('e_request_list:', len(request_list))
        print('e_response_list:', len(response_list))
        print('e_text:', len(texts))
        print('e_remaining:', remaining)
        print('e_count:', count)
        '''

        session.close()

        return response_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

