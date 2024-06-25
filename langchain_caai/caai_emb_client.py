import os
from urllib.parse import urljoin
from tqdm import tqdm
import multiprocessing

import requests
from typing import List
from langchain_core.embeddings import Embeddings

class caai_emb_client(Embeddings):
    def __init__(self, api_key: str, api_url: str, model="", max_batch_size=10, num_workers=1, progress_bar=False):
        self.api_key = 'Bearer ' + api_key
        self.api_url = urljoin(os.path.join(api_url, ''),'embeddings')
        self.model = model
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.progress_bar = progress_bar

    def query_data(self, request_list, session, something) -> list:

        response_list = []

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
            print('request_list:', request_list)

        return response_list
    def query_data_old(self, session, request_list, response_list):
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
            print('request_list:', request_list)
        request_list.clear()
        return response_list


    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        if self.progress_bar:
            progress_bar = tqdm(total=len(texts), unit="record", desc="Get embeddings")

        workers_pool = multiprocessing.Pool(self.num_workers)
        jobs_list = []

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

            print('request_list_size:', len(request_list))
            print('request_list:', request_list)
            if len(request_list) > 0:
                exit()

            if self.progress_bar:
                job = workers_pool.apply_async(self.query_data, args=(list(request_list), session, None),
                                         callback=lambda arg: progress_bar.update(self.max_batch_size))
            else:
                job = workers_pool.apply_async(self.query_data, args=(list(request_list), session, None))

            jobs_list.append(job)

            #response_list = self.query_data(request_list, response_list)

        workers_pool.close()
        workers_pool.join()

        session.close()

        if self.progress_bar:
            progress_bar.close()

        print('job_list:', len(jobs_list))
        for proc in jobs_list:
            j = proc.get()
            for record in j:
                response_list.append(record)

        print('in:', len(texts))
        print('out:', len(response_list))

        return response_list

    def embed_documents_old(self, texts: List[str]) -> List[List[float]]:

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