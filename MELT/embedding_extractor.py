import json
import multiprocessing
import os.path
import time

from tqdm import tqdm
from LoraXAPIEmbeddings import LoraXAPIEmbeddings

with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

max_batch_size = 100

embeddings = LoraXAPIEmbeddings(
    api_key=llm_api_key,
    api_url=llm_api_base,
    max_batch_size=max_batch_size
)


def combine_datasets(output_path):
    combined_dataset = []

    datasets = ['case-chat-med-train.json','multi-choice-med-train.json','qa-med-train.json']
    # datasets = ['test.json']

    count = 0

    for dataset in datasets:

        dataset = os.path.join('MELT_dataset', dataset)

        with open(dataset) as f:

            data = json.load(f)
            for record in data:
                embedding_str = ''
                if 'case-chat-med-train.json' in dataset:
                    embedding_str = record['input']
                elif 'multi-choice-med-train.json' in dataset:
                    embedding_str = record['instruction'] + '\n\n' + record['input']
                elif 'qa-med-train.json' in dataset:
                    embedding_str = record['instruction']
                elif 'test.json' in dataset:
                    embedding_str = record['instruction']

                # embedding_result = embeddings.embed_query(embedding_str)

                if len(embedding_str) < 10:
                    print('dataset:', dataset)
                    print('str:', embedding_str)
                else:
                    record['embedding_str'] = embedding_str
                    record['id'] = count
                    combined_dataset.append(record)
                    count += 1

    print('combined total records:', count)

    with open(output_path, 'w') as fp:
        json.dump(combined_dataset, fp, indent=4)


def worker(records, job_list) -> list:

    emb_list = []

    for record in records:
        emb_list.append(record['embedding_str'])

    results = embeddings.embed_documents(emb_list)


    for idx, result in enumerate(results):
        records[idx]['embedding'] = result

    return records


if __name__ == '__main__':

    combined_dataset = []

    num_workers = 9

    combined_str_path = 'combined_str_dataset.json'
    #combine_datasets(combined_str_path)

    with open(combined_str_path) as f:
        combined_str_data = json.load(f)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(combined_str_data), unit="record", desc="Get embeddings")
    workers_pool = multiprocessing.Pool(num_workers)
    jobs_list = []
    run_list = []
    c = 0

    max_size = max_batch_size
    offset = 0


    while len(combined_str_data) != c:

        remaining = len(combined_str_data) - c

        if max_size > remaining:
            max_size = remaining

        request_list = combined_str_data[offset:offset + max_size]
        offset += max_size
        job = workers_pool.apply_async(worker, args=(list(request_list), None), callback=lambda arg: progress_bar.update(max_batch_size))
        jobs_list.append(job)

        c += 1

    workers_pool.close()
    workers_pool.join()
    progress_bar.close()

    print('job_list:', len(jobs_list))
    for proc in jobs_list:
        j = proc.get()
        for record in j:
            combined_dataset.append(record)

    print('in:', len(combined_str_data))
    print('out:', len(combined_dataset))

    print('count:', c)

    with open('combined_dataset.json', 'w') as fp:
        json.dump(combined_dataset, fp, indent=4)
