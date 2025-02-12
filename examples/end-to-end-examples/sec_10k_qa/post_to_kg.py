import argparse
from multiprocessing.pool import ThreadPool
import logging
from urllib.parse import urlparse
import os
import openai
import re
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import wikipedia
import pandas as pd

import boto3
import botocore
from boto3.s3.transfer import TransferConfig

from llama_index import (
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index import Document
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI

from transformers import pipeline

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_NAME = "downloads" 
LOCAL_PATH = "/tmp/"
LOCAL_FILE = "post_jtbd.csv"
DATA_PATH = "gs://mosaicml_test/u__bojan/kg/post_jtbd.csv"
DESTINATION_PATH = "gs://mosaicml_test/u__bojan/kg/"
NUM_PARALLEL_FILES = 4
NUM_THREADS_PER_FILE = 10
MB = 1024 * 1024

LIMIT = 10000

openai.api_key = os.environ["OPENAI_API_KEY"]

triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
    # comment this line to run on CPU
    #device="cuda:0",
)


class DownloadTask:
    file_key: str
    size: int

    def __init__(self, file_key: str, size):
        self.file_key = file_key
        self.size = size


class CloudPath:
    s3_path: Optional[str]
    hf_path: Optional[str]
    gcp_path: Optional[str]


class DataPath(CloudPath):
    """
    Typed dictionary for default download parameters that are
    compatible with our out-of-the-box downloader
    """
    s3_path: Optional[str] = None
    hf_path: Optional[str] = None
    gcp_path: Optional[str] = None

    def __init__(self, **path):
        self.gcp_path = path['gcp_path']

    def is_empty(self):
        return not (self.s3_path or self.hf_path or self.gcp_path)

    def get_path(self) -> str:
        if not self.hf_path:
            return os.path.join(LOCAL_PATH, LOCAL_NAME)
        return self.s3_path or self.hf_path or self.gcp_path

def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode(
        [
            triplet_extractor(input_text, return_tensors=True, return_text=False)[0][
                "generated_token_ids"
            ]
        ]
    )[0]

    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject != "" and relation != "" and object_ != "":
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


class WikiFilter:
    def __init__(self):
        self.cache = {}

    def filter(self, candidate_entity):
        # check the cache to avoid network calls
        if candidate_entity in self.cache:
            return self.cache[candidate_entity]["title"]

        # pull the page from wikipedia -- if it exists
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary,
            }

            # cache the page title and original entity
            self.cache[candidate_entity] = entity_data
            self.cache[page.title] = entity_data

            return entity_data["title"]
        except:
            return None


def extract_triplets_wiki(text):
    relations = extract_triplets(text)

    filtered_relations = []
    for relation in relations:
        (subj, rel, obj) = relation
        filtered_subj = wiki_filter.filter(subj)
        filtered_obj = wiki_filter.filter(obj)

        # skip if at least one entity not linked to wiki
        if filtered_subj is None and filtered_obj is None:
            continue

        filtered_relations.append(
            (
                filtered_subj or subj,
                rel,
                filtered_obj or obj,
            )
        )

    return filtered_relations


def download_model(download_parameters: DataPath):
    """
    This function runs at server startup and handles downloading all relevant model files
    """

    if download_parameters.s3_path or download_parameters.gcp_path:
        config = botocore.client.Config(max_pool_connections=NUM_PARALLEL_FILES * NUM_THREADS_PER_FILE)
        if download_parameters.s3_path:
            model_name_or_path = download_parameters.s3_path
            # s3 creds need to already be present as env vars
            # s3 = boto3.client('s3', config=config)
            # todo
            # s3 = boto3.client('s3',
            #                    aws_access_key_id=AWS_ACCESS_KEY_ID,
            #                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            #                   )
        else:
            s3 = boto3.client("s3",
                              region_name="auto",
                              endpoint_url="https://storage.googleapis.com",
                              aws_access_key_id=os.environ["GCS_KEY"],
                              aws_secret_access_key=os.environ["GCS_SECRET"],
                              config=config)
            model_name_or_path = download_parameters.gcp_path
        local_path = os.path.join(LOCAL_PATH, LOCAL_NAME)
        # Download model files
        if os.path.exists(local_path):
            logger.info(f"[+] Path {local_path} already exists, checking for missing files")
        else:
            Path(local_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading data from path: {model_name_or_path}")

        parsed_path = urlparse(model_name_or_path)

        objs = s3.list_objects_v2(
            Bucket=parsed_path.netloc,
            Prefix=parsed_path.path.lstrip("/"),
        )
        downloaded_file_set = set(os.listdir(local_path))

        file_keys = []
        for obj in objs['Contents']:
            file_key = obj['Key']
            file_name = os.path.basename(file_key)
            if not file_name or file_name.startswith("."):
                # Ignore hidden files
                continue
            if file_name not in downloaded_file_set:
                file_keys.append(file_key)

        def get_tasks(file_keys: List[str]) -> List[DownloadTask]:
            key_to_size = {}
            for file_key in file_keys:
                size = s3.head_object(Bucket=parsed_path.netloc, Key=file_key)["ContentLength"]
                key_to_size[file_key] = size

            sorted_keys = sorted(file_keys, key=lambda x: key_to_size[x])

            tasks = []
            for file_key in sorted_keys:
                tasks.append(DownloadTask(file_key, key_to_size[file_key]))
            return tasks

        config = TransferConfig(multipart_threshold=512 * MB,
                                multipart_chunksize=256 * MB,
                                max_concurrency=NUM_THREADS_PER_FILE)

        tasks = get_tasks(file_keys)

        def download_file(task: DownloadTask) -> None:
            file_name = os.path.basename(task.file_key)
            with tqdm(total=task.size, unit="B", unit_scale=True, desc=file_name) as pbar:
                try:
                    s3.download_file(
                        Bucket=parsed_path.netloc,
                        Key=task.file_key,
                        Filename=os.path.join(local_path, file_name),
                        Callback=lambda x: pbar.update(x),
                        Config=config,
                    )
                except botocore.exceptions.ClientError as e:
                    print(f"Error downloading file with key: {file_key} with error: {e}")
        
        with ThreadPool(NUM_PARALLEL_FILES) as pool:
            pool.map(download_file, tasks)


def parse_data_path(path: str):
    # Regular expression for S3 path: s3://bucket_name/object_key
    s3_regex = re.compile(r'^s3://(?P<bucket>[a-zA-Z0-9._-]+)/(?P<key>.+)$')
    
    # Regular expression for GCS path: gs://bucket_name/object_key
    gcs_regex = re.compile(r'^gs://(?P<bucket>[a-zA-Z0-9._-]+)/(?P<key>.+)$')
    
    # Check if the path is an S3 path
    s3_match = s3_regex.match(path)
    if s3_match:
        return {"s3_path": path}
    
    # Check if the path is a GCS path
    gcs_match = gcs_regex.match(path)
    if gcs_match:
        return {"gcp_path": path}
    
    # If the path is neither S3 nor GCS, raise an error
    raise ValueError("Invalid checkpoint path, please double check. Supported paths are S3 (s3://bucket/key) and GCS (gs://bucket/key)")


def generate_and_store_graph(documents, storage_context, service_context,
                             limit, max_triplets, use_wiki=False):

    triplet_fn = extract_triplets if not use_wiki else extract_triplets_wiki

    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=max_triplets,
        kg_triplet_extract_fn=triplet_fn,
        storage_context=storage_context,
        service_context=service_context,
        include_embeddings=True,
    )

    ## create graph
    from pyvis.network import Network

    g = index.get_networkx_graph(limit=10000)
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)

    file_prefix = "" if not use_wiki else "wiki_"

    file_name = f"{file_prefix}non_filtered_graph_{limit}.html"
    GRAPH_PATH = os.path.join(LOCAL_PATH, LOCAL_NAME, file_name) 
    net.save_graph(GRAPH_PATH)

    config = botocore.client.Config(max_pool_connections=NUM_PARALLEL_FILES * NUM_THREADS_PER_FILE)
    s3 = boto3.client("s3",
                      region_name="auto",
                      endpoint_url="https://storage.googleapis.com",
                      aws_access_key_id=os.environ["GCS_KEY"],
                      aws_secret_access_key=os.environ["GCS_SECRET"],
                      config=config
    )

    s3.upload_file(Filename=GRAPH_PATH, Bucket='mosaicml_test', Key=f'u__bojan/kg/{file_name}')
                    

def main(limit: int, run_on_gpu: bool, max_triplets: int) -> None:
    data_path = parse_data_path(DATA_PATH)
    download_parameters = DataPath(**data_path)
    download_model(download_parameters)

    local_path = os.path.join(LOCAL_PATH, LOCAL_NAME, LOCAL_FILE)
    df = pd.read_csv(local_path)
    df = df.head(limit)

    # merge all documents into one, since it's split by page
    documents = [Document(text="".join([x.text for x in df.itertuples() if x.text]))]

    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=256)

    # set up graph storage context
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    logger.info("Generating graph w/o wiki filtering")
    generate_and_store_graph(documents, service_context=service_context,
                             storage_context=storage_context, limit=limit,
                             max_triplets=max_triplets)
    logger.info("Generating graph with wiki filtering")
    generate_and_store_graph(documents, service_context=service_context,
                             storage_context=storage_context, limit=limit,
                             max_triplets=max_triplets, use_wiki=True)


wiki_filter = WikiFilter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process and generate KG from the posts')
    parser.add_argument(
        '--folder_for_upload',
        type=str,
        help='Object store prefix to upload the processed data to')
    parser.add_argument(
        '--run_on_gpu',
        type=bool,
        help='If this code should run on GPU')
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Number of rows to include in the graph.')
    parser.add_argument(
        '--max_triplets',
        type=int,
        default=10,
        help='Max triplets per document to extract.')
    args = parser.parse_args()

    main(args.limit, args.run_on_gpu, args.max_triplets)
