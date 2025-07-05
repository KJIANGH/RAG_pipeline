from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from docx import Document
from flasgger import Swagger, swag_from
import os
import re
import requests
import json
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


app = Flask(__name__)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
    "tags_sorter": "manual",
    "swagger_ui_config": {
        "tagsSorter": "manual"
    }
}

swagger_template = {
    "info": {
        "title": "Your RAG Pipeline",
        "version": "1.0",
        "description": "Upload > Convert > Chunk > Embed > Search"
    },
    "tags": [
        {"name": "Step 1: Upload & Embed"},
        {"name": "Step 2: Search & Rerank"},
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)



UPLOAD_FOLDER = 'uploads'
CHUNK_FOLDER = 'chunks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHUNK_FOLDER, exist_ok=True)
END_PUNCTUATIONS = ('。', '!', '?', '.', ';', ':', '：', '；', '！', '？','”')


def upload_file(file):
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return save_path

def chunk_file(docx_path, parent_size=1000, child_size=300, overlap=50):
    loader = Docx2txtLoader(docx_path)
    parent_docs = loader.load()

    # parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=overlap)
    parent_chunks = parent_splitter.split_documents(parent_docs)  # List[Document]

    # child splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=overlap)
    
    all_child_chunks = []
    for parent_doc in parent_chunks:
        children = child_splitter.split_text(parent_doc.page_content)
        for child in children:
            all_child_chunks.append({
                "child": child,
                "parent": parent_doc.page_content
            })

    # 保存 parent-child JSON（可扩展加 metadata）
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    chunk_path = os.path.join(CHUNK_FOLDER, base_name + "_parent_child.json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump({"parent_child": all_child_chunks}, f, ensure_ascii=False, indent=2)

    return chunk_path


def get_ollama_embedding(text, model_name="qwen3-embed"):
    url = "http://localhost:11434/api/embeddings"
    response = requests.post(url, json={"model": model_name, "prompt": text})
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


def setup_milvus_collection(name, dim):
    connections.connect(host="127.0.0.1", port="19530", alias="default")
    if utility.has_collection(name):
        collection = Collection(name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="parent", dtype=DataType.VARCHAR, max_length=2000), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Chunk Embedding Collection with Parent")
    collection = Collection(name, schema)

    return collection


def embed_chunks_internal(chunk_path, collection_name):
    with open(chunk_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    if "parent_child" in json_data:
        records = json_data["parent_child"]
        texts = [item["child"] for item in records]
        metadatas = [{"parent": item["parent"]} for item in records]
    else:
        texts = json_data.get("chunks", [])
        metadatas = [{} for _ in texts]

    embeddings = [(text, get_ollama_embedding(text)) for text in texts]
    dim = len(embeddings[0][1])
    collection = setup_milvus_collection(name=collection_name, dim=dim)

    texts_only = [t[0] for t in embeddings]
    vectors_only = [t[1] for t in embeddings]

    collection.insert([texts_only, metadatas, vectors_only])
    collection.flush()
    return len(embeddings)

@app.route("/upload_and_embed", methods=["POST"])
@swag_from({
    'tags': ['Step 1: Upload & Embed'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {'name': 'file', 'in': 'formData', 'type': 'file', 'required': True},
        {'name': 'collection_name', 'in': 'formData', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'All steps successful'}}
})
def upload_and_embed():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    file = request.files['file']
    collection_name = request.form.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Missing collection_name'}), 400

    docx_path = upload_file(file)
    chunk_path = chunk_file(docx_path)
    count = embed_chunks_internal(chunk_path, collection_name)

    return jsonify({
        "message": "All steps completed",
        "chunks_embedded": count,
        "collection": collection_name
    })

def get_qwen_embedding(text):
    url = "http://localhost:11434/api/embeddings" 
    headers = {"Content-Type": "application/json"}
    payload = {"model": "qwen3-embed", "prompt": text}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['embedding']
    else:
        raise ValueError("Embedding failed: " + response.text)


def rerank_documents(query, documents, xinference_url="http://localhost:6006/v1/models"):
    payload = {
        "query": query,
        "documents": documents
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(xinference_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        results = response.json()["results"]
        sorted_docs = [documents[i["index"]] for i in sorted(results, key=lambda x: x["relevance_score"], reverse=True)]
        return sorted_docs
    else:
        raise ValueError("Rerank failed: " + response.text)


@app.route('/search', methods=['POST'])
@swag_from({
    'tags': ['Step 2: Search & Rerank'],
    'summary': 'Search top-k relevant chunks from Milvus and rerank results',
    'consumes': ['multipart/form-data'], 
    'parameters': [
        {
            'name': 'query',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'Your question for retrieval',
            'example': 'What are the termination conditions of the contract?'
        },
        {
            'name': 'top_k',
            'in': 'formData',
            'type': 'integer',
            'required': False,
            'example': 5,
            'description': 'Number of top results to return'
        },
        {
            'name': 'collection',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'embedding_chunks',
            'description': 'Collection name in Milvus to search'
        }
    ],
    'responses': {
        200: {
            'description': 'Search and rerank results returned',
            'examples': {
                'application/json': {
                    'query': 'What are the termination conditions of the contract?',
                    'results': [
                        'Either party may terminate with 30 days written notice...',
                        'Termination occurs automatically upon breach of confidentiality...'
                    ],
                    'count': 2,
                    'message': 'Search and rerank success'
                }
            }
        }
    }
})
def search():
    query = request.form.get("query")
    collection_name = request.form.get("collection", "embedding_chunks")
    top_k = int(request.form.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    try:
        query_embedding = get_qwen_embedding(query)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "parent"]  # 加上 parent
    )


    documents = [
        {
            "child": hit.entity.get("text"),
            "parent": hit.entity.get("parent")
        }
        for hit in results[0]
    ]

    try:
        reranked_texts = rerank_documents(query, [doc["child"] for doc in documents])
        reranked_full = []
        for text in reranked_texts:
            for doc in documents:
                if doc["child"] == text:
                    reranked_full.append(doc)
                    break

    except Exception as e:
        return jsonify({"error": f"Rerank failed: {str(e)}"}), 500

    return jsonify({
        "query": query,
        "results": reranked_full,  # 是 [{"child": ..., "parent": ...}, ...]
        "count": len(reranked_full),
        "message": "Search and rerank success"
    }), 200


if __name__ == '__main__':
    app.run(debug=True)
