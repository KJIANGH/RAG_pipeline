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
from langchain.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from metadata_extractor import extract_metadata
from bm25 import save_bm25_texts, bm25_search
from langdetect import detect
from transformers import pipeline
import uuid


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


def preprocess_text(doc):
    """Fix undesired line breaks between English and Chinese characters"""
    cleaned_text = re.sub(r"(?<=[a-zA-Z])\n(?=[\u4e00-\u9fff])", " ", doc.page_content)
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+$', '', cleaned_text, flags=re.MULTILINE)
    doc.page_content = cleaned_text
    return doc

    
def load_and_structure_document(file_path):
    """Load and structure document using Unstructured library"""
    if file_path.lower().endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    elif file_path.lower().endswith('.pdf'):
        loader = UnstructuredPDFLoader(file_path, strategy="hi_res", mode="elements")
    else:
        raise ValueError("Unsupported file format. Only DOCX and PDF are supported.")
    
    structured_docs = loader.load()
    filename = os.path.basename(file_path)

    for doc in structured_docs:
        doc.metadata["filename"] = filename

    return structured_docs


def split_by_language(text):
    # 把中英文混排的段落分出来（简单规则：句子级别的切割）
    sentences = re.split(r'(?<=[。.!?])\s*', text)
    result = []
    current_lang = None
    current_chunk = []

    for sent in sentences:
        if not sent.strip():
            continue
        try:
            lang = detect(sent)
        except:
            lang = 'unknown'

        if current_lang is None:
            current_lang = lang

        if lang == current_lang:
            current_chunk.append(sent)
        else:
            result.append(''.join(current_chunk))
            current_chunk = [sent]
            current_lang = lang

    if current_chunk:
        result.append(''.join(current_chunk))

    return result

# def chunk_with_parent_child(docs, parent_chunk_size=500, parent_overlap=50, child_chunk_size=100, child_overlap=15):
#     # First split into parent chunks
#     parent_splitter = NLTKTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_overlap)
#     parent_docs = parent_splitter.split_documents(docs)

#     # Then split each parent into children
#     child_splitter = NLTKTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_overlap)
#     parent_child_pairs = []

#     for parent_doc in parent_docs:
#         children = child_splitter.split_documents([parent_doc])
#         for child_doc in children:
#             parent_child_pairs.append({
#                 "parent": parent_doc.page_content,
#                 "child": child_doc.page_content,
#                 "metadata": parent_doc.metadata
#             })
#     return parent_child_pairs


def inject_metadata_to_content(text: str, metadata: dict) -> str:
    prefix_parts = []
    
    filename = metadata.get("filename")
    section = metadata.get("section")
    term = metadata.get("term")

    if filename:
        prefix_parts.append(f"文件名: {filename}")
    if section:
        prefix_parts.append(f"条款编号: {section}")
    if term:
        prefix_parts.append(f"关键词: {term}")

    prefix = "; ".join(prefix_parts)
    return f"{prefix}\n{text}" if prefix else text


def chunk_with_parent_child(docs, file_path):
    # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。", ".", " ", ""])
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30,separators=["\n\n", "\n", "。", ".", " ", ""])
    
    all_chunks = []

    for doc in docs:
        parent_metadata = {**doc.metadata, **extract_metadata(doc, file_path)}
        parent_id = doc.metadata.get("element_id", str(uuid.uuid4()))

        language_blocks = split_by_language(doc.page_content)

        for block in language_blocks:
            child_chunks = child_splitter.split_text(block)

            for child_text in child_chunks:
                metadata = parent_metadata.copy()
                metadata["parent_id"] = parent_id
                enriched_text = inject_metadata_to_content(child_text, metadata)
                all_chunks.append(Document(page_content=enriched_text, metadata=metadata))
                        
    return all_chunks


def process_file(file_path):
    structured_docs = load_and_structure_document(file_path)
    save_bm25_texts(structured_docs, file_path)
    parent_child_chunks = chunk_with_parent_child(structured_docs, file_path)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    chunk_path = os.path.join(CHUNK_FOLDER, base_name + "_parent_child.json")
    os.makedirs(CHUNK_FOLDER, exist_ok=True)

    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump({
            "parent_child": [
                {"child": doc.page_content, "metadata": doc.metadata}
                for doc in parent_child_chunks
            ]
        }, f, ensure_ascii=False, indent=2)

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
        metadatas = [item.get("metadata", {}) for item in records]
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
    chunk_path = process_file(docx_path)
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
    

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
def classify_query_type(query):
    candidate_labels = ["precise_lookup", "explain_compare", "general_qa"]
    
    result = classifier(query, candidate_labels)
    predicted_class = result["labels"][0]
    
    return predicted_class


def embedding_search(query, collection_name="embedding_chunks", top_k=5):
    try:
        query_embedding = get_qwen_embedding(query)
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}, 500

    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "parent_id", "filename", "section", "term"]
        )
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}, 500

    results_list = []
    for hit in results[0]:
        entity = hit.entity
        results_list.append({
            "content": entity.get("text"),
            "parent_id": entity.get("parent_id"),
            "filename": entity.get("filename"),
            "section": entity.get("section"),
            "term": entity.get("term"),
            "score": hit.distance,
            "source": "embedding"
        })

    return results_list

def hybrid_search(query, bm25_k=3, embed_k=3):
    bm25_results = bm25_search(query, top_k=bm25_k)
    embed_results = embedding_search(query, top_k=embed_k)

    for r in embed_results:
        r["score"] = 1 / (1 + r["score"]) 
    
    combined = bm25_results + embed_results

    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)

    return combined_sorted


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

    query_type = classify_query_type(query)
    
    if query_type == "precise_lookup":
        results = bm25_search(query, top_k)
    
    elif query_type == "explain_compare":
        results = hybrid_search(query, top_k-1, top_k-1)
    
    else:
        connections.connect(host="localhost", port="19530")
        collection = Collection(collection_name)
        collection.load()
        query_embedding = get_qwen_embedding(query)
        results = embedding_search(collection, query_embedding, top_k)



    try:
        texts_to_rerank = [doc["child"] + "\n\n" + doc["parent"] for doc in results]
        reranked_texts = rerank_documents(query, texts_to_rerank)

        text_to_doc = {doc["child"] + "\n\n" + doc["parent"]: doc for doc in results}
        reranked_full = [text_to_doc[text] for text in reranked_texts if text in text_to_doc]

        prompt = query + "\n\n" + "\n\n---\n\n".join(reranked_texts) + "\n\n" + "请根据这个用户的query加上利用RAG系统检索出来的相关词条完善回答"

        llm_api_url = "http://localhost:11434/api/"  #还没有部署,记得完善api
        llm_payload = {
            "prompt": prompt,
            "max_tokens": 400,   
            "temperature": 0.7
        }

        llm_response = requests.post(llm_api_url, json=llm_payload)
        if llm_response.status_code == 200:
            answer = llm_response.json().get("answer", "")
        else:
            answer = "LLM调用失败"

    except Exception as e:
        return jsonify({"error": f"Rerank failed: {str(e)}"}), 500

    return jsonify({
        "query": query,
        "results": reranked_full,
        "count": len(reranked_full),
        "answer": answer,
        "message": "Search, rerank and generation success"
    }), 200


if __name__ == '__main__':
    app.run(debug=True)
