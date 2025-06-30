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
    "tags_sorter": "manual",  # 👈 控制顺序！
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
TXT_FOLDER = 'txts'
CHUNK_FOLDER = 'chunks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TXT_FOLDER, exist_ok=True)
os.makedirs(CHUNK_FOLDER, exist_ok=True)
END_PUNCTUATIONS = ('。', '!', '?', '.', ';', ':', '：', '；', '！', '？','”')


def upload_file(file):
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return save_path

def is_end_of_paragraph(text):
    return text.strip().endswith(END_PUNCTUATIONS)

def is_meaningful_paragraph(text):
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r'[\W\d_]+', text):  
        return False
    if len(text) <= 2 and re.fullmatch(r'[.。、；;…]+', text):
        return False
    return True

def merge_incomplete_paragraphs(paragraphs):
    merged = []
    buffer = ''
    for para in paragraphs:
        if not is_meaningful_paragraph(para):
            continue 

        if not buffer:
            buffer = para
        else:
            buffer += ' ' + para

        if is_end_of_paragraph(para):
            merged.append(buffer.strip())
            buffer = ''

    if buffer:
        merged.append(buffer.strip())

    return merged


def convert_docx_to_txt():
    data = request.get_json()
    docx_path = data.get('docx_path')
    if not docx_path or not os.path.exists(docx_path):
        return jsonify({'error': 'Invalid or missing docx_path'}), 400

    doc = Document(docx_path)
    raw_lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    
    merged_paragraphs = merge_incomplete_paragraphs(raw_lines)
    
    text_content = '\n\n'.join(merged_paragraphs)

    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    txt_path = os.path.join(TXT_FOLDER, base_name + ".txt")
    os.makedirs(TXT_FOLDER, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

    return txt_path


def split_into_sentences(text):
    raw_sentences = re.split(r'(?<=[。！？.!?])\s+', text)
    sentences = [
        re.sub(r'[。！？.!?]+$', '', s).strip()
        for s in raw_sentences
        if s.strip() and not re.fullmatch(r'[\W_]+', s.strip())
    ]
    return sentences

def chunk_text_file(txt_path, max_length=500):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    raw_paragraphs = content.split('\n\n')
    chunks = []

    for para in raw_paragraphs:
        para = para.strip().replace('\n', ' ')
        if not para:
            continue
        if len(para) <= max_length:
            chunks.append(para)
        else:
            sentences = split_into_sentences(para)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())

    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    chunk_path = os.path.join(CHUNK_FOLDER, base_name + ".json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
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
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Chunk Embedding Collection")
    collection = Collection(name, schema)
    
    return collection


def embed_chunks_internal(chunk_path, collection_name):
    with open(chunk_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    texts = json_data.get("chunks") or [item["parent"] for item in json_data.get("parent_child", [])]
    embeddings = [(text, get_ollama_embedding(text)) for text in texts]
    dim = len(embeddings[0][1])
    collection = setup_milvus_collection(name=collection_name, dim=dim)
    texts_only = [t[0] for t in embeddings]
    vectors_only = [t[1] for t in embeddings]
    collection.insert([texts_only, vectors_only])
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
    txt_path = convert_docx_to_txt(docx_path)
    chunk_path = chunk_text_file(txt_path)
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

    # 连接 Milvus
    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)

    # 获取嵌入向量
    try:
        query_embedding = get_qwen_embedding(query)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    # Milvus 检索
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    texts = [hit.entity.get("text") for hit in results[0]]
    
    # rerank 排序
    try:
        reranked = rerank_documents(query, texts)
    except Exception as e:
        return jsonify({"error": f"Rerank failed: {str(e)}"}), 500

    return jsonify({
        "query": query,
        "results": reranked,
        "count": len(reranked),
        "message": "Search and rerank success"
    }), 200



if __name__ == '__main__':
    app.run(debug=True)
