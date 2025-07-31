from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
import os

def save_bm25_texts(docs, file_path, index_dir="bm25_index"):
    os.makedirs(index_dir, exist_ok=True)

    schema = Schema(
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        path=ID(stored=True, unique=True)
    )

    if not os.listdir(index_dir):
        ix = create_in(index_dir, schema)
    else:
        ix = open_dir(index_dir)

    writer = ix.writer()
    for i, doc in enumerate(docs):
        text = doc.page_content
        path = f"{file_path}_chunk_{i}"
        writer.add_document(content=text, path=path)
    writer.commit()


def bm25_search(query, top_k=5, index_dir="bm25_index"):
    ix = open_dir(index_dir)
    results_list = []
    
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        myquery = parser.parse(query)
        
        # 执行搜索
        results = searcher.search(myquery, limit=top_k)
        
        for hit in results:
            results_list.append({
                "content": hit["content"],
                "parent_id": None,
                "filename": hit.get("filename", None),
                "section": hit.get("section", None),
                "term": hit.get("term", None),
                "score": hit.score,
                "source": "bm25"
            })
    
    return results_list