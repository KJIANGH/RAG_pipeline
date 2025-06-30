from pymilvus import connections, Collection, utility

# 先连接 Milvus 服务
connections.connect(host="127.0.0.1", port="19530")

# # 然后加载已有 collection
collection = Collection("embedding_chunk_1")
print(f"Number of entities stored in embedding_chunk_1: {collection.num_entities}")

# collection.load()
# results = collection.query(expr="id >= 0", output_fields=["id", "text"], limit=3)
# print(results)

# def drop_collection_if_exists(name):
#     if utility.has_collection(name):
#         utility.drop_collection(name)
#         print(f"Collection '{name}' dropped.")
#     else:
#         print(f"Collection '{name}' does not exist.")

# drop_collection_if_exists("embedding_chunk_1")