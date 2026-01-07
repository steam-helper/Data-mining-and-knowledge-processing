import streamlit as st
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE,
    INDEX_PARAMS, SEARCH_PARAMS, TOP_K, id_to_doc_map
)

@st.cache_resource
def get_milvus_client():
    """初始化并返回 Milvus Lite 客户端实例"""
    try:
        st.write(f"初始化 Milvus Lite 客户端，数据路径: {MILVUS_LITE_DATA_PATH}")
        os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH), exist_ok=True)
        client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
        st.success("Milvus Lite 客户端初始化成功！")
        return client
    except Exception as e:
        st.error(f"初始化 Milvus Lite 客户端失败: {e}")
        return None

@st.cache_resource
def setup_milvus_collection(_client):
    """确保 Milvus Lite 的指定集合已存在并设置正确"""
    if not _client:
        st.error("Milvus 客户端不可用。")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM
        has_collection = collection_name in _client.list_collections()

        if not has_collection:
            st.write(f"未找到集合 '{collection_name}'，正在创建...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500),
            ]
            schema = CollectionSchema(fields, f"医疗 RAG (dim={dim})")

            _client.create_collection(collection_name=collection_name, schema=schema)
            st.write(f"集合 '{collection_name}' 创建成功。")
            st.write(f"创建索引 ({INDEX_TYPE})...")
            index_params = _client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type=INDEX_METRIC_TYPE,
                params=INDEX_PARAMS
            )
            _client.create_index(collection_name, index_params)
            st.success(f"集合 '{collection_name}' 索引创建成功。")
        else:
            st.write(f"已找到集合: '{collection_name}'。")

        current_count = _client.num_entities(collection_name) if hasattr(_client, 'num_entities') else 0
        st.write(f"集合 '{collection_name}' 已准备好。当前实体数: {current_count}")
        return True
    except Exception as e:
        st.error(f"设置 Milvus 集合 '{COLLECTION_NAME}' 时发生错误: {e}")
        return False

def index_data_if_needed(client, data, embedding_model):
    """检查数据是否需要索引，并使用 MilvusClient 执行索引"""
    global id_to_doc_map  # 修改全局映射

    if not client:
        st.error("Milvus 客户端不可用进行索引。")
        return False

    collection_name = COLLECTION_NAME
    try:
        current_count = client.num_entities(collection_name) if hasattr(client, 'num_entities') else 0
        st.write(f"Milvus 集合 '{collection_name}' 中已有 {current_count} 条数据")

        data_to_index = data[:MAX_ARTICLES_TO_INDEX]  # 限制数据量
        docs_for_embedding = []
        data_to_insert = []  # Milvus 插入数据的字典列表
        temp_id_map = {}  # 临时映射

        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            context = doc.get('context', '') or ""
            corpus_name = doc.get('corpus_name', '') or ""  # Add corpus_name

            content = f"标题: {title}\n摘要: {abstract}\n内容: {context}\nCorpus Name: {corpus_name}".strip()
            if not content:
                continue

            doc_id = i  # 使用索引作为 ID
            temp_id_map[doc_id] = {
                'title': title, 'abstract': abstract, 'context': context, 'corpus_name': corpus_name, 'content': content
            }
            docs_for_embedding.append(content)
            data_to_insert.append({
                "id": doc_id,
                "embedding": None,  # 占位符，待填充
                "content_preview": content[:500]  # Store preview if field exists
            })

        if current_count < len(docs_for_embedding):
            st.warning(f"索引需要插入 {len(docs_for_embedding)} 个文档...")

            st.write(f"生成 {len(docs_for_embedding)} 个嵌入向量...")
            with st.spinner("生成嵌入向量..."):
                embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)

            for i, emb in enumerate(embeddings):
                data_to_insert[i]["embedding"] = emb

            st.write("正在插入数据到 Milvus Lite...")
            try:
                start_insert = time.time()
                res = client.insert(collection_name=collection_name, data=data_to_insert)
                end_insert = time.time()
                inserted_count = len(data_to_insert)
                st.success(f"成功尝试索引 {inserted_count} 条数据，插入耗时: {end_insert - start_insert:.2f} 秒")
                id_to_doc_map.update(temp_id_map)  # 更新全局文档映射
                return True
            except Exception as e:
                st.error(f"插入数据到 Milvus Lite 时发生错误: {e}")
                return False
        else:
            st.write(f"数据索引已完成，当前已有 {current_count} 条数据。")
            if not id_to_doc_map:
                id_to_doc_map.update(temp_id_map)  # 如果没有映射，初始化全局映射
            return True
    except Exception as e:
        st.error(f"数据索引时发生错误: {e}")
        return False

def search_similar_documents(client, query, embedding_model):
    """使用 MilvusClient 搜索与查询相关的文档"""
    if not client or not embedding_model:
        st.error("Milvus 客户端或嵌入模型不可用进行搜索。")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        query_embedding = embedding_model.encode([query])[0]

        search_params = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"]
        }

        res = client.search(**search_params, **SEARCH_PARAMS)

        if not res or not res[0]:
            return [], []

        hit_ids = [hit['id'] for hit in res[0]]
        distances = [hit['distance'] for hit in res[0]]
        return hit_ids, distances
    except Exception as e:
        st.error(f"搜索过程中发生错误: {e}")
        return [], []
