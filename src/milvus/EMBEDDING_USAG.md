## 向量检索与嵌入管理使用指南

本指南介绍如何使用本项目提供的 `MilvusClient` 完成以下操作：

- **向量检索**：基于文本或向量在 Milvus 集合中搜索相似记录  
- **删除一条记录**：按指定字段删除集合中的记录（包括向量）  
- **新增一条记录**：自动生成嵌入并写入集合  
- **对任意文本生成向量**：只做嵌入计算，不访问 Milvus

---

## 前置条件

- 已经在 Milvus 中创建好对应集合（可用 `src/milvus/scripts/milvus_create_collections.py` 脚本完成）。
- `.env` 或环境变量中配置好：
  - `MILVUS_HOST`, `MILVUS_PORT`
  - `QWEN_API_KEY` 或 `DASHSCOPE_API_KEY`（嵌入模型 API Key）
  - 可选：`MILVUS_EMBEDDING_MODEL`, `QWEN_API_BASE` 等
- 代码可以正常导入：

```python
from milvus import MilvusClient
```

---

## 创建客户端

### 基于环境变量创建

```python
from milvus import MilvusClient

client = MilvusClient.from_env()
client.connect()
```

### 显式传入配置

```python
from milvus import MilvusClient

client = MilvusClient(
    host="localhost",
    port="19530",
    embedding_api_key="your-qwen-or-dashscope-api-key",
    # 可选
    embedding_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    embedding_model="text-embedding-v3",
)
client.connect()
```

使用完成后记得断开连接：

```python
client.disconnect()
```

---

## 向量检索

### 按文本在单个集合中检索

```python
query_text = "人工智能"
collection_type = "text_unit"  # 例如：relationship / entity_title 等

results = client.query_manager.search_by_text(
    query_text=query_text,
    collection_type=collection_type,
    limit=5,
)

for item in results:
    print("score:", item["score"], "text:", item.get("text") or item.get("description"))
```

### 同时在多个集合中检索

```python
query_text = "人工智能"
collection_types = ["relationship", "text_unit", "entity_title"]

results_by_collection = client.query_manager.search_multiple_collections(
    query_text=query_text,
    collection_types=collection_types,
    limit_per_collection=3,
)

for ctype, items in results_by_collection.items():
    print(f"=== {ctype} ===")
    for item in items:
        print("score:", item["score"], "source_id:", item["source_id"])
```

### 基于向量检索

先将文本转换为向量，再按向量检索：

```python
# 文本 -> 向量
embedding = client.embed("人工智能相关的文本")

# 使用 embedding 在集合中搜索
results = client.query_manager.search_by_embedding(
    query_embedding=embedding,
    collection_type="text_unit",
    limit=5,
)
```

---

## 删除一条记录

使用 `MilvusClient.delete_record`，按指定字段删除记录（包括其中的向量）：

```python
# 按 source_id 删除 text_unit 集合中的记录
deleted_count = client.delete_record(
    collection_type="text_unit",
    where_field="source_id",
    where_value="doc_123",
)
print("deleted:", deleted_count)

# 按 title 删除 entity_title 集合中的记录
deleted_count = client.delete_record(
    collection_type="entity_title",
    where_field="title",
    where_value="人工智能",
)
```

- **collection_type**：集合逻辑名（不含前缀），例如  
  `"relationship"`, `"text_unit"`, `"document"`, `"entity_title"`,  
  `"entity_description"`, `"community_title"`, `"community_summary"`, `"community_full_content"`  
- **where_field**：集合 schema 中的任意字段，如 `source_id`、`title`、`description`、`text` 等  
- **where_value**：字段匹配值，支持 `str` 或数字

> 注意：调用前需要先 `client.connect()`。

---

## 新增一条记录（自动生成嵌入）

使用 `MilvusClient.add_embedding_record`，只需提供业务字段，嵌入由内部自动生成。

### 在 text_unit 集合新增一条记录

```python
record = {
    "source_id": "doc_123",
    "text": "这是一个关于人工智能的段落。",
}

inserted = client.add_embedding_record(
    collection_type="text_unit",
    data=record,
)
print("inserted:", inserted)  # 通常为 1
```

### 在 entity_description 集合新增一条记录

```python
record = {
    "source_id": "ent_123",
    "title": "人工智能",
    "description": "模拟人类智能的计算机系统。",
}

inserted = client.add_embedding_record(
    collection_type="entity_description",
    data=record,
)
```

内部逻辑（与 `ParquetImporter` 行为保持一致）：

- 根据 **collection_type** 选择用于生成向量的文本：
  - `relationship`：`description`
  - `text_unit` / `document`：`text`
  - `entity_title` / `community_title`：`title`
  - `entity_description`：`f"{title}:{description}"`（或 `title:summary`）
  - `community_summary`：`summary`
  - `community_full_content`：`full_content`
- 使用上述文本调用 Qwen 嵌入接口生成 `embedding`；
- 组合业务字段 + `embedding`，写入对应集合。

> 注意：调用前同样需要 `client.connect()`，并确保 `embedding_api_key` 配置正确。

---

## 文本向量化（不访问 Milvus）

有些场景只需要对文本做向量化，而不需要访问 Milvus，这种情况下可以直接使用 `embed` / `embed_batch`。

### 单条文本

```python
vec = client.embed("这是需要生成向量的一段文本")
print(len(vec))  # 通常是 1024 维
```

### 批量文本

```python
texts = ["文本一", "文本二", "文本三"]
vectors = client.embed_batch(texts)

for t, v in zip(texts, vectors):
    print(t, "->", len(v))
```

特点：

- 这两个方法 **只依赖嵌入模型配置**（`embedding_api_key` / `embedding_api_base` / `embedding_model`），
  不需要与 Milvus 建立连接；
- 适合在你的服务中独立提供 “文本 -> 向量” 的功能。

---

## 小结

- **向量检索**：使用 `client.query_manager.search_by_text(...)` 或 `search_by_embedding(...)`。
- **删除记录**：使用 `client.delete_record(collection_type, where_field, where_value)`。
- **新增记录**：使用 `client.add_embedding_record(collection_type, data)`，内部自动生成嵌入。
- **文本向量化**：使用 `client.embed(text)` / `client.embed_batch(texts)`，不访问 Milvus。 

