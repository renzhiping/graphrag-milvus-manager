# Milvus 向量数据库集成

## 概述

本模块提供了GraphRAG系统与Milvus向量数据库的集成，用于存储和检索实体嵌入向量数据。

## 功能特性

- ✅ **多集合支持**: 支持8种不同的实体类型集合
- ✅ **批量操作**: 支持批量插入和查询
- ✅ **相似性搜索**: 基于向量相似度的语义搜索
- ✅ **混合搜索**: 跨多个集合的联合搜索
- ✅ **数据管理**: 完整的CRUD操作
- ✅ **异步支持**: 全异步API设计

## 集合设计

基于 `generate_text_embeddings.py` 的向量化配置，设计了8个集合：

| 集合名称 | 描述 | 字段 |
|---------|------|------|
| `graphrag_document` | 文档内容嵌入 | source_id, text, embedding |
| `graphrag_relationship` | 关系描述嵌入 | source_id, description, embedding |
| `graphrag_text_unit` | 文本单元嵌入 | source_id, text, embedding |
| `graphrag_entity_title` | 实体标题嵌入 | source_id, title, embedding |
| `graphrag_entity_description` | 实体描述组合嵌入 | source_id, title, description, title_description, embedding |
| `graphrag_community_title` | 社区标题嵌入 | source_id, title, embedding |
| `graphrag_community_summary` | 社区摘要嵌入 | source_id, summary, embedding |
| `graphrag_community_full_content` | 社区完整内容嵌入 | source_id, full_content, embedding |

## 安装依赖

```bash
pip install pymilvus
```

## 快速开始

### 1. 启动Milvus服务

```bash
# 在dgraph目录下
docker-compose up -d
```

### 2. 基本使用示例

```python
import asyncio
from milvus import MilvusManager

async def main():
    async with MilvusManager() as manager:
        # 插入文档数据
        documents = [
            {
                "source_id": "doc_1", 
                "text": "人工智能概述",
                "embedding": [0.1, 0.2, ..., 0.4096]  # 4096维向量
            }
        ]
        
        doc_ids = await manager.storage.insert_documents(documents)
        print(f"插入文档ID: {doc_ids}")
        
        # 相似性搜索
        query_embedding = [0.1, 0.2, ..., 0.4096]  # 查询向量
        results = await manager.query.search_similar("document", query_embedding, limit=5)
        print(f"搜索结果: {results}")

asyncio.run(main())
```

### 3. 与GraphRAG集成

```python
from milvus import MilvusManager
from graphrag.index.operations.embed_text import embed_text

async def store_embeddings(entities_df, relationships_df, text_units_df):
    """将GraphRAG生成的嵌入存储到Milvus"""
    async with MilvusManager() as manager:
        
        # 存储实体嵌入
        if entities_df is not None:
            await manager.storage.batch_insert_from_dataframe(
                "entity_title", entities_df, "title_embedding"
            )
            await manager.storage.batch_insert_from_dataframe(
                "entity_description", entities_df, "description_embedding"
            )
        
        # 存储关系嵌入
        if relationships_df is not None:
            await manager.storage.batch_insert_from_dataframe(
                "relationship", relationships_df, "description_embedding"
            )
        
        # 存储文本单元嵌入
        if text_units_df is not None:
            await manager.storage.batch_insert_from_dataframe(
                "text_unit", text_units_df, "text_embedding"
            )
```

## API 参考

### MilvusManager

主管理器类，提供统一的接口：

- `initialize()`: 初始化连接和集合
- `storage`: 数据存储操作
- `query`: 数据查询操作
- `close()`: 关闭连接

### MilvusCollectionStore

运行期集合存储层，封装常用写操作：

- `insert_documents(documents)`: 插入文档数据
- `insert_entity_titles(entities)`: 插入实体标题数据
- `insert_entity_descriptions(entities)`: 插入实体描述数据
- `batch_insert_from_dataframe()`: 从DataFrame批量插入
- `delete_by_source_ids()`: 根据源ID删除数据
- `clear_collection()`: 清空集合

### MilvusQuery

数据查询操作：

- `search_similar()`: 相似性搜索
- `search_similar_text()`: 基于文本的相似性搜索
- `query_by_source_id()`: 根据源ID查询
- `query_by_ids()`: 根据ID列表查询
- `get_collection_stats()`: 获取集合统计
- `hybrid_search()`: 混合搜索
- `batch_search()`: 批量搜索

## 配置说明

### 向量维度

所有嵌入向量统一为 **4096维度**，与 Qwen `text-embedding-v3` 模型保持一致。

### 索引配置

- **索引类型**: IVF_FLAT
- **距离度量**: L2（欧几里得距离）
- **nlist参数**: 128
- **nprobe参数**: 10

### 连接配置

- **主机**: localhost
- **gRPC端口**: 19530
- **HTTP端口**: 9091

## 性能优化

1. **批量操作**: 使用批量插入提高性能
2. **连接池**: 管理数据库连接
3. **索引优化**: 根据数据量调整nlist参数
4. **缓存**: 使用PipelineCache缓存嵌入结果

## 测试

运行基本功能测试：

```bash
python -m milvus.test_basic
```

## 部署说明

### Docker Compose 部署

使用项目根目录的 `docker-compose.yml` 文件部署Milvus：

```bash
docker-compose up -d
docker-compose ps  # 检查状态
```

### 健康检查

```bash
curl -X GET "http://localhost:9091/healthz"
```

## 相关文档

- [嵌入字段定义](./EMBEDDING_FIELDS.md)
- [Milvus官方文档](https://milvus.io/docs)
- [GraphRAG嵌入生成](../workflows/generate_text_embeddings.py)

---
**最后更新**: 2025-08-23
