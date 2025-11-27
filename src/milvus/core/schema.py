"""
Milvus 集合配置和 Parquet 映射定义
"""

from pymilvus import DataType, FieldSchema

from .constants import EMBEDDING_DIM

# Milvus集合配置
COLLECTION_CONFIGS = {
    "document": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "文档内容存储集合"
    },
    "relationship": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "关系描述存储集合"
    },
    "text_unit": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "文本单元存储集合"
    },
    "entity_title": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "实体标题嵌入存储集合"
    },
    "entity_description": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="title_description", dtype=DataType.VARCHAR, max_length=2500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "实体标题和描述组合嵌入存储集合"
    },
    "community_title": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "社区标题嵌入存储集合"
    },
    "community_summary": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "社区摘要嵌入存储集合"
    },
    "community_full_content": {
        "fields": [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="full_content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ],
        "description": "社区完整内容嵌入存储集合"
    }
}

# Parquet文件到集合的映射
PARQUET_MAPPING = {
    "relationships.parquet": "relationship", 
    "text_units.parquet": "text_unit",
    "entities.parquet": "entity_title",  # 实体数据使用实体标题集合
    "communities.parquet": "community_title",  # 社区数据使用社区标题集合
    "community_reports.parquet": "entity_description"  # 社区报告使用实体描述集合
}
