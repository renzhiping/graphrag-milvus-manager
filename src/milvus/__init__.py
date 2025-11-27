"""
Milvus 数据库集成模块
提供GraphRAG实体向量数据的存储和检索功能
"""

from .core.collection_manager import (
    MilvusCollectionManager,
    COLLECTION_NAMES,
    create_collection_manager_with_connection,
)
from .core.parquet_importer import MilvusParquetImporter
from .legacy.collection_store import MilvusCollectionStore
from .query.client import MilvusClient
from .query.query_manager import MilvusQueryManager

__all__ = [
    'MilvusCollectionManager',
    'COLLECTION_NAMES',
    'create_collection_manager_with_connection',
    'MilvusParquetImporter',
    'MilvusCollectionStore',
    'MilvusClient',
    'MilvusQueryManager'
]

