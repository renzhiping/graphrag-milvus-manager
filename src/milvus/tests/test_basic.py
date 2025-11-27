"""
Milvus 基本功能测试
"""

import argparse
import asyncio
import logging
from typing import Any, Dict

import numpy as np

from milvus.query.client import MilvusClient
from milvus.core.constants import EMBEDDING_DIM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _run_milvus_basic(client_kwargs: Dict[str, Any] | None = None):
    """协程：测试Milvus基本功能（原异步实现）。"""
    client_kwargs = client_kwargs or {}
    
    # 创建模拟的嵌入向量生成函数
    def mock_embedding_generator(text):
        """模拟文本嵌入生成（实际使用时应该替换为真实的嵌入模型）"""
        # 这里使用随机向量作为模拟
        return np.random.rand(EMBEDDING_DIM).tolist()
    
    async with MilvusClient(**client_kwargs) as client:
        logger.info("Milvus客户端初始化成功")
        
        # 测试数据准备
        test_documents = [
            {"source_id": "doc_1", "text": "人工智能是计算机科学的一个分支", "embedding": mock_embedding_generator("人工智能")},
            {"source_id": "doc_2", "text": "机器学习使计算机能够自主学习", "embedding": mock_embedding_generator("机器学习")}
        ]
        
        test_entities = [
            {"source_id": "ent_1", "title": "人工智能", "description": "模拟人类智能的计算机系统", "embedding": mock_embedding_generator("人工智能")},
            {"source_id": "ent_2", "title": "机器学习", "description": "从数据中自动学习的算法", "embedding": mock_embedding_generator("机器学习")}
        ]
        
        # 测试数据插入
        logger.info("测试数据插入...")
        
        # 插入文档数据
        doc_ids = await client.storage.insert_documents(test_documents)
        logger.info(f"插入文档数据成功，ID: {doc_ids}")
        
        # 插入实体标题数据
        entity_title_ids = await client.storage.insert_entity_titles(test_entities)
        logger.info(f"插入实体标题数据成功，ID: {entity_title_ids}")
        
        # 插入实体描述数据
        entity_desc_ids = await client.storage.insert_entity_descriptions(test_entities)
        logger.info(f"插入实体描述数据成功，ID: {entity_desc_ids}")
        
        # 测试查询
        logger.info("测试数据查询...")
        
        # 查询集合统计信息
        doc_stats = client.query_manager.get_collection_stats("document")
        logger.info(f"文档集合统计: {doc_stats}")
        
        # 相似性搜索测试
        query_embedding = mock_embedding_generator("人工智能")
        similar_docs = client.query_manager.search_by_embedding(
            query_embedding, "document", limit=5
        )
        logger.info(f"找到 {len(similar_docs)} 个相似文档")
        
        # 根据源ID查询
        doc_info = client.query_manager.query_by_source_id("document", "doc_1")
        logger.info(f"文档doc_1的信息: {doc_info}")
        
        # 测试文本搜索
        similar_to_ai = client.query_manager.search_by_text(
            "人工智能", "document", limit=3
        )
        logger.info(f"与'人工智能'相似的文档: {len(similar_to_ai)} 个")
        
        # 测试多集合搜索（替代 hybrid_search）
        hybrid_results = client.query_manager.search_multiple_collections(
            "人工智能", ["document", "entity_title"], limit_per_collection=2
        )
        logger.info(f"多集合搜索结果 - 文档: {len(hybrid_results.get('document', []))}, 实体: {len(hybrid_results.get('entity_title', []))}")
        
        logger.info("所有测试完成!")


def test_milvus_basic(client_kwargs: Dict[str, Any] | None = None):
    """同步 pytest 用例：包装异步实现，避免依赖 pytest-asyncio 等插件。"""
    asyncio.run(_run_milvus_basic(client_kwargs))


async def _run_collection_management(client_kwargs: Dict[str, Any] | None = None):
    """协程：测试集合管理功能（原异步实现）。"""
    client_kwargs = client_kwargs or {}
    
    async with MilvusClient(**client_kwargs) as client:
        # 列出所有集合
        collections = client.collection_manager.list_collections()
        logger.info(f"当前集合: {collections}")
        
        # 获取集合统计信息
        for collection_type in ["document", "entity_title", "entity_description"]:
            try:
                stats = client.query_manager.get_collection_stats(collection_type)
                logger.info(f"{collection_type} 集合统计: {stats['num_entities']} 条数据")
            except Exception as e:
                logger.warning(f"获取 {collection_type} 统计失败: {e}")


def test_collection_management(client_kwargs: Dict[str, Any] | None = None):
    """同步 pytest 用例：包装异步集合管理测试。"""
    asyncio.run(_run_collection_management(client_kwargs))


def _build_client_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """根据命令行参数构造 MilvusClient 关键字参数"""
    kwargs: Dict[str, Any] = {}
    if args.embedding_api_key:
        kwargs["embedding_api_key"] = args.embedding_api_key
    if args.embedding_api_base:
        kwargs["embedding_api_base"] = args.embedding_api_base
    if args.embedding_model:
        kwargs["embedding_model"] = args.embedding_model
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus 基础功能测试")
    parser.add_argument(
        "--embedding-api-key",
        help="Qwen/DashScope 嵌入模型 API Key，可用于覆盖环境变量",
    )
    parser.add_argument(
        "--embedding-api-base",
        help="Qwen/DashScope API Base，可选，默认为 DashScope 兼容地址",
    )
    parser.add_argument(
        "--embedding-model",
        help="嵌入模型名称，默认 text-embedding-v3",
    )
    cli_args = parser.parse_args()
    client_kwargs = _build_client_kwargs(cli_args)

    # 运行测试
    asyncio.run(_run_milvus_basic(client_kwargs))
    asyncio.run(_run_collection_management(client_kwargs))
