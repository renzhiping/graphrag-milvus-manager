#!/usr/bin/env python3
"""
Milvus查询管理器
专门负责向量查询，不包含collection创建逻辑
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, cast

from pymilvus import Collection

from ..core.collection_manager import MilvusCollectionManager
from ..core.constants import DEFAULT_QWEN_EMBEDDING_MODEL
from ..core.embedding_generator import QwenEmbeddingGenerator

logger = logging.getLogger(__name__)


class MilvusQueryManager:
    """
    Milvus查询管理器
    专门负责向量查询，依赖于已存在的collection
    """
    
    def __init__(
        self,
        collection_manager: MilvusCollectionManager | None = None, 
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
    ):
        """
        初始化查询管理器
        
        Args:
            collection_manager: Collection管理器实例，如果为None则创建新实例
            embedding_model: 嵌入模型名称，默认使用 Qwen text-embedding-v3
            embedding_api_key: Qwen/DashScope API Key，默认读取环境变量
            embedding_api_base: Qwen API Base，默认 DashScope 兼容地址
        
        注意：如果传入 None，会创建新的 MilvusCollectionManager，
             但不会建立连接，需要外部先建立连接
        """
        self.collection_manager = collection_manager or MilvusCollectionManager()

        # 使用默认的embedding模型名称
        self.embedding_model_name = embedding_model or DEFAULT_QWEN_EMBEDDING_MODEL

        # 为了避免在业务代码中隐式读取环境变量，embedding_api_key 必须由外部显式传入
        if embedding_api_key is None:
            raise ValueError(
                "初始化 MilvusQueryManager 需要显式提供 embedding_api_key，"
                "请在创建时通过参数 embedding_api_key 传入 Qwen/DashScope API Key。"
            )

        self.embedding_generator = QwenEmbeddingGenerator(
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            model=self.embedding_model_name,
            name="milvus_query_manager",
        )
        
        logger.info(f"初始化查询管理器")
        logger.info(
            "使用 Qwen 嵌入模型: %s，维度=%s",
            self.embedding_model_name,
            self.embedding_generator.dimension,
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成单个文本的embedding向量"""
        clean_text = str(text).strip() if text else "空内容"
        try:
            return self.embedding_generator.embed(clean_text)
        except Exception as e:  # noqa: BLE001
            logger.error(f"生成embedding失败，返回零向量: {e}")
            return self.embedding_generator.zero_vector()
    
    def search_by_text(self, query_text: str, collection_type: str, 
                      limit: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        通过文本查询相似向量
        
        Args:
            query_text: 查询文本
            collection_type: 目标集合类型
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            # 检查集合是否存在
            if not self.collection_manager.collection_exists(collection_type):
                logger.warning(f"集合 '{collection_type}' 不存在")
                return []
            
            # 生成查询向量
            query_embedding = self._generate_embedding(query_text)
            
            # 执行向量搜索
            return self.search_by_embedding(query_embedding, collection_type, limit, score_threshold)
            
        except Exception as e:
            logger.error(f"文本查询失败: {e}")
            return []
    
    def search_by_embedding(self, query_embedding: List[float], collection_type: str,
                           limit: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        通过embedding向量查询相似向量
        
        Args:
            query_embedding: 查询向量
            collection_type: 目标集合类型
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        try:
            # 确保连接到Milvus（连接应由上层 MilvusClient 或调用方统一管理）
            if not self.collection_manager._connected:
                raise RuntimeError(
                    "MilvusCollectionManager 尚未连接，请先通过 MilvusClient.connect() "
                    "或手动调用 collection_manager.connect(host, port) 建立连接。"
                )
            
            # 获取集合
            collection = self.collection_manager.get_collection(collection_type)
            
            # 加载集合到内存（如果尚未加载）
            collection.load()
            
            # 设置搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            search_future = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=self._get_output_fields(collection_type)
            )
            results = cast(Iterable, search_future)
            
            # 处理结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    if hit.distance <= score_threshold:
                        continue
                    
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0 / (1.0 + hit.distance),  # 转换为相似度分数
                    }
                    
                    # 添加字段数据
                    for field_name in self._get_output_fields(collection_type):
                        if hasattr(hit.entity, field_name):
                            result[field_name] = getattr(hit.entity, field_name)
                    
                    formatted_results.append(result)
            
            logger.info(f"在集合 {collection_type} 中找到 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量查询失败: {e}")
            return []
    
    def batch_search_by_embeddings(self, query_embeddings: List[List[float]], 
                                  collection_type: str, limit: int = 5) -> List[List[Dict[str, Any]]]:
        """
        批量向量查询
        
        Args:
            query_embeddings: 查询向量列表
            collection_type: 目标集合类型
            limit: 每个查询的返回结果数量限制
            
        Returns:
            List[List[Dict[str, Any]]]: 每个查询的结果列表
        """
        try:
            # 确保连接到Milvus（连接应由上层 MilvusClient 或调用方统一管理）
            if not self.collection_manager._connected:
                raise RuntimeError(
                    "MilvusCollectionManager 尚未连接，请先通过 MilvusClient.connect() "
                    "或手动调用 collection_manager.connect(host, port) 建立连接。"
                )
            
            # 获取集合
            collection = self.collection_manager.get_collection(collection_type)
            
            # 加载集合到内存（如果尚未加载）
            collection.load()
            
            # 设置搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行批量搜索
            search_future = collection.search(
                data=query_embeddings,
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=self._get_output_fields(collection_type)
            )
            results = cast(Iterable, search_future)
            
            # 处理结果
            all_results = []
            for hits in results:
                query_results = []
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0 / (1.0 + hit.distance),
                    }
                    
                    # 添加字段数据
                    for field_name in self._get_output_fields(collection_type):
                        if hasattr(hit.entity, field_name):
                            result[field_name] = getattr(hit.entity, field_name)
                    
                    query_results.append(result)
                
                all_results.append(query_results)
            
            logger.info(f"批量查询完成，处理了 {len(query_embeddings)} 个查询")
            return all_results
            
        except Exception as e:
            logger.error(f"批量查询失败: {e}")
            return [[] for _ in query_embeddings]
    
    def _get_output_fields(self, collection_type: str) -> List[str]:
        """获取集合的输出字段"""
        field_mapping = {
            "relationship": ["source_id", "description"],
            "text_unit": ["source_id", "text"],
            "entity_title": ["source_id", "title"],
            "entity_description": ["source_id", "title", "description", "title_description"],
            "community_title": ["source_id", "title"],
            "community_summary": ["source_id", "summary"],
            "community_full_content": ["source_id", "full_content"]
        }
        
        return field_mapping.get(collection_type, ["source_id"])
    
    def get_collection_stats(self, collection_type: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_type: 集合类型
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if not self.collection_manager.collection_exists(collection_type):
                return {"error": f"集合 '{collection_type}' 不存在"}
            
            collection = self.collection_manager.get_collection(collection_type)
            
            return {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "description": collection.description,
                "is_loaded": bool(getattr(collection, "is_loaded", False))
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def search_multiple_collections(self, query_text: str, collection_types: List[str],
                                   limit_per_collection: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        在多个集合中搜索
        
        Args:
            query_text: 查询文本
            collection_types: 目标集合类型列表
            limit_per_collection: 每个集合的返回结果数量限制
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 每个集合的查询结果
        """
        results = {}
        
        # 生成查询向量（只生成一次）
        query_embedding = self._generate_embedding(query_text)
        
        for collection_type in collection_types:
            try:
                collection_results = self.search_by_embedding(
                    query_embedding, collection_type, limit_per_collection
                )
                results[collection_type] = collection_results
            except Exception as e:
                logger.error(f"在集合 {collection_type} 中搜索失败: {e}")
                results[collection_type] = []
        
        return results
    
    def query_by_source_id(self, collection_type: str, source_id: str,
                          output_fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        根据源ID查询数据（非向量查询）
        
        Args:
            collection_type: 集合类型
            source_id: 源ID
            output_fields: 需要返回的字段列表，默认为 ["id", "source_id"]
            
        Returns:
            Optional[Dict[str, Any]]: 查询结果，如果未找到则返回None
        """
        try:
            # 确保连接到Milvus（连接应由上层 MilvusClient 或调用方统一管理）
            if not self.collection_manager._connected:
                raise RuntimeError(
                    "MilvusCollectionManager 尚未连接，请先通过 MilvusClient.connect() "
                    "或手动调用 collection_manager.connect(host, port) 建立连接。"
                )
            
            # 获取集合
            collection = self.collection_manager.get_collection(collection_type)
            collection.load()
            
            # 设置默认输出字段
            if output_fields is None:
                output_fields = ["id", "source_id"]
            
            # 执行查询
            results = collection.query(
                expr=f"source_id == '{source_id}'",
                output_fields=output_fields
            )
            
            if results:
                logger.info(f"在集合 {collection_type} 中找到 source_id='{source_id}' 的数据")
                return results[0]
            
            logger.info(f"在集合 {collection_type} 中未找到 source_id='{source_id}' 的数据")
            return None
            
        except Exception as e:
            logger.error(f"根据源ID查询失败: {e}")
            raise
    
    def query_by_ids(self, collection_type: str, ids: List[int],
                    output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据ID列表批量查询数据（非向量查询）
        
        Args:
            collection_type: 集合类型
            ids: ID列表
            output_fields: 需要返回的字段列表，默认为 ["id", "source_id"]
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        if not ids:
            return []
        
        try:
            # 确保连接到Milvus（连接应由上层 MilvusClient 或调用方统一管理）
            if not self.collection_manager._connected:
                raise RuntimeError(
                    "MilvusCollectionManager 尚未连接，请先通过 MilvusClient.connect() "
                    "或手动调用 collection_manager.connect(host, port) 建立连接。"
                )
            
            # 获取集合
            collection = self.collection_manager.get_collection(collection_type)
            collection.load()
            
            # 设置默认输出字段
            if output_fields is None:
                output_fields = ["id", "source_id"]
            
            # 构建查询表达式
            ids_str = ", ".join(map(str, ids))
            
            # 执行查询
            results = collection.query(
                expr=f"id in [{ids_str}]",
                output_fields=output_fields
            )
            
            logger.info(f"在集合 {collection_type} 中查询到 {len(results)}/{len(ids)} 条数据")
            return results
            
        except Exception as e:
            logger.error(f"根据ID列表查询失败: {e}")
            raise


def main():
    """主函数 - 用于测试查询功能（通过统一的 MilvusClient）"""
    from dotenv import load_dotenv
    from milvus import MilvusClient

    # 读取环境变量配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    load_dotenv(os.path.join(project_root, ".env"))

    # 基于环境变量创建客户端
    client = MilvusClient.from_env()
    print(f"ℹ️ 使用的Milvus配置: {client.host}:{client.port}")

    collection_manager = client.collection_manager
    
    try:
        # 检查集合是否存在
        client.connect()

        existing_collections = collection_manager.list_collections()
        if not existing_collections:
            print("❌ 未找到任何集合，请先运行collection_manager.py创建集合")
            return False
        
        print(f"✅ 找到 {len(existing_collections)} 个现有集合")
        
        # 创建查询管理器
        query_manager = MilvusQueryManager(collection_manager)
        
        # 测试文本搜索
        print("\n🔍 测试文本搜索...")
        # 假设有一个名为 "document" 的集合用于文本搜索
        # 实际使用时请替换为你的集合类型
        text_search_results = query_manager.search_by_text("人工智能", "text_unit", limit=3)
        print(f"找到 {len(text_search_results)} 个结果")
        for i, result in enumerate(text_search_results, 1):
            print(f"  {i}. 相似度: {result.get('score', 0):.3f}, 文本: {result.get('text', '')[:50]}...")

        # 测试查询
        test_query = "人工智能"
        print(f"\n🔍 测试查询: '{test_query}'")
        
        # 在所有集合中搜索
        available_types = ["relationship", "text_unit", "entity_title", "entity_description"]
        results = query_manager.search_multiple_collections(test_query, available_types, limit_per_collection=3)
        
        # 输出结果
        print("\n📊 查询结果:")
        print("=" * 50)
        for collection_type, collection_results in results.items():
            print(f"\n{collection_type}: {len(collection_results)} 个结果")
            for i, result in enumerate(collection_results[:2], 1):  # 只显示前2个结果
                print(f"  {i}. 相似度: {result.get('score', 0):.3f}")
                if 'title' in result:
                    print(f"     标题: {result['title'][:50]}...")
                elif 'description' in result:
                    print(f"     描述: {result['description'][:50]}...")
                elif 'text' in result:
                    print(f"     文本: {result['text'][:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"查询测试出错: {e}")
        print(f"❌ 查询测试失败: {e}")
        return False
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
