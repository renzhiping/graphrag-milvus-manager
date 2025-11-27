#!/usr/bin/env python3
"""
简化的Milvus Collection管理器
"""

import logging
from typing import Dict, Optional
from pymilvus import Collection, utility, connections, CollectionSchema

from .config import get_milvus_config
from .schema import COLLECTION_CONFIGS

# 配置日志
logger = logging.getLogger(__name__)

# 默认集合前缀
DEFAULT_COLLECTION_PREFIX = "graphrag_"

# Milvus集合配置

COLLECTION_TYPES = list(COLLECTION_CONFIGS.keys())
COLLECTION_NAMES = {
    collection_type: f"{DEFAULT_COLLECTION_PREFIX}{collection_type}"
    for collection_type in COLLECTION_TYPES
}


class MilvusCollectionManager:
    """
    Milvus Collection 管理器
    负责 collection 的创建、删除和操作，并维护与 Milvus 的连接状态。
    """
    
    def __init__(self, collection_prefix: str = DEFAULT_COLLECTION_PREFIX):
        """
        初始化 Collection 管理器
        
        Args:
            collection_prefix: 集合前缀，默认为 DEFAULT_COLLECTION_PREFIX
        """
        # 确保collection_prefix不为空
        self.collection_prefix = collection_prefix or DEFAULT_COLLECTION_PREFIX
        self.collections: Dict[str, Collection] = {}
        self._connected: Optional[bool] = False
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        logger.info(f"初始化 Collection 管理器，集合前缀: {self.collection_prefix}")
    
    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        建立到 Milvus 的连接。
        
        注意：
        - 为了避免在业务代码中隐式读取配置，这里不再从环境或配置中推导 host/port；
        - 调用方（通常是 MilvusClient）必须显式传入 host 和 port。
        """
        if self._connected:
            logger.info("Milvus 已连接，跳过重复连接")
            return
        
        if host is None or port is None:
            raise ValueError(
                "连接 Milvus 时必须显式提供 host 和 port，"
                "请通过 MilvusClient 或在调用 connect 时传入这两个参数。"
            )
        self._host = host
        self._port = int(port)
        
        try:
            connections.connect(host=self._host, port=self._port)
            self._connected = True
            logger.info(f"成功连接到 Milvus: {self._host}:{self._port}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"连接 Milvus 失败: {e}")
            raise
    
    def disconnect(self) -> None:
        """断开与 Milvus 的连接。"""
        if not self._connected:
            return
        
        try:
            connections.disconnect("default")
            logger.info("已断开 Milvus 连接")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"断开 Milvus 连接时发生异常: {e}")
        finally:
            self._connected = False
    
    def create_collections(self) -> Dict[str, Collection]:
        """
        创建所有需要的集合
        
        Returns:
            Dict[str, Collection]: 集合名称到Collection对象的映射
        
        注意：假设 Milvus 连接已建立
        """
        
        created_collections = {}
        
        for collection_name, config in COLLECTION_CONFIGS.items():
            full_name = f"{self.collection_prefix}{collection_name}"
            
            if utility.has_collection(full_name):
                logger.info(f"集合 '{full_name}' 已存在，直接加载")
                collection = Collection(full_name)
                self.collections[collection_name] = collection
                created_collections[collection_name] = collection
                continue
            
            # 创建新集合
            logger.info(f"创建新集合: {full_name}")
            schema = CollectionSchema(
                fields=config["fields"],
                description=config["description"]
            )
            
            collection = Collection(full_name, schema)
            
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            
            self.collections[collection_name] = collection
            created_collections[collection_name] = collection
            logger.info(f"集合 '{full_name}' 创建成功")
        
        return created_collections
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        获取集合实例
        
        Args:
            collection_name: 集合名称（不包含前缀）
            
        Returns:
            Collection: 集合实例
        """
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        # 如果集合不在缓存中，尝试从数据库加载
        full_name = f"{self.collection_prefix}{collection_name}"
        if not utility.has_collection(full_name):
            raise ValueError(f"集合 '{full_name}' 不存在，请先创建集合")
        
        collection = Collection(full_name)
        self.collections[collection_name] = collection
        logger.info(f"从数据库加载集合: {full_name}")
        return collection
    
    def list_collections(self) -> list[str]:
        """
        列出所有GraphRAG相关的集合
        
        注意：假设 Milvus 连接已建立
        """
        
        all_collections = utility.list_collections()
        # 确保collection_prefix不为空，避免startswith("")匹配所有集合
        if not self.collection_prefix:
            logger.warning("collection_prefix为空，返回空列表")
            return []
            
        graphrag_collections = [
            name for name in all_collections 
            if name.startswith(self.collection_prefix)
        ]
        return graphrag_collections
    
    def drop_collections(self) -> int:
        """
        删除所有GraphRAG相关的集合
        
        Returns:
            int: 删除的集合数量
        
        注意：假设 Milvus 连接已建立
        """
        
        dropped_count = 0
        
        # 删除已知的集合
        for collection_name in COLLECTION_CONFIGS.keys():
            full_name = f"{self.collection_prefix}{collection_name}"
            if utility.has_collection(full_name):
                utility.drop_collection(full_name)
                logger.info(f"删除集合: {full_name}")
                dropped_count += 1
                
                # 从缓存中移除
                if collection_name in self.collections:
                    del self.collections[collection_name]
        
        # 删除其他可能存在的GraphRAG集合
        all_collections = utility.list_collections()
        for collection_name in all_collections:
            if (collection_name.startswith(self.collection_prefix) and 
                not any(collection_name.endswith(known) for known in COLLECTION_CONFIGS.keys())):
                utility.drop_collection(collection_name)
                logger.info(f"删除额外集合: {collection_name}")
                dropped_count += 1
        
        return dropped_count
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        检查集合是否存在
        
        Args:
            collection_name: 集合名称（不包含前缀）
            
        Returns:
            bool: 集合是否存在
        
        注意：假设 Milvus 连接已建立
        """
        
        full_name = f"{self.collection_prefix}{collection_name}"
        return utility.has_collection(full_name)
    
    def get_collection_info(self, collection_name: str) -> dict:
        """获取集合基本信息"""
        collection = self.get_collection(collection_name)
        return {
            "name": collection.name,
            "num_entities": collection.num_entities
        }


def create_collection_manager_with_connection(
    host: str = "localhost",
    port: int = 19530,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX
) -> MilvusCollectionManager:
    """
    便捷函数：创建 Collection Manager 并建立 Milvus 连接
    
    此函数用于独立脚本，自动处理连接管理。
    如果在应用中使用，推荐使用 MilvusClient 代替。
    
    Args:
        host: Milvus服务器地址
        port: Milvus服务器端口
        collection_prefix: 集合前缀
        
    Returns:
        MilvusCollectionManager: 已连接的 collection manager
    """
    manager = MilvusCollectionManager(collection_prefix=collection_prefix)
    manager.connect(host=host, port=port)
    return manager


def main(host: Optional[str] = None, port: Optional[int] = None, collection_prefix: Optional[str] = None):
    """
    主函数 - 创建和初始化所有集合
    
    Args:
        host: Milvus服务器地址，如果为None则从配置文件读取
        port: Milvus服务器端口，如果为None则从配置文件读取
        collection_prefix: 集合前缀，如果为None则从配置文件读取
    """
    print("🔗 连接到Milvus并创建集合...")
    
    # 如果没有提供参数，从配置文件读取
    if host is None or port is None:
        config = get_milvus_config()
        host = host or config.host or "localhost"
        port = port or config.port or 19530
        collection_prefix = collection_prefix or config.collection_prefix or DEFAULT_COLLECTION_PREFIX
    else:
        collection_prefix = collection_prefix or DEFAULT_COLLECTION_PREFIX
    
    # 使用辅助函数创建 manager
    manager = create_collection_manager_with_connection(
        host=host,
        port=port,
        collection_prefix=collection_prefix,
    )
    
    try:
        collections = manager.create_collections()
        
        print(f"✅ 成功创建/验证 {len(collections)} 个集合:")
        for name in collections.keys():
            full_name = f"{manager.collection_prefix}{name}"
            print(f"  - {full_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        print(f"❌ 错误: {e}")
        return False
    finally:
        # 断开连接
        manager.disconnect()
        logger.info("已断开 Milvus 连接")


if __name__ == "__main__":
    main()
