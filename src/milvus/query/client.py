"""
Milvus 客户端模块
提供 Milvus 数据库的统一访问接口
整合 collection 管理、查询和存储功能
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

from ..core.config import get_milvus_config
from ..core.collection_manager import (
    MilvusCollectionManager,
    COLLECTION_TYPES,
    DEFAULT_COLLECTION_PREFIX,
)
from .query_manager import MilvusQueryManager
from ..legacy.collection_store import MilvusCollectionStore

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus 客户端
    应用层统一入口，整合 collection 管理、查询和存储功能
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        embedding_model: str | None = None,
        use_lite: bool = False,
        db_path: str | None = None,
    ):
        # 存储连接信息
        self.host = host
        self.port = int(port) if isinstance(port, str) else port
        self._connected = False
        # 预留 Lite 模式参数（当前主要使用服务器模式）
        self.use_lite = use_lite
        self.db_path = db_path
        
        # 创建 Collection 管理器（不传递连接信息）
        self.collection_manager = MilvusCollectionManager(
            collection_prefix=DEFAULT_COLLECTION_PREFIX,
        )
        self.storage = MilvusCollectionStore()

        # 为了保证敏感配置（如 embedding_api_key）由外部项目显式传入，
        # 这里不再为其提供隐式默认值；如果缺失则抛出异常，提醒调用方补全配置。
        if embedding_api_key is None:
            raise ValueError(
                "初始化 MilvusClient 需要显式提供 embedding_api_key，"
                "请通过参数 embedding_api_key 传入 Qwen/DashScope API Key。"
            )

        self.query_manager = MilvusQueryManager(
            self.collection_manager,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
        )
        # 向后兼容：query 属性指向 query_manager
        self.query = self.query_manager
        self.initialized = False

    # ===== 嵌入生成相关 API（对外可独立调用） =====

    def embed(self, text: str) -> List[float]:
        """
        生成单条文本的嵌入向量。

        注意：此方法只调用底层 Qwen 嵌入服务，不依赖 Milvus 连接，
        前提是在初始化 MilvusClient 时已经正确传入 embedding_api_key 等参数。
        """
        return self.query_manager.embedding_generator.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入向量。

        Args:
            texts: 文本列表
        """
        return self.query_manager.embedding_generator.embed_batch(texts)

    # ===== 单条记录的增删操作（基于嵌入集合） =====

    def delete_record(
        self,
        collection_type: str,
        where_field: str,
        where_value: Any,
    ) -> int:
        """
        根据指定字段删除集合中的一条或多条记录（包括其中的 embedding 向量）。

        实际删除逻辑委托给底层的 MilvusCollectionStore，仅在此处做连接状态校验。
        """
        if not self._connected:
            raise RuntimeError("MilvusClient 尚未连接，请先调用 connect()")

        return self.storage.delete_by_field(
            collection_type=collection_type,
            field=where_field,
            value=where_value,
        )

    def add_embedding_record(
        self,
        collection_type: str,
        data: Dict[str, Any],
    ) -> int:
        """
        新增一条带有 embedding 的记录到指定集合。

        此方法会根据 collection_type 推断用于生成 embedding 的文本字段，
        行为与 Parquet 导入器保持一致：
        - relationship: 使用 description
        - text_unit: 使用 text
        - entity_title / community_title: 使用 title
        - entity_description: 使用 \"title:description\" 或 \"title:summary\"
        - community_summary: 使用 summary
        - community_full_content: 使用 full_content

        Args:
            collection_type: 集合类型
            data: 字段字典，至少需要包含 source_id 及对应文本字段

        Returns:
            int: 成功插入的记录数量（0 或 1）
        """
        if not self._connected:
            raise RuntimeError("MilvusClient 尚未连接，请先调用 connect()")

        if "source_id" not in data:
            raise ValueError("新增记录时必须提供 'source_id' 字段")

        # 根据集合类型确定用于生成 embedding 的文本内容
        text_to_embed: str
        record: Dict[str, Any] = {"source_id": str(data.get("source_id", ""))}

        if collection_type == "relationship":
            desc = str(data.get("description", ""))
            record["description"] = desc
            text_to_embed = desc
        elif collection_type == "text_unit":
            text = str(data.get("text", ""))
            record["text"] = text
            text_to_embed = text
        elif collection_type == "entity_title":
            title = str(data.get("title", ""))
            record["title"] = title
            text_to_embed = title
        elif collection_type == "entity_description":
            title = str(data.get("title", ""))
            # 兼容 description / summary 两种来源
            desc = ""
            if "description" in data:
                desc = str(data.get("description", ""))
            elif "summary" in data:
                desc = str(data.get("summary", ""))
            combined = f"{title}:{desc}"
            record.update(
                {
                    "title": title,
                    "description": desc,
                    "title_description": combined,
                }
            )
            text_to_embed = combined
        elif collection_type == "community_title":
            title = str(data.get("title", ""))
            record["title"] = title
            text_to_embed = title
        elif collection_type == "community_summary":
            summary = str(data.get("summary", ""))
            record["summary"] = summary
            text_to_embed = summary
        elif collection_type == "community_full_content":
            full_content = str(data.get("full_content", ""))
            record["full_content"] = full_content
            text_to_embed = full_content
        elif collection_type == "document":
            text = str(data.get("text", ""))
            record["text"] = text
            text_to_embed = text
        else:
            raise ValueError(f"不支持的集合类型: {collection_type}")

        # 生成 embedding
        embedding = self.query_manager.embedding_generator.embed(text_to_embed or "空内容")
        record["embedding"] = embedding

        # 实际插入逻辑委托给底层的 MilvusCollectionStore
        return self.storage.insert_single_record(collection_type, record)

    @classmethod
    def from_env(cls) -> "MilvusClient":
        """基于环境变量配置创建客户端，统一读取配置和连接信息。"""
        config = get_milvus_config()

        # from_env 作为便捷工厂方法，可以在脚本/CLI 中使用，
        # 这里集中处理环境变量到参数的映射，但业务代码应优先使用显式传参方式。
        import os

        embedding_api_key = (
            os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("GRAPHRAG_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        if not embedding_api_key:
            raise RuntimeError(
                "使用 MilvusClient.from_env() 需要在环境变量中提供嵌入模型 API Key，"
                "例如 QWEN_API_KEY 或 DASHSCOPE_API_KEY。"
            )

        embedding_api_base = (
            os.environ.get("QWEN_API_BASE")
            or os.environ.get("DASHSCOPE_API_BASE")
            or None
        )

        return cls(
            host=config.host or "localhost",
            port=str(config.port or 19530),
            embedding_model=config.embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
        )
    
    def connect(self) -> None:
        """连接到 Milvus"""
        if self._connected:
            logger.info("已经连接到 Milvus，跳过重复连接")
            return
        
        try:
            # 通过 CollectionManager 统一管理底层连接
            self.collection_manager.connect(host=self.host, port=self.port)
            self._connected = True
            logger.info(f"成功连接到 Milvus: {self.host}:{self.port}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"连接 Milvus 失败: {e}")
            raise
    
    def disconnect(self) -> None:
        """断开 Milvus 连接"""
        if not self._connected:
            return
        
        try:
            self.collection_manager.disconnect()
            self._connected = False
            logger.info("已断开 Milvus 连接")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"断开连接时出现警告: {e}")
    
    async def initialize(self) -> None:
        """
        初始化 Milvus 客户端
        建立连接并检查集合
        """
        try:
            # 连接到 Milvus
            self.connect()
            
            # 检查必要的集合是否存在
            required_collections = COLLECTION_TYPES
            existing_collections = self.collection_manager.list_collections()
            
            missing_collections = []
            for collection_type in required_collections:
                full_name = f"{self.collection_manager.collection_prefix}{collection_type}"
                if full_name not in existing_collections:
                    missing_collections.append(collection_type)
            
            if missing_collections:
                logger.warning(f"缺少以下集合: {missing_collections}")
                logger.warning("请先运行collection_manager.py创建集合")
                # 不抛出异常，允许程序继续运行，但会在查询时失败
            
            # 为现有集合设置实例
            available_collections = {}
            for collection_type in required_collections:
                try:
                    if self.collection_manager.collection_exists(collection_type):
                        collection = self.collection_manager.get_collection(collection_type)
                        available_collections[collection_type] = collection
                        
                        # 设置存储的集合实例（向后兼容）
                        self.storage.set_collection(collection_type, collection)
                except Exception as e:
                    logger.warning(f"无法加载集合 {collection_type}: {e}")
            
            self.initialized = True
            logger.info(f"Milvus客户端初始化成功，加载了 {len(available_collections)} 个集合")
            
        except Exception as e:
            logger.error(f"Milvus客户端初始化失败: {e}")
            raise
    
    async def close(self) -> None:
        """关闭连接"""
        self.disconnect()
        self.initialized = False
        logger.info("Milvus客户端已关闭")
    
    def __enter__(self):
        raise NotImplementedError("请使用异步上下文管理器")
    
    def __exit__(self, *args):
        raise NotImplementedError("请使用异步上下文管理器")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
