"""
Milvus配置模块，从环境变量读取配置
"""

from typing import Optional

from environs import Env

from .constants import DEFAULT_QWEN_EMBEDDING_MODEL


def _get_str_from_env(env: Env, keys: list[str], default: str) -> str:
    """
    按顺序从多个环境变量名中读取字符串，若都不存在则返回默认值。
    
    等价于原先 graphrag 的 EnvironmentReader.str("a", "A", "default") 语义：
    - 依次尝试 "a"、"A"
    - 只要有一个存在（即使是空字符串也视为存在），就返回它
    - 都不存在时返回 default
    """
    for key in keys:
        value = env.str(key, default=None)
        if value is not None:
            return value
    return default


def _get_int_from_env(env: Env, keys: list[str], default: int) -> int:
    """按顺序从多个环境变量名中读取整数，不存在则返回默认值。"""
    for key in keys:
        value = env.int(key, default=None)
        if value is not None:
            return value
    return default


class MilvusConfig:
    """Milvus数据库配置类"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_prefix: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self.embedding_model = embedding_model

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        """从环境变量创建配置实例（不依赖 graphrag 的 EnvironmentReader）。"""
        env = Env()
        # 默认会从当前环境读取；上层脚本已通过 python-dotenv 预先加载 .env

        return cls(
            host=_get_str_from_env(env, ["milvus_host", "MILVUS_HOST"], "localhost"),
            port=_get_int_from_env(env, ["milvus_port", "MILVUS_PORT"], 19530),
            collection_prefix=_get_str_from_env(
                env, ["milvus_collection_prefix", "MILVUS_COLLECTION_PREFIX"], "graphrag_"
            ),
            embedding_model=_get_str_from_env(
                env,
                ["milvus_embedding_model", "MILVUS_EMBEDDING_MODEL"],
                DEFAULT_QWEN_EMBEDDING_MODEL,
            ),
        )

    def get_grpc_address(self) -> str:
        """获取gRPC地址"""
        return f"{self.host}:{self.port}"

    def __repr__(self) -> str:
        return (
            f"MilvusConfig(host={self.host}, port={self.port}, "
            f"collection_prefix={self.collection_prefix})"
        )


def get_milvus_config() -> MilvusConfig:
    """获取Milvus配置实例"""
    return MilvusConfig.from_env()