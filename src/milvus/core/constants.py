"""
Milvus 向量存储模块的公共常量.
"""

DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 10  # Qwen API 限制批量大小不超过10
EMBEDDING_DIM = 1024  # Qwen text-embedding-v3 最大支持1024维

