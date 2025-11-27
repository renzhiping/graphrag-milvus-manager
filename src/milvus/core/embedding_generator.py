"""
统一的 Qwen 嵌入生成器，负责与 DashScope 兼容接口交互.
"""

from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from .constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_QWEN_API_BASE,
    DEFAULT_QWEN_EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


class QwenEmbeddingGenerator:
    """
    面向 Milvus 导入/查询的 Qwen 嵌入生成工具.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str | None = None,
        model: str | None = None,
        name: str = "milvus_embedding_generator",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_retries: int = 20,
    ) -> None:
        """
        初始化 Qwen 嵌入生成器.

        注意：
        - api_key 必须由调用方显式传入，本模块不再从环境变量中解析默认值；
        - 这样可以确保在作为库被引用时，所有敏感配置均由外部统一管理。
        """
        if not api_key:
            raise ValueError("api_key 不能为空，请由调用方显式传入 Qwen/DashScope API Key")

        self.api_key = api_key
        # api_base / model 为非敏感配置，这里仍然提供合理默认值
        self.api_base = api_base or DEFAULT_QWEN_API_BASE
        self.model_name = model or DEFAULT_QWEN_EMBEDDING_MODEL
        self.batch_size = batch_size
        self.dimension = EMBEDDING_DIM
        self.max_retries = max_retries

        logger.info(
            "初始化 Qwen 嵌入模型: model=%s, api_base=%s, batch_size=%s",
            self.model_name,
            self.api_base,
            self.batch_size,
        )

        # 使用 OpenAI 官方客户端，以 Qwen 的 OpenAI-Compatible 接口调用嵌入模型
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            max_retries=self.max_retries,
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        以批次方式生成嵌入，自动分块并处理失败回退.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            try:
                # Qwen text-embedding-v3 默认返回1024维向量
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=chunk,
                )
                # OpenAI Embeddings API 返回一个 data 列表，每个元素包含 embedding 向量
                chunk_embeddings = [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001
                logger.error("批量生成嵌入失败，使用零向量回退: %s", exc)
                chunk_embeddings = self.zero_vectors(len(chunk))
            embeddings.extend(chunk_embeddings)

        return embeddings

    def embed(self, text: str) -> List[float]:
        """
        生成单条文本嵌入.
        """
        batch = self.embed_batch([text])
        if batch:
            return batch[0]
        return self.zero_vector()

    def zero_vector(self) -> List[float]:
        """
        返回单个零向量.
        """
        return [0.0] * self.dimension

    def zero_vectors(self, count: int) -> List[List[float]]:
        """
        返回指定数量的零向量.
        """
        return [[0.0] * self.dimension for _ in range(count)]

