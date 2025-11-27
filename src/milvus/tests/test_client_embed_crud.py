"""
针对 MilvusClient 新增的嵌入及单条记录增删方法的单元测试。

这些测试不依赖真实的 Milvus 或 Qwen 服务，通过 monkeypatch 注入假的
QwenEmbeddingGenerator 和存储实现，只验证调用链和数据拼装是否正确。
"""

from __future__ import annotations

from typing import Any, List

import pytest

from milvus.query.client import MilvusClient


class DummyEmbeddingGenerator:
    """用于替代真实 QwenEmbeddingGenerator 的测试专用实现。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ANN401
        # 维度仅用于日志，不影响本测试逻辑
        self.dimension = 4
        self.last_text: str | None = None
        self.last_texts: List[str] | None = None

    def embed(self, text: str) -> List[float]:
        self.last_text = text
        # 返回一个固定向量，便于断言
        return [1.0, 2.0, 3.0, 4.0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.last_texts = texts
        return [[1.0, 2.0, 3.0, 4.0] for _ in texts]


@pytest.fixture(autouse=True)
def patch_embedding_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    将 QwenEmbeddingGenerator 替换为 DummyEmbeddingGenerator，避免真实网络调用。
    """
    import milvus.core.embedding_generator as eg_mod
    import milvus.query.query_manager as qm_mod

    # 替换核心模块和已经引用该类的 query_manager 中的 QwenEmbeddingGenerator
    monkeypatch.setattr(eg_mod, "QwenEmbeddingGenerator", DummyEmbeddingGenerator)
    monkeypatch.setattr(qm_mod, "QwenEmbeddingGenerator", DummyEmbeddingGenerator)


def _build_client(monkeypatch: pytest.MonkeyPatch) -> MilvusClient:
    """
    构造一个带假存储的 MilvusClient，并强制标记为已连接。
    """
    # 构造客户端（此时会使用 DummyEmbeddingGenerator）
    client = MilvusClient(
        host="localhost",
        port="19530",
        embedding_api_key="dummy-key",
    )
    # 测试中不真正连接 Milvus，仅标记为已连接
    client._connected = True  # noqa: SLF001

    # 注入假的存储实现，拦截底层 CRUD 调用
    class DummyStore:
        def __init__(self) -> None:
            self.last_delete_args: tuple[Any, ...] | None = None
            self.last_insert_args: tuple[Any, ...] | None = None

        def delete_by_field(self, collection_type: str, field: str, value: Any) -> int:
            self.last_delete_args = (collection_type, field, value)
            return 3

        def insert_single_record(self, collection_type: str, record: dict[str, Any]) -> int:
            self.last_insert_args = (collection_type, record)
            return 1

    dummy_store = DummyStore()
    client.storage = dummy_store  # type: ignore[assignment]

    return client


def test_delete_record_delegates_to_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_record 应该委托给 storage.delete_by_field，并返回删除数量。"""
    client = _build_client(monkeypatch)
    # 类型检查：我们知道这里是 DummyStore
    dummy_store = client.storage  # type: ignore[assignment]

    deleted = client.delete_record(
        collection_type="text_unit",
        where_field="source_id",
        where_value="doc_1",
    )

    assert deleted == 3
    assert getattr(dummy_store, "last_delete_args", None) == (
        "text_unit",
        "source_id",
        "doc_1",
    )


@pytest.mark.parametrize(
    "collection_type,data,expected_text_field,expected_text",
    [
        ("text_unit", {"source_id": "doc_1", "text": "Hello"}, "text", "Hello"),
        ("relationship", {"source_id": "rel_1", "description": "desc"}, "description", "desc"),
        ("entity_title", {"source_id": "ent_1", "title": "Title"}, "title", "Title"),
        (
            "entity_description",
            {"source_id": "ent_2", "title": "T", "description": "D"},
            "title_description",
            "T:D",
        ),
        ("community_summary", {"source_id": "c1", "summary": "Sum"}, "summary", "Sum"),
    ],
)
def test_add_embedding_record_builds_record_and_calls_storage(
    monkeypatch: pytest.MonkeyPatch,
    collection_type: str,
    data: dict[str, Any],
    expected_text_field: str,
    expected_text: str,
) -> None:
    """
    add_embedding_record:
    - 使用正确的文本字段生成 embedding；
    - 将 embedding 与业务字段一起传给 storage.insert_single_record。
    """
    client = _build_client(monkeypatch)
    dummy_store = client.storage  # type: ignore[assignment]

    inserted = client.add_embedding_record(collection_type=collection_type, data=data)

    assert inserted == 1
    # 断言存储层收到的参数
    assert getattr(dummy_store, "last_insert_args", None) is not None
    stored_collection_type, record = dummy_store.last_insert_args  # type: ignore[misc]
    assert stored_collection_type == collection_type
    assert record["source_id"] == data["source_id"]
    assert record.get(expected_text_field) is not None
    # 确保 embedding 已经写入
    assert "embedding" in record
    assert isinstance(record["embedding"], list) and len(record["embedding"]) == 4

    # 断言嵌入生成器收到的文本内容正确
    eg = client.query_manager.embedding_generator  # type: ignore[attr-defined]
    assert getattr(eg, "last_text", None) == (expected_text or "空内容")


