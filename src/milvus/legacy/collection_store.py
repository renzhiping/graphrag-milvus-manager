"""
Milvus 集合存储模块
负责将已生成的嵌入数据写入/更新/删除到指定集合
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from pymilvus import Collection
import logging

logger = logging.getLogger(__name__)


class MilvusCollectionStore:
    """Milvus 集合级存储包装，封装常用 CRUD 操作"""
    
    def __init__(self):
        self.collections = {}
    
    def set_collection(self, collection_type: str, collection: Collection) -> None:
        """设置集合实例"""
        self.collections[collection_type] = collection
    
    def get_collection(self, collection_type: str) -> Collection:
        """获取集合实例"""
        if collection_type not in self.collections:
            raise ValueError(f"集合类型 '{collection_type}' 未初始化")
        return self.collections[collection_type]
    
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """插入文档数据"""
        return self._insert_data("document", documents, ["source_id", "text", "embedding"])
    
    async def insert_relationships(self, relationships: List[Dict[str, Any]]) -> List[int]:
        """插入关系数据"""
        return self._insert_data("relationship", relationships, ["source_id", "description", "embedding"])
    
    async def insert_text_units(self, text_units: List[Dict[str, Any]]) -> List[int]:
        """插入文本单元数据"""
        return self._insert_data("text_unit", text_units, ["source_id", "text", "embedding"])
    
    async def insert_entity_titles(self, entities: List[Dict[str, Any]]) -> List[int]:
        """插入实体标题数据"""
        return self._insert_data("entity_title", entities, ["source_id", "title", "embedding"])
    
    async def insert_entity_descriptions(self, entities: List[Dict[str, Any]]) -> List[int]:
        """插入实体描述数据"""
        # 预处理数据，组合title和description
        processed_data = []
        for entity in entities:
            processed = entity.copy()
            processed["title_description"] = f"{entity.get('title', '')}:{entity.get('description', '')}"
            processed_data.append(processed)
        
        return self._insert_data("entity_description", processed_data, 
                               ["source_id", "title", "description", "title_description", "embedding"])
    
    async def insert_community_titles(self, communities: List[Dict[str, Any]]) -> List[int]:
        """插入社区标题数据"""
        return self._insert_data("community_title", communities, ["source_id", "title", "embedding"])
    
    async def insert_community_summaries(self, communities: List[Dict[str, Any]]) -> List[int]:
        """插入社区摘要数据"""
        return self._insert_data("community_summary", communities, ["source_id", "summary", "embedding"])
    
    async def insert_community_full_contents(self, communities: List[Dict[str, Any]]) -> List[int]:
        """插入社区完整内容数据"""
        return self._insert_data("community_full_content", communities, ["source_id", "full_content", "embedding"])
    
    def _insert_data(self, collection_type: str, data: List[Dict[str, Any]], 
                    field_names: List[str]) -> List[int]:
        """通用数据插入方法"""
        if not data:
            return []
        
        collection = self.get_collection(collection_type)
        
        # 准备插入数据
        insert_data = []
        for field in field_names:
            field_values = [item.get(field) for item in data]
            insert_data.append(field_values)
        
        # 插入数据
        try:
            mr = collection.insert(insert_data)
            collection.flush()
            
            logger.info(f"向 '{collection_type}' 集合插入 {len(data)} 条数据成功")
            return mr.primary_keys
            
        except Exception as e:
            logger.error(f"插入数据到 '{collection_type}' 失败: {e}")
            raise
    
    async def batch_insert_from_dataframe(self, collection_type: str, 
                                        df: pd.DataFrame, 
                                        embedding_field: str = "embedding") -> List[int]:
        """
        从DataFrame批量插入数据
        
        Args:
            collection_type: 集合类型
            df: 包含数据的DataFrame
            embedding_field: 嵌入向量字段名
        """
        if df is None or df.empty:
            return []
        
        collection = self.get_collection(collection_type)
        
        # 根据集合类型准备数据
        if collection_type == "document":
            data = df[["id", "text", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "text", "embedding"]
            
        elif collection_type == "relationship":
            data = df[["id", "description", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "description", "embedding"]
            
        elif collection_type == "text_unit":
            data = df[["id", "text", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "text", "embedding"]
            
        elif collection_type == "entity_title":
            data = df[["id", "title", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "title", "embedding"]
            
        elif collection_type == "entity_description":
            # 需要组合title和description
            data = df.copy()
            data["source_id"] = data["id"]
            data["title_description"] = data["title"] + ":" + data["description"]
            field_names = ["source_id", "title", "description", "title_description", "embedding"]
            
        elif collection_type == "community_title":
            data = df[["id", "title", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "title", "embedding"]
            
        elif collection_type == "community_summary":
            data = df[["id", "summary", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "summary", "embedding"]
            
        elif collection_type == "community_full_content":
            data = df[["id", "full_content", embedding_field]].rename(columns={"id": "source_id"})
            field_names = ["source_id", "full_content", "embedding"]
            
        else:
            raise ValueError(f"不支持的集合类型: {collection_type}")
        
        # 转换为字典列表格式
        records = data.to_dict('records')
        return self._insert_data(collection_type, records, field_names)
    
    async def delete_by_source_ids(self, collection_type: str, source_ids: List[str]) -> int:
        """根据源ID删除数据"""
        if not source_ids:
            return 0
        
        collection = self.get_collection(collection_type)
        
        # 构建删除表达式
        ids_str = ", ".join([f"'{id}'" for id in source_ids])
        expr = f"source_id in [{ids_str}]"
        
        try:
            result = collection.delete(expr)
            collection.flush()
            
            logger.info(f"从 '{collection_type}' 集合删除 {len(source_ids)} 条数据")
            return result.delete_count
            
        except Exception as e:
            logger.error(f"删除数据失败: {e}")
            raise
    
    async def clear_collection(self, collection_type: str) -> None:
        """清空集合数据"""
        collection = self.get_collection(collection_type)
        
        # 查询所有数据并删除
        try:
            results = collection.query(expr="", output_fields=["source_id"])
            source_ids = [str(item["source_id"]) for item in results]
            
            if source_ids:
                await self.delete_by_source_ids(collection_type, source_ids)
                logger.info(f"清空 '{collection_type}' 集合完成")
            else:
                logger.info(f"'{collection_type}' 集合已为空")
                
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            raise

    # ===== 通用同步 CRUD 方法（供 MilvusClient 等高层封装调用） =====

    def delete_by_field(self, collection_type: str, field: str, value: Any) -> int:
        """
        根据任意字段删除记录（同步方法）。

        Args:
            collection_type: 集合类型，例如 "text_unit"、"entity_title" 等
            field: 字段名，例如 "source_id"、"title"
            value: 字段值

        Returns:
            int: 删除的记录数量
        """
        collection = self.get_collection(collection_type)

        if isinstance(value, str):
            expr = f"{field} == '{value}'"
        else:
            expr = f"{field} == {value}"

        try:
            result = collection.delete(expr)
            collection.flush()
            delete_count = getattr(result, "delete_count", 0)
            logger.info(
                "从 '%s' 集合删除记录: field=%s, value=%r, count=%s",
                collection_type,
                field,
                value,
                delete_count,
            )
            return delete_count
        except Exception as e:  # noqa: BLE001
            logger.error("根据字段删除记录失败: %s", e)
            raise

    def insert_single_record(self, collection_type: str, record: Dict[str, Any]) -> int:
        """
        向指定集合插入一条记录（同步方法），record 中应包含所有业务字段和 embedding。

        字段映射规则与 Parquet 导入器保持一致：
        - relationship: ["source_id", "description", "embedding"]
        - text_unit/document: ["source_id", "text", "embedding"]
        - entity_title/community_title: ["source_id", "title", "embedding"]
        - entity_description: ["source_id", "title", "description", "title_description", "embedding"]
        - community_summary: ["source_id", "summary", "embedding"]
        - community_full_content: ["source_id", "full_content", "embedding"]
        """
        field_mapping: Dict[str, List[str]] = {
            "relationship": ["source_id", "description", "embedding"],
            "text_unit": ["source_id", "text", "embedding"],
            "document": ["source_id", "text", "embedding"],
            "entity_title": ["source_id", "title", "embedding"],
            "entity_description": ["source_id", "title", "description", "title_description", "embedding"],
            "community_title": ["source_id", "title", "embedding"],
            "community_summary": ["source_id", "summary", "embedding"],
            "community_full_content": ["source_id", "full_content", "embedding"],
        }

        if collection_type not in field_mapping:
            raise ValueError(f"不支持的集合类型: {collection_type}")

        collection = self.get_collection(collection_type)
        fields = field_mapping[collection_type]

        insert_data: List[List[Any]] = []
        for field in fields:
            insert_data.append([record.get(field)])

        try:
            collection.insert(insert_data)
            collection.flush()
            logger.info(
                "向 '%s' 集合插入 1 条记录，source_id=%s",
                collection_type,
                record.get("source_id"),
            )
            return 1
        except Exception as e:  # noqa: BLE001
            logger.error("插入单条记录失败: %s", e)
            raise
