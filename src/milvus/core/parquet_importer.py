#!/usr/bin/env python3
"""
Parquet -> Milvus 数据导入工具
负责从磁盘读取 parquet、生成嵌入并写入集合
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd
from pymilvus import Collection
from dotenv import load_dotenv
from .collection_manager import MilvusCollectionManager
from .constants import DEFAULT_QWEN_EMBEDDING_MODEL
from .embedding_generator import QwenEmbeddingGenerator
from .schema import PARQUET_MAPPING

# 配置日志
logger = logging.getLogger(__name__)



class MilvusParquetImporter:
    """
    Parquet 数据导入器
    负责离线导入，依赖于已存在的集合
    """
    
    def __init__(
        self,
        collection_manager: MilvusCollectionManager | None = None,
        batch_size: int = 1000,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        max_text_length: int = 4000,
    ):
        """
        初始化导入器
        
        Args:
            collection_manager: Collection管理器实例，如果为None则创建新实例
            batch_size: 批量插入的大小
            embedding_model: 嵌入模型名称
            embedding_api_key: Qwen/DashScope API Key，必须由外部显式传入
            embedding_api_base: Qwen/DashScope API Base，可选
        
        注意：
        - 如果传入 None，会创建新的 MilvusCollectionManager，但不会建立连接，
          需要外部先建立连接（调用 import_data 时会自动连接）；
        - 为避免在业务代码中隐式读取环境变量，这里不再为 embedding_api_key 提供默认值。
        """
        self.collection_manager = collection_manager or MilvusCollectionManager()

        # 使用默认的 embedding 模型名称与配置（可通过参数覆盖）
        self.embedding_model_name = embedding_model or DEFAULT_QWEN_EMBEDDING_MODEL

        if embedding_api_key is None:
            raise ValueError(
                "初始化 MilvusParquetImporter 需要显式提供 embedding_api_key，"
                "请在创建时通过参数 embedding_api_key 传入 Qwen/DashScope API Key。"
            )

        self.embedding_generator = QwenEmbeddingGenerator(
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            model=self.embedding_model_name,
            name="milvus_parquet_importer",
        )
        self.max_text_length = max_text_length  # 文本最大长度限制

        logger.info("初始化Parquet导入器")
        logger.info(
            "使用 Qwen 嵌入模型: %s，维度=%s",
            self.embedding_model_name,
            self.embedding_generator.dimension,
        )
        logger.info("文本最大长度限制: %s 字符", max_text_length)
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本的embedding向量"""
        if not texts:
            return []

        clean_texts = [str(text).strip() if text else "空内容" for text in texts]
        try:
            return self.embedding_generator.embed_batch(clean_texts)
        except Exception as e:  # noqa: BLE001
            logger.error(f"生成embedding失败，返回零向量: {e}")
            return self.embedding_generator.zero_vectors(len(texts))
    
    def _split_long_text(self, text: str, max_length: int) -> List[str]:
        """智能分割长文本为多个块"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # 如果当前段落加上现有块仍在限制内
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 如果当前块不为空，保存它
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 如果段落本身就超过限制，需要进一步分割
                if len(paragraph) > max_length:
                    # 按句子分割
                    sentences = paragraph.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_length:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            # 如果单个句子太长，强制分割
                            if len(sentence) > max_length:
                                for i in range(0, len(sentence), max_length - 100):
                                    chunk_part = sentence[i:i + max_length - 100]
                                    if chunk_part.strip():
                                        chunks.append(chunk_part.strip())
                            else:
                                temp_chunk = sentence
                    
                    if temp_chunk.strip():
                        current_chunk = temp_chunk.strip()
                else:
                    current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"长文本分割为 {len(chunks)} 个块")
        return chunks
    
    def import_data(
        self,
        parquet_file: str,
        collection_type: str,
        drop_existing: bool = False,
    ) -> bool:
        """
        从 Parquet 文件导入数据到 Milvus
        
        Args:
            parquet_file: Parquet 文件路径
            collection_type: 集合类型
            drop_existing: 是否删除现有集合
        
        Returns:
            bool: 导入是否成功
        
        注意：假设 Milvus 连接已由外部建立
        """
        try:
            df = pd.read_parquet(parquet_file)
            filename = os.path.basename(parquet_file)
            logger.info(f"读取 {filename}: {df.shape}")
            
            if df.empty:
                logger.warning("文件为空，跳过")
                return True # 视为成功，因为没有数据要导入
            
            # 如果需要，删除现有集合
            # 当前 MilvusCollectionManager 尚未提供单集合删除接口，这里仅记录日志，逻辑保持向后兼容。
            if drop_existing:
                logger.warning(
                    "参数 drop_existing=True，但单集合删除尚未在 MilvusCollectionManager 中实现，跳过删除步骤"
                )
            
            # 确保集合存在
            if not self.collection_manager.collection_exists(collection_type):
                logger.error(f"集合 '{collection_type}' 不存在，请先创建集合")
                return False
            
            # 导入数据
            inserted_count = self._import_dataframe_to_collection(df, collection_type)
            return inserted_count > 0
            
        except Exception as e:
            logger.error(f"导入文件失败: {e}")
            return False

    def import_parquet_file(self, file_path: str, collection_type: str) -> int:
        """
        导入单个Parquet文件
        
        Args:
            file_path: Parquet文件路径
            collection_type: 目标集合类型
            
        Returns:
            int: 导入的记录数量
        """
        try:
            df = pd.read_parquet(file_path)
            filename = os.path.basename(file_path)
            logger.info(f"读取 {filename}: {df.shape}")
            
            if df.empty:
                logger.warning("文件为空，跳过")
                return 0
            
            # collection_type 参数是映射结果，但同时将原始文件名传递给 import_dataframe，
            # 以便保持 API 一致性（直接调用 import_dataframe 时也仅依赖文件名判断集合）
            return self.import_dataframe(df, filename, collection_type=collection_type)
            
        except Exception as e:
            logger.error(f"导入文件失败: {e}")
            return 0

    def import_dataframe(self, df: pd.DataFrame, parquet_filename: str, collection_type: str | None = None) -> int:
        """
        直接将DataFrame内容导入Milvus集合
        
        Args:
            df: 预处理后的DataFrame
            parquet_filename: 对应的parquet文件名（用于确定集合类型）
            collection_type: 可选集合类型提示（若已知可直接传入）
            
        Returns:
            int: 成功导入的记录数
        """
        if df is None or df.empty:
            logger.warning("DataFrame为空，跳过导入")
            return 0
        
        filename = os.path.basename(parquet_filename)
        resolved_collection_type = collection_type or PARQUET_MAPPING.get(filename)
        if not resolved_collection_type:
            raise ValueError(f"未找到文件 '{filename}' 对应的集合映射，请检查 PARQUET_MAPPING 配置")
        
        return self._import_dataframe_to_collection(df, resolved_collection_type)

    def _import_dataframe_to_collection(self, df: pd.DataFrame, collection_type: str) -> int:
        """执行实际的DataFrame写入逻辑"""
        if not self.collection_manager.collection_exists(collection_type):
            raise ValueError(f"集合 '{collection_type}' 不存在，请先创建集合")
        
        collection = self.collection_manager.get_collection(collection_type)
        records = self._prepare_records(df, collection_type)
        
        if not records:
            logger.warning("没有有效数据可插入")
            return 0
        
        inserted_count = self._insert_records(collection, collection_type, records)
        logger.info(f"插入 {inserted_count} 条数据到 {collection_type}")
        return inserted_count
    
    def _prepare_records(self, df: pd.DataFrame, collection_type: str) -> List[Dict[str, Any]]:
        """准备数据记录并生成embedding"""
        records = []
        texts_to_embed = []
        
        for _, row in df.iterrows():
            record = {"source_id": str(getattr(row, 'id', ''))}
            
            if collection_type == "relationship":
                if hasattr(row, 'description'):
                    desc_content = str(getattr(row, 'description', ''))
                    record.update({"description": desc_content})
                    records.append(record)
                    texts_to_embed.append(desc_content)
            
            elif collection_type == "text_unit":
                if hasattr(row, 'text'):
                    text_content = str(getattr(row, 'text', ''))
                    record.update({"text": text_content})
                    records.append(record)
                    texts_to_embed.append(text_content)
            
            elif collection_type == "entity_title":
                if hasattr(row, 'title'):
                    title_content = str(getattr(row, 'title', ''))
                    record.update({"title": title_content})
                    records.append(record)
                    texts_to_embed.append(title_content)
            
            elif collection_type == "entity_description":
                if hasattr(row, 'title'):
                    title_content = str(getattr(row, 'title', ''))
                    # 检查是否有description字段（entities.parquet）或summary字段（community_reports.parquet）
                    desc_content = ""
                    if hasattr(row, 'description'):
                        desc_content = str(getattr(row, 'description', ''))
                    elif hasattr(row, 'summary'):
                        desc_content = str(getattr(row, 'summary', ''))
                    
                    combined_content = f"{title_content}:{desc_content}"
                    record.update({
                        "title": title_content,
                        "description": desc_content,
                        "title_description": combined_content
                    })
                    records.append(record)
                    texts_to_embed.append(combined_content)
            
            elif collection_type == "community_title":
                if hasattr(row, 'title'):
                    title_content = str(getattr(row, 'title', ''))
                    record.update({"title": title_content})
                    records.append(record)
                    texts_to_embed.append(title_content)
            
            elif collection_type == "community_summary":
                if hasattr(row, 'summary'):
                    summary_content = str(getattr(row, 'summary', ''))
                    record.update({"summary": summary_content})
                    records.append(record)
                    texts_to_embed.append(summary_content)
            
            elif collection_type == "community_full_content":
                if hasattr(row, 'full_content'):
                    full_content = str(getattr(row, 'full_content', ''))
                    record.update({"full_content": full_content})
                    records.append(record)
                    texts_to_embed.append(full_content)
        
        # 批量生成embedding
        if records and texts_to_embed:
            logger.info(f"为{len(texts_to_embed)}条记录生成embedding向量...")
            embeddings = self._generate_embeddings(texts_to_embed)
            
            # 将embedding添加到记录中
            for i, record in enumerate(records):
                if i < len(embeddings):
                    record["embedding"] = embeddings[i]
                else:
                    record["embedding"] = self.embedding_generator.zero_vector()
        
        return records
    
    def _insert_records(self, collection: Collection, collection_type: str, records: List[Dict[str, Any]]) -> int:
        """插入记录到集合"""
        field_mapping = {
            "relationship": ["source_id", "description", "embedding"],
            "text_unit": ["source_id", "text", "embedding"],
            "entity_title": ["source_id", "title", "embedding"],
            "entity_description": ["source_id", "title", "description", "title_description", "embedding"],
            "community_title": ["source_id", "title", "embedding"],
            "community_summary": ["source_id", "summary", "embedding"],
            "community_full_content": ["source_id", "full_content", "embedding"]
        }
        
        field_names = field_mapping[collection_type]
        insert_data = []
        
        for field in field_names:
            field_values = [item.get(field) for item in records]
            insert_data.append(field_values)
        
        try:
            mr = collection.insert(insert_data)
            collection.flush()
            return len(records)
            
        except Exception as e:
            logger.error(f"插入失败: {e}")
            return 0
    
    def import_directory(self, directory_path: str) -> Dict[str, int]:
        """
        导入整个目录的所有parquet文件
        
        Args:
            directory_path: 包含parquet文件的目录路径
            
        Returns:
            Dict[str, int]: 文件名到导入记录数的映射
        """
        results = {}
        
        # 确保连接到Milvus（连接应由上层 MilvusClient 或调用方统一管理）
        if not self.collection_manager._connected:
            raise RuntimeError(
                "MilvusCollectionManager 尚未连接，请先通过 MilvusClient.connect() "
                "或手动调用 collection_manager.connect(host, port) 建立连接。"
            )
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.parquet'):
                file_path = os.path.join(directory_path, filename)
                collection_type = PARQUET_MAPPING.get(filename)
                
                if collection_type:
                    count = self.import_parquet_file(file_path, collection_type)
                    results[filename] = count
                else:
                    logger.warning(f"跳过文件 {filename} (无映射配置)")
        
        return results


def main():
    """主函数 - 用于测试导入功能（通过统一的 MilvusClient）"""
    from milvus import MilvusClient

    # 读取环境变量配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    load_dotenv(os.path.join(project_root, ".env"))

    # 基于环境变量创建客户端
    client = MilvusClient.from_env()
    print(f"ℹ️ 使用的Milvus配置: {client.host}:{client.port}")

    # 复用客户端中的 CollectionManager
    collection_manager = client.collection_manager

    try:
        client.connect()

        # 检查集合是否存在
        existing_collections = collection_manager.list_collections()
        if not existing_collections:
            print("❌ 未找到任何集合，请先运行创建集合脚本")
            return False

        print(f"✅ 找到 {len(existing_collections)} 个现有集合")

        # 创建数据导入器
        importer = MilvusParquetImporter(collection_manager)

        # 导入数据
        parquet_dir = os.path.join(current_dir, "../tests/parquet")
        if not os.path.exists(parquet_dir):
            print(f"❌ Parquet目录不存在: {parquet_dir}")
            return False

        print(f"📂 从目录导入数据: {parquet_dir}")
        results = importer.import_directory(parquet_dir)

        # 输出结果
        print("\n📊 导入结果:")
        print("=" * 50)
        total = 0
        for file, count in results.items():
            print(f"{file}: {count} 条")
            total += count
        print(f"\n总计: {total} 条数据")

        return True

    except Exception as e:
        logger.error(f"导入过程出错: {e}")
        print(f"❌ 导入失败: {e}")
        return False

    finally:
        client.disconnect()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
