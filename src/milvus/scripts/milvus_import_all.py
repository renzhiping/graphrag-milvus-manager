#!/usr/bin/env python3
"""
导入所有parquet文件数据到Milvus向量数据库
重构后直接使用 MilvusParquetImporter，简化导入流程
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from milvus import MilvusClient
from milvus.core.parquet_importer import MilvusParquetImporter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_parquet_data(parquet_dir: str, client: MilvusClient) -> bool:
    """
    直接导入parquet数据到Milvus
    
    Args:
        parquet_dir: parquet文件目录
        collection_manager: 已初始化的Collection管理器
        
    Returns:
        bool: 导入是否成功
    """
    try:
        # 检查parquet文件
        parquet_path = Path(parquet_dir)
        if not parquet_path.exists():
            logger.error("Parquet目录不存在: %s", parquet_dir)
            return False
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        if not parquet_files:
            logger.error("在目录 %s 中未找到任何.parquet文件", parquet_dir)
            return False
        
        logger.info("找到 %d 个parquet文件:", len(parquet_files))
        for file in parquet_files:
            logger.info("  - %s", file.name)
        
        # 创建数据导入器，复用同一个客户端中的 CollectionManager
        importer = MilvusParquetImporter(client.collection_manager)
        
        # 导入数据
        logger.info("开始导入parquet数据到Milvus...")
        results = importer.import_directory(parquet_dir)
        
        # 输出结果
        logger.info("导入结果:")
        total_imported = 0
        for filename, count in results.items():
            logger.info("  %s: %d 条记录", filename, count)
            total_imported += count
        
        logger.info("总计导入: %d 条记录", total_imported)
        
        if total_imported > 0:
            logger.info("✅ 数据导入成功")
            return True
        else:
            logger.warning("⚠️  没有导入任何数据")
            return False
        
    except Exception as e:
        logger.exception("导入过程中发生错误: %s", e)
        return False

def main(parquet_dir: Optional[str] = None) -> bool:
    """
    主函数
    
    Args:
        parquet_dir: parquet文件目录（必填），不再自动检索
        
    Returns:
        bool: 导入是否成功
    """
    # 读取环境变量配置（使用 python-dotenv）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    load_dotenv(os.path.join(project_root, ".env"))
    
    # 获取并打印Milvus配置
    from milvus.core.config import get_milvus_config
    milvus_config = get_milvus_config()
    print(f"ℹ️ 使用的Milvus配置: {milvus_config.host}:{milvus_config.port}")

    # 校验参数：不再自动查找目录，必须显式传入
    if parquet_dir is None:
        print("❌ 必须显式指定 parquet 目录路径，例如：")
        print("   uv run python src/milvus/scripts/milvus_import_all.py /path/to/parquet_dir")
        return False
    
    print(f"📂 使用parquet目录: {parquet_dir}")
    
    # 创建并连接统一的 MilvusClient
    client: Optional[MilvusClient] = None
    try:
        print("🔗 连接到Milvus...")
        client = MilvusClient(
            host=milvus_config.host or "localhost",
            port=str(milvus_config.port or 19530),
        )
        client.connect()
        
        # 检查必要的集合是否存在
        existing_collections = client.collection_manager.list_collections()
        if not existing_collections:
            print("❌ 未找到任何集合")
            print("💡 请先运行以下命令创建集合:")
            print("   python -m milvus.milvus_create_collections")
            return False
        
        print(f"✅ 找到 {len(existing_collections)} 个现有集合")
        for collection_name in existing_collections:
            print(f"   - {collection_name}")
        
        # 导入数据
        print("\n🚀 开始导入数据...")
        success = import_parquet_data(parquet_dir, client)
        
        if success:
            print("\n✅ 数据导入完成！")
        else:
            print("\n❌ 数据导入失败")
            
        return success
        
    except Exception as e:
        logger.exception("导入过程中发生错误: %s", e)
        print(f"❌ 导入失败: {e}")
        return False
        
    finally:
        if client:
            client.disconnect()

if __name__ == "__main__":
    # 简单的命令行参数解析：只接收一个必填的 parquet 目录路径
    import argparse

    parser = argparse.ArgumentParser(description="导入指定目录下的所有 parquet 文件到 Milvus")
    parser.add_argument(
        "parquet_dir",
        type=str,
        help="包含 .parquet 文件的目录路径",
    )
    args = parser.parse_args()

    success = main(args.parquet_dir)
    sys.exit(0 if success else 1)
