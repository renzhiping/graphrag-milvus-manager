#!/usr/bin/env python3
"""
创建Milvus向量数据库的所有collection
这是一个独立的脚本，专门负责collection的创建和初始化
"""

import os
import sys
import logging
import argparse

from dotenv import load_dotenv
from milvus import MilvusClient

# 添加当前目录到Python路径，确保可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_collections(client: MilvusClient | None = None) -> bool:
    """
    创建所有需要的Milvus集合
    
    Args:
        client: 已初始化的 MilvusClient 实例；如果为 None，则基于环境变量创建
    """
    try:
        # 使用统一的 MilvusClient 管理连接和集合
        if client is None:
            client = MilvusClient.from_env()

        client.connect()
        manager = client.collection_manager

        logger.info("开始创建Milvus集合...")
        collections = manager.create_collections()

        if collections:
            logger.info("✅ 集合创建完成")
            for name in collections.keys():
                full_name = f"{manager.collection_prefix}{name}"
                logger.info("   - %s", full_name)
            return True
        else:
            logger.error("❌ 未创建任何集合")
            return False
        
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.error("请确保collection_manager.py文件存在")
        return False
    except Exception as e:
        logger.error(f"创建集合过程中发生错误: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="创建Milvus向量数据库集合")
    parser.add_argument(
        "--milvus-host",
        type=str,
        default=None,
        help="Milvus服务器地址，格式: host:port（例如: 192.168.3.101:19530），如果未指定则从配置文件读取"
    )
    args = parser.parse_args()
    
    # 读取环境变量配置（使用 python-dotenv）
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    load_dotenv(os.path.join(project_root, ".env"))
    
    # 获取/构造 MilvusClient
    client: MilvusClient | None = None
    if args.milvus_host:
        # 如果提供了命令行参数，使用参数中的地址
        host_port = args.milvus_host.split(":")
        if len(host_port) != 2:
            print(f"❌ 错误的Milvus地址格式: {args.milvus_host}，应为 host:port")
            return False

        host, port_str = host_port[0], host_port[1]
        client = MilvusClient(host=host, port=port_str)
        print(f"ℹ️ 使用命令行指定的Milvus配置: {host}:{port_str}")
    else:
        # 基于环境变量创建客户端
        client = MilvusClient.from_env()
        print(f"ℹ️ 使用配置文件中的Milvus配置: {client.host}:{client.port}")

    print("📦 准备创建Milvus集合...")
    print("   这将创建以下集合:")
    print("   - graphrag_relationship (关系描述)")
    print("   - graphrag_text_unit (文本单元)")
    print("   - graphrag_entity_title (实体标题)")
    print("   - graphrag_entity_description (实体描述)")
    print("   - graphrag_community_title (社区标题)")
    print("   - graphrag_community_summary (社区摘要)")
    print("   - graphrag_community_full_content (社区完整内容)")
    
    confirmation = input("\n确定要创建这些集合吗？(y/N): ")
    if confirmation.lower() not in ['y', 'yes']:
        print("操作已取消")
        return False
    
    print("🚀 开始创建集合...")
    
    success = create_collections(client=client)
    
    if success:
        print("✅ 所有集合创建成功！")
        print("💡 现在可以运行数据导入脚本：")
        print("   python -m milvus.milvus_import_all")
    else:
        print("❌ 集合创建失败")
        
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
