#!/usr/bin/env python3
"""
清空Milvus向量数据库的所有集合和schema
用于重新初始化向量数据库
"""

import os
import logging
import argparse
from typing import List

from dotenv import load_dotenv
from milvus import MilvusClient
from milvus.core.collection_manager import COLLECTION_NAMES as NAMES_DICT

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取所有集合的全名列表
COLLECTION_NAMES = list(NAMES_DICT.values())

class MilvusReset:
    """Milvus数据库重置工具（只依赖外部传入的 MilvusClient，不再自行读取配置或管理连接）"""
    
    def __init__(self, client: MilvusClient):
        """
        初始化重置工具
        
        Args:
            client: 已由外部构造好的 MilvusClient 实例（建议由调用方负责连接与关闭）
        """
        self.client: MilvusClient = client
        logger.info("初始化重置工具，使用外部提供的 MilvusClient 实例")
    
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        if not self.client:
            logger.error("尚未提供 MilvusClient，无法列出集合")
            return []
        try:
            collections = self.client.collection_manager.list_collections()
            logger.info(f"当前存在的GraphRAG集合: {collections}")
            return collections
        except Exception as e:  # noqa: BLE001
            logger.error(f"获取集合列表失败: {e}")
            return []
    
    def drop_collection(self, collection_name: str) -> bool:
        """删除指定集合"""
        # 为了保持接口兼容，这里依然按名称删除，但具体删除逻辑交给 CollectionManager 统一处理
        if not self.client:
            logger.error("尚未连接 Milvus，无法删除集合")
            return False
        try:
            # CollectionManager.drop_collections 会统一删除 GraphRAG 相关集合，
            # 这里仅作为单集合删除时的日志辅助。
            logger.info(f"请求删除集合: {collection_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"删除集合 {collection_name} 失败: {e}")
            return False
    
    def drop_all_collections(self) -> int:
        """删除所有GraphRAG相关的集合（通过 CollectionManager 统一处理）"""
        if not self.client:
            logger.error("尚未提供 MilvusClient，无法删除集合")
            return 0
        
        try:
            dropped_count = self.client.collection_manager.drop_collections()
            logger.info(f"通过 CollectionManager 删除 GraphRAG 集合数量: {dropped_count}")
            return dropped_count
        except Exception as e:  # noqa: BLE001
            logger.error(f"删除 GraphRAG 集合失败: {e}")
            return 0
    
    def reset_database(self) -> None:
        """重置整个数据库（删除所有集合）"""
        if not self.client:
            raise RuntimeError("MilvusReset 需要传入有效的 MilvusClient 实例")
        # 这里不再负责调用 client.connect()/disconnect()，由外部脚本统一管理连接生命周期
        try:
            logger.info("开始重置Milvus数据库...")
            
            # 删除所有GraphRAG集合
            dropped_count = self.drop_all_collections()
            
            logger.info(f"重置完成，共删除 {dropped_count} 个集合")
            
            # 列出剩余的集合
            remaining_collections = self.list_collections()
            if remaining_collections:
                logger.info(f"剩余集合: {remaining_collections}")
            else:
                logger.info("数据库已清空，无任何集合")
                
        except Exception as e:
            logger.error(f"重置数据库过程中发生错误: {e}")
            raise

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="重置Milvus向量数据库")
    parser.add_argument(
        "--milvus-host",
        type=str,
        default=None,
        help="Milvus服务器地址，格式: host:port（例如: 192.168.3.101:19530），如果未指定则从配置文件读取"
    )
    args = parser.parse_args()
    
    # 读取环境变量配置（使用 python-dotenv）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    load_dotenv(os.path.join(project_root, ".env"))

    # 先基于环境变量构造 MilvusClient（集中通过 MilvusClient.from_env 处理）
    client = MilvusClient.from_env()

    # 如果提供了命令行参数，则覆盖 host/port（其余配置仍由 from_env 决定）
    if args.milvus_host:
        host_port = args.milvus_host.split(":")
        if len(host_port) != 2:
            print(f"❌ 错误的Milvus地址格式: {args.milvus_host}，应为 host:port")
            return
        client.host = host_port[0]
        client.port = int(host_port[1])

    print(f"ℹ️ 使用的Milvus配置: {client.host}:{client.port}")

    # 创建重置工具（只依赖外部传入的 client）
    reset_tool = MilvusReset(client=client)
    
    # 确认操作
    print("⚠️  警告：此操作将删除所有GraphRAG相关的Milvus集合！")
    print("集合列表:")
    for name in COLLECTION_NAMES:
        print(f"  - {name}")
    
    confirmation = input("\n确定要继续吗？(y/N): ")
    
    if confirmation.lower() not in ['y', 'yes']:
        print("操作已取消")
        return
    
    try:
        # 建立连接并执行重置
        client.connect()
        reset_tool.reset_database()
        print("✅ 数据库重置完成")
        
    except Exception as e:
        logger.error(f"重置失败: {e}")
        print("❌ 数据库重置失败")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()