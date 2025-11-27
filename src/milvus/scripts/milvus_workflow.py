#!/usr/bin/env python3
"""
Milvus向量数据库工作流程管理脚本
提供完整的操作流程指导，确保正确的操作顺序
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# 添加当前目录到Python路径，确保可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from milvus import MilvusClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_menu():
    """显示操作菜单"""
    print("\n" + "=" * 60)
    print("🚀 GraphRAG Milvus 向量数据库管理工具")
    print("=" * 60)
    print("重构后的工作流程：")
    print("1. 创建集合 (Collection) - 只需执行一次")
    print("2. 导入数据 - 可多次执行")
    print("3. 重置数据库 - 删除所有集合和数据")
    print("4. 查看集合状态")
    print("5. 测试查询功能")
    print("0. 退出")
    print("=" * 60)


def create_collections(client: MilvusClient) -> bool:
    """创建集合"""
    print("\n📦 步骤1: 创建Milvus集合")
    print("-" * 40)
    
    try:
        manager = client.collection_manager
        
        # 检查是否已有集合
        existing = manager.list_collections()
        if existing:
            print(f"⚠️  发现已存在的集合: {len(existing)} 个")
            for collection in existing:
                print(f"   - {collection}")
            
            choice = input("\n是否要重新创建所有集合？(y/N): ")
            if choice.lower() in ['y', 'yes']:
                dropped = manager.drop_collections()
                print(f"🗑️  已删除 {dropped} 个集合")
            else:
                print("保持现有集合不变")
                return True
        
        # 创建集合
        collections = manager.create_collections()
        print(f"✅ 成功创建 {len(collections)} 个集合:")
        for name, collection in collections.items():
            full_name = f"{manager.collection_prefix}{name}"
            print(f"   - {full_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        print(f"❌ 创建集合失败: {e}")
        return False


def import_data(client: MilvusClient) -> bool:
    """导入数据"""
    print("\n📊 步骤2: 导入Parquet数据")
    print("-" * 40)
    
    try:
        from ..core.parquet_importer import MilvusParquetImporter
        
        # 检查集合是否存在
        manager = client.collection_manager
        
        existing = manager.list_collections()
        if not existing:
            print("❌ 未找到任何集合，请先创建集合")
            return False
        
        print(f"✅ 找到 {len(existing)} 个集合")
        
        # 检查parquet文件
        parquet_dir = os.path.join(current_dir, "tests", "parquet")
        if not os.path.exists(parquet_dir):
            print(f"❌ Parquet目录不存在: {parquet_dir}")
            return False
        
        parquet_files = list(Path(parquet_dir).glob("*.parquet"))
        if not parquet_files:
            print(f"❌ 在目录 {parquet_dir} 中未找到任何.parquet文件")
            return False
        
        print(f"📁 找到 {len(parquet_files)} 个parquet文件:")
        for file in parquet_files:
            print(f"   - {file.name}")
        
        # 导入数据
        importer = MilvusParquetImporter(manager)
        results = importer.import_directory(parquet_dir)
        
        print(f"\n📊 导入结果:")
        total = 0
        for file, count in results.items():
            print(f"   {file}: {count} 条")
            total += count
        print(f"\n总计: {total} 条数据")
        return True
        
    except Exception as e:
        logger.error(f"导入数据失败: {e}")
        print(f"❌ 导入数据失败: {e}")
        return False


def reset_database(client: MilvusClient) -> bool:
    """重置数据库"""
    print("\n🗑️  步骤3: 重置数据库")
    print("-" * 40)
    
    try:
        from .milvus_reset import MilvusReset
        
        # 复用同一个 MilvusClient，避免在脚本中直接管理连接
        reset_tool = MilvusReset(client=client)
        
        existing = reset_tool.list_collections()
        if not existing:
            print("ℹ️  数据库已经是空的")
            return True
        
        print(f"⚠️  将删除以下集合:")
        for collection in existing:
            if collection.startswith("graphrag_"):
                print(f"   - {collection}")
        
        confirmation = input("\n确定要删除所有GraphRAG集合吗？(y/N): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("操作已取消")
            return False
        
        dropped = reset_tool.drop_all_collections()
        print(f"✅ 成功删除 {dropped} 个集合")
        return True
        
    except Exception as e:
        logger.error(f"重置数据库失败: {e}")
        print(f"❌ 重置数据库失败: {e}")
        return False


def show_status(client: MilvusClient) -> bool:
    """显示集合状态"""
    print("\n📋 集合状态")
    print("-" * 40)
    
    try:
        manager = client.collection_manager
        
        existing = manager.list_collections()
        if not existing:
            print("ℹ️  未找到任何集合")
            return True
        
        print(f"📊 找到 {len(existing)} 个集合:")
        
        for collection_name in existing:
            # 提取集合类型
            if collection_name.startswith(manager.collection_prefix):
                collection_type = collection_name[len(manager.collection_prefix):]
                try:
                    info = manager.get_collection_info(collection_type)
                    print(f"   - {collection_name}: {info['num_entities']} 条记录")
                except Exception as e:
                    print(f"   - {collection_name}: 无法获取信息 ({e})")
            else:
                print(f"   - {collection_name}: 非GraphRAG集合")
        
        return True
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        print(f"❌ 获取状态失败: {e}")
        return False


def test_query(client: MilvusClient) -> bool:
    """测试查询功能"""
    print("\n🔍 测试查询功能")
    print("-" * 40)
    
    try:
        # 检查集合是否存在
        manager = client.collection_manager
        
        existing = manager.list_collections()
        if not existing:
            print("❌ 未找到任何集合，请先创建集合并导入数据")
            return False
        
        # 使用统一的 MilvusClient 中的查询管理器
        query_manager = client.query_manager
        
        # 测试查询
        test_query = input("请输入测试查询文本 (默认: '人工智能'): ").strip()
        if not test_query:
            test_query = "人工智能"
        
        print(f"\n🔍 搜索: '{test_query}'")
        
        # 在多个集合中搜索
        collection_types = ["relationship", "text_unit", "entity_title"]
        results = query_manager.search_multiple_collections(test_query, collection_types, 3)
        
        for collection_type, collection_results in results.items():
            print(f"\n📊 {collection_type}: {len(collection_results)} 个结果")
            for i, result in enumerate(collection_results[:2], 1):
                score = result.get('score', 0)
                print(f"   {i}. 相似度: {score:.3f}")
                
                if 'title' in result:
                    title = result['title'][:50] + "..." if len(result['title']) > 50 else result['title']
                    print(f"      标题: {title}")
                elif 'description' in result:
                    desc = result['description'][:50] + "..." if len(result['description']) > 50 else result['description']
                    print(f"      描述: {desc}")
                elif 'text' in result:
                    text = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
                    print(f"      文本: {text}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试查询失败: {e}")
        print(f"❌ 测试查询失败: {e}")
        return False


def main():
    """主函数"""
    # 读取环境变量配置（使用 python-dotenv）
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    load_dotenv(os.path.join(project_root, ".env"))
    
    # 获取并显示Milvus配置
    from ..core.config import get_milvus_config
    milvus_config = get_milvus_config()
    
    print(f"ℹ️ Milvus配置: {milvus_config.host}:{milvus_config.port}")
    print(f"ℹ️ Lite模式: {milvus_config.use_lite}")
    if milvus_config.use_lite:
        print(f"ℹ️ 数据库文件: {milvus_config.lite_db_path}")
    
    # 在整个交互过程中只创建并复用一个 MilvusClient
    client = MilvusClient(
        host=milvus_config.host or "localhost",
        port=str(milvus_config.port or 19530),
    )
    
    try:
        client.connect()
        
        while True:
            show_menu()
            
            try:
                choice = input("\n请选择操作 (0-5): ").strip()
                
                if choice == "0":
                    print("👋 再见!")
                    break
                elif choice == "1":
                    create_collections(client)
                elif choice == "2":
                    import_data(client)
                elif choice == "3":
                    reset_database(client)
                elif choice == "4":
                    show_status(client)
                elif choice == "5":
                    test_query(client)
                else:
                    print("❌ 无效选择，请输入 0-5")
                    
            except KeyboardInterrupt:
                print("\n\n👋 操作已取消，再见!")
                break
            except Exception as e:
                logger.error(f"操作失败: {e}")
                print(f"❌ 操作失败: {e}")
            
            input("\n按回车键继续...")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
