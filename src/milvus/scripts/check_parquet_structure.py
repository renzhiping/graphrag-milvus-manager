#!/usr/bin/env python3
"""
检查Parquet文件结构，了解数据格式
"""

import os
import pandas as pd

def check_parquet_structure(directory_path):
    """检查目录中所有Parquet文件的结构"""
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return
    
    print(f"检查目录: {directory_path}")
    print("=" * 60)
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # 读取Parquet文件
                df = pd.read_parquet(file_path)
                
                print(f"\n文件: {filename}")
                print(f"形状: {df.shape}")
                print("列名:", df.columns.tolist())
                print("前3行数据:")
                print(df.head(3))
                print("-" * 40)
                
            except Exception as e:
                print(f"读取文件 {filename} 失败: {e}")

if __name__ == "__main__":
    parquet_dir = "/Users/renzhiping/workspace2/graphrag210/graphrag/index/dgraph/tests/parquet"
    check_parquet_structure(parquet_dir)