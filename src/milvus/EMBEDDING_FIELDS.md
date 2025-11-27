# 实体向量化字段定义

## 概述
本文档记录了 GraphRAG 系统中各个实体类型需要向量化的属性字段，基于 `/Users/renzhiping/workspace2/graphrag210/graphrag/index/workflows/generate_text_embeddings.py` 文件定义。

## 向量化字段配置

### 1. Document（文档）
- **字段**: `text`
- **嵌入配置**: `document_text_embedding`
- **数据源**: `documents` 表的 `text` 列

### 2. Relationship（关系）
- **字段**: `description`
- **嵌入配置**: `relationship_description_embedding`
- **数据源**: `relationships` 表的 `description` 列

### 3. TextUnit（文本单元）
- **字段**: `text`
- **嵌入配置**: `text_unit_text_embedding`
- **数据源**: `text_units` 表的 `text` 列

### 4. Entity（实体）
#### 4.1 标题嵌入
- **字段**: `title`
- **嵌入配置**: `entity_title_embedding`
- **数据源**: `entities` 表的 `title` 列

#### 4.2 标题+描述组合嵌入
- **字段**: `title_description`（组合字段）
- **嵌入配置**: `entity_description_embedding`
- **数据源**: 组合 `entities` 表的 `title` 和 `description` 列，格式为 `title + ":" + description`

### 5. CommunityReport（社区报告）
#### 5.1 标题嵌入
- **字段**: `title`
- **嵌入配置**: `community_title_embedding`
- **数据源**: `community_reports` 表的 `title` 列

#### 5.2 摘要嵌入
- **字段**: `summary`
- **嵌入配置**: `community_summary_embedding`
- **数据源**: `community_reports` 表的 `summary` 列

#### 5.3 完整内容嵌入
- **字段**: `full_content`
- **嵌入配置**: `community_full_content_embedding`
- **数据源**: `community_reports` 表的 `full_content` 列

## 嵌入配置映射

| 嵌入配置名称 | 实体类型 | 字段 | 数据表 | 列名 |
|-------------|----------|------|--------|------|
| `document_text_embedding` | Document | text | documents | text |
| `relationship_description_embedding` | Relationship | description | relationships | description |
| `text_unit_text_embedding` | TextUnit | text | text_units | text |
| `entity_title_embedding` | Entity | title | entities | title |
| `entity_description_embedding` | Entity | title_description | entities | title + description |
| `community_title_embedding` | CommunityReport | title | community_reports | title |
| `community_summary_embedding` | CommunityReport | summary | community_reports | summary |
| `community_full_content_embedding` | CommunityReport | full_content | community_reports | full_content |

## 技术实现

### 嵌入生成流程
1. 从配置中获取需要嵌入的字段集合 (`get_embedded_fields`)
2. 获取嵌入设置配置 (`get_embedding_settings`)
3. 根据 `embedding_param_map` 映射表准备数据
4. 对每个需要嵌入的字段调用 `_run_and_snapshot_embeddings` 生成嵌入向量
5. 结果包含 `id` 和 `embedding` 两列

### 数据预处理
- Entity 的 description 嵌入会组合 title 和 description 字段
- 所有嵌入操作都基于 Pandas DataFrame
- 嵌入结果会缓存到 PipelineCache 中

## 使用说明

这些嵌入配置用于构建知识图谱的向量表示，支持后续的相似性搜索和语义检索功能。每个嵌入配置对应特定的业务场景和查询需求。

---
**最后更新**: 2025-08-23  
**来源文件**: `/Users/renzhiping/workspace2/graphrag210/graphrag/index/workflows/generate_text_embeddings.py`