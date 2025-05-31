# OpenAI to Z Challenge - 亚马逊考古遗址发现系统

基于OpenAI模型的亚马逊地区考古遗址发现和分析系统，整合了三个demo的功能。

## 项目概述

本项目实现了OpenAI to Z挑战赛的核心功能，旨在利用OpenAI的GPT模型发现亚马逊地区的未知考古遗址。系统整合了卫星图像分析、LiDAR数据处理、地理空间分析和AI驱动的考古评估。

## 主要功能

### 🔍 核心功能
- **多数据源集成**: 支持Sentinel-2卫星图像、LiDAR高程数据、地理符号数据
- **AI驱动分析**: 使用GPT-4.1、GPT-4o等模型进行考古特征识别
- **异常检测**: 自动识别高考古价值的地点
- **可重现性**: 确保分析结果的一致性和可验证性
- **地理空间处理**: 支持WKT格式的边界框和坐标系统

### 📊 检查点实现
- **检查点1**: 基础数据下载和OpenAI模型调用
- **检查点2**: 多数据源挖掘和异常足迹检测
- **完整分析**: 综合所有功能的端到端分析流程

## 安装和设置

### 1. 环境要求
- Python 3.8+
- OpenAI API密钥

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 设置API密钥
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

## 使用方法

### 快速开始

运行简化版本（基于demo3）：
```bash
python run_challenge.py
```

运行完整版本（整合所有demo功能）：
```bash
python openai_to_z_challenge.py
```

### 自定义分析

```python
from openai_to_z_challenge import OpenAIToZChallenge

# 初始化挑战赛
challenge = OpenAIToZChallenge(seed=42)

# 运行特定检查点
results1 = challenge.run_checkpoint_1()
results2 = challenge.run_checkpoint_2()

# 运行完整分析
full_results = challenge.run_full_analysis()
```

## 输出结果

### 文件结构
```
outputs/
├── checkpoint1_results.json     # 检查点1结果
├── checkpoint2_results.json     # 检查点2结果
├── final_analysis_results.json  # 完整分析结果
├── tiles/                       # 卫星瓦片图像
│   ├── tile_-12.626200_-53.049900.png
│   └── ...
├── lidar_data.png              # LiDAR数据可视化
└── sentinel2_data.png          # Sentinel-2数据
```

### 结果格式

#### 异常足迹（Anomaly Footprints）
```json
{
  "id": "upper_xingu_point_1",
  "center_lat": -12.6262,
  "center_lon": -53.0499,
  "bbox_wkt": "POLYGON((-53.0599 -12.6362, -53.0399 -12.6362, ...))",
  "significance": 8.5,
  "region": "Upper Xingu"
}
```

#### AI分析结果
```json
{
  "point_id": "upper_xingu_point_1",
  "location": "-12.6262, -53.0499",
  "analysis": "该位置显示出强烈的考古潜力...",
  "model_used": "gpt-4.1",
  "success": true
}
```

## 技术特性

### 🔄 可重现性
- 固定随机种子（默认42）
- 哈希验证确保结果一致性
- 详细的提示日志记录

### 🌍 地理覆盖
- **上辛古盆地** (Upper Xingu): -13.0°到-12.0°S, -53.5°到-52.5°W
- **巴西北部** (Northern Brazil): -3.5°到-2.5°S, -61.0°到-60.0°W
- **朗多尼亚州** (Rondonia): -9.5°到-8.5°S, -65.5°到-64.5°W

### 🤖 支持的AI模型
- GPT-4.1 (挑战赛推荐)
- GPT-4o
- GPT-4o-mini

## 数据源

### 已实现的数据源
1. **Sentinel-2光学图像** - 10m分辨率，13波段场景
2. **LiDAR高程数据** - 高分辨率穿透树冠的高程数据
3. **地理符号数据** - 已知考古特征的地理位置
4. **TerraBrasilis森林砍伐数据** - 巴西亚马逊森林变化监测

### 潜在扩展数据源
- GEDI森林结构数据
- SRTM数字高程模型
- 历史殖民地日记
- 土著口述地图

## 分析方法

### 考古特征检测
1. **几何形状识别** - 圆形、矩形、线性特征
2. **植被异常分析** - 揭示埋藏结构的植被模式
3. **土壤标记检测** - 古代定居点的土壤痕迹
4. **地形异常** - 人工土丘、凹陷、堤道

### AI提示策略
- 专业考古学背景设定
- 多层次分析（地形、植被、几何）
- 置信度评分（0-1）
- 具体坐标标识

## 验证和质量控制

### 可重现性验证
```bash
# 多次运行应产生相同结果（±50m精度）
python run_challenge.py
python run_challenge.py
# 比较reproducibility_hash值
```

### 结果验证
- 自动异常检测阈值调整
- 多模型交叉验证
- 地理合理性检查

## 扩展功能

### 高级地图功能（需要geemap）
```python
# 交互式地图可视化
if GEEMAP_AVAILABLE:
    challenge.create_interactive_map()
```

### 网页截图（需要selenium）
```python
# 自动生成卫星瓦片截图
if SELENIUM_AVAILABLE:
    challenge.create_satellite_screenshots()
```

## 故障排除

### 常见问题

1. **API密钥错误**
   ```
   ⚠️ 请设置OPENAI_API_KEY环境变量
   ```
   解决：确保正确设置OpenAI API密钥

2. **依赖包缺失**
   ```
   ImportError: No module named 'geopandas'
   ```
   解决：运行 `pip install -r requirements.txt`

3. **模型访问错误**
   ```
   Error: Model 'gpt-4.1' not available
   ```
   解决：检查API密钥权限或使用其他可用模型

### 性能优化
- 减少API调用次数
- 批量处理数据点
- 缓存中间结果

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目基于MIT许可证开源。

## 致谢

- OpenAI to Z Challenge组织者
- 亚马逊考古学研究社区
- 开源地理空间数据提供者

## 联系信息

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 参与Kaggle竞赛讨论

---

**注意**: 本实现仅用于研究和教育目的。实际考古发现需要专业考古学家的现场验证。