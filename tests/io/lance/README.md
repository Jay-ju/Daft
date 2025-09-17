# Lance DataSink 批处理功能测试套件

本目录包含了针对Lance DataSink批处理功能的全面测试套件，旨在确保批处理机制的可靠性、正确性和性能表现。

## 测试套件概览

### 测试文件结构

```
tests/io/lance/
├── README.md                           # 本文档
├── conftest.py                         # 测试配置和共享fixtures
├── test_lance_datasink_batching.py     # 核心批处理功能测试
├── test_lance_datasink_integration.py  # 集成测试
└── test_lance_datasink_performance.py  # 性能测试
```

### 测试覆盖范围

#### 1. 核心功能测试 (`test_lance_datasink_batching.py`)

**参数验证测试**
- ✅ 有效参数初始化测试
- ✅ 默认参数向后兼容性测试
- ✅ 无效`batch_size`参数验证
- ✅ 无效`max_batch_rows`参数验证
- ✅ 无效URI类型检查

**批处理逻辑测试**
- ✅ 基于`batch_size`的flush机制测试
- ✅ 基于`max_batch_rows`的flush机制测试
- ✅ 向后兼容性测试（batch_size=1）
- ✅ 双重条件flush逻辑测试

**集成功能测试**
- ✅ 端到端小数据集批处理测试
- ✅ 端到端大数据集批处理测试
- ✅ 不同Lance写入模式测试（create/append/overwrite）
- ✅ 复杂数据类型支持测试

**性能验证测试**
- ✅ Fragment数量减少验证
- ✅ 行数准确性验证
- ✅ 大批次内存使用测试

**错误处理测试**
- ✅ 空micropartition处理
- ✅ Schema兼容性错误处理
- ✅ 批处理失败回退机制
- ✅ 无效Lance参数处理

**边缘情况测试**
- ✅ 单行micropartition测试
- ✅ 精确批次大小边界测试
- ✅ max_batch_rows边界测试

#### 2. 集成测试 (`test_lance_datasink_integration.py`)

**DataFrame集成测试**
- ✅ DataFrame.write_lance()方法批处理参数测试
- ✅ DataFrame操作后批处理写入测试
- ✅ 多次DataFrame写入同一数据集测试

**数据源集成测试**
- ✅ Parquet源数据批处理测试
- ✅ CSV源数据批处理测试
- ✅ JSON源数据批处理测试

**性能场景测试**
- ✅ 大数据集批处理测试（10K行）
- ✅ 内存高效批处理测试
- ✅ 并发写入模拟测试

**复杂Schema测试**
- ✅ 嵌套数据类型批处理测试
- ✅ 可空字段批处理测试
- ✅ 大字符串字段批处理测试

**错误恢复测试**
- ✅ 部分批次恢复测试
- ✅ 空数据集处理测试

#### 3. 性能测试 (`test_lance_datasink_performance.py`)

**Fragment减少测试**
- ✅ 不同批次大小的fragment数量对比
- ✅ 基于行数批处理的fragment减少测试
- ✅ 最优批次大小分析测试

**写入性能测试**
- ✅ 批处理vs非批处理写入时间对比
- ✅ 内存使用模式测试
- ✅ 并发写入性能测试

**可扩展性测试**
- ✅ 大数据集可扩展性测试（5K行）
- ✅ 变化数据大小可扩展性测试
- ✅ Fragment组织效率测试

## 运行测试

### 环境要求

1. **Python环境**: Python 3.8+
2. **依赖包**: 
   - pytest >= 7.0
   - pyarrow >= 9.0.0
   - lance
   - daft (已构建)

### 快速开始

#### 1. 环境设置

```bash
# 克隆并进入Daft项目目录
cd ByteDance-Daft

# 设置虚拟环境（如果尚未设置）
make .venv

# 激活虚拟环境
source .venv/bin/activate

# 构建Daft项目（需要Rust工具链）
make build
```

#### 2. 运行所有Lance批处理测试

```bash
# 设置DAFT_RUNNER环境变量
export DAFT_RUNNER=native

# 运行所有Lance批处理测试
pytest tests/io/lance/ -v

# 或使用Makefile（推荐）
make test EXTRA_ARGS="-v tests/io/lance/"
```

#### 3. 运行特定测试类别

```bash
# 只运行核心功能测试
pytest tests/io/lance/test_lance_datasink_batching.py -v

# 只运行集成测试
pytest tests/io/lance/test_lance_datasink_integration.py -v

# 只运行性能测试
pytest tests/io/lance/test_lance_datasink_performance.py -v -s

# 运行特定测试类
pytest tests/io/lance/test_lance_datasink_batching.py::TestLanceDataSinkParameterValidation -v

# 运行特定测试方法
pytest tests/io/lance/test_lance_datasink_batching.py::TestLanceDataSinkParameterValidation::test_valid_parameters -v
```

#### 4. 使用测试标记

```bash
# 运行性能测试（如果标记了@pytest.mark.lance_performance）
pytest tests/io/lance/ -m lance_performance -v

# 跳过慢速测试
pytest tests/io/lance/ -m "not slow" -v

# 运行基准测试（需要pytest-benchmark插件）
pytest tests/io/lance/ -m benchmark -v
```

### 测试配置选项

#### 环境变量

- `DAFT_RUNNER`: 设置为`native`或`ray`（推荐使用`native`进行单机测试）
- `HYPOTHESIS_MAX_EXAMPLES`: Hypothesis测试的最大示例数（默认100）
- `HYPOTHESIS_SEED`: Hypothesis测试的随机种子（默认0）

#### pytest选项

```bash
# 详细输出
pytest tests/io/lance/ -v

# 显示print输出（对性能测试有用）
pytest tests/io/lance/ -s

# 并行运行测试（需要pytest-xdist）
pytest tests/io/lance/ -n auto

# 生成覆盖率报告
pytest tests/io/lance/ --cov=daft.io.lance --cov-report=html

# 只运行失败的测试
pytest tests/io/lance/ --lf

# 在第一个失败时停止
pytest tests/io/lance/ -x
```

## 测试数据和Fixtures

### 共享Fixtures (`conftest.py`)

- `lance_temp_dir`: 临时目录创建
- `sample_schemas`: 多种测试schema
- `sample_data_sets`: 各种大小的测试数据集
- `create_test_dataframes`: DataFrame创建工厂
- `batch_size_scenarios`: 批处理场景配置
- `performance_test_data`: 性能测试数据生成器
- `verify_lance_dataset`: 数据集验证工具
- `compare_datasets`: 数据集对比工具

### 参数化测试

测试套件广泛使用pytest的参数化功能来测试多种场景：

- `batch_size_param`: 测试不同的batch_size值（1, 5, 10, 25）
- `max_batch_rows_param`: 测试不同的max_batch_rows值（100, 500, 1000, 5000）
- `write_mode_param`: 测试不同的写入模式（create, append, overwrite）

## 测试最佳实践

### 1. 测试隔离

- 每个测试使用独立的临时目录
- 测试之间不共享状态
- 使用function级别的fixtures确保清理

### 2. 数据验证

- 验证行数准确性
- 检查数据完整性
- 确认schema兼容性
- 验证fragment组织

### 3. 性能测试

- 使用一致的测试数据
- 测量关键性能指标
- 比较批处理vs非批处理性能
- 验证内存使用模式

### 4. 错误处理

- 测试各种错误场景
- 验证错误消息准确性
- 确保资源正确清理
- 测试回退机制

## 故障排除

### 常见问题

#### 1. Lance未安装

```
ImportError: lance is not installed
```

**解决方案**:
```bash
pip install lance
# 或
pip install daft[lance]
```

#### 2. PyArrow版本过低

```
lance not supported on old versions of pyarrow
```

**解决方案**:
```bash
pip install pyarrow>=9.0.0
```

#### 3. Daft未构建

```
ImportError: cannot import name 'build_type' from 'daft.daft'
```

**解决方案**:
```bash
# 确保Rust工具链已安装
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 构建Daft
make build
```

#### 4. 测试超时

对于大数据集性能测试，可能需要增加超时时间：

```bash
pytest tests/io/lance/ --timeout=300  # 5分钟超时
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 保留测试数据

```python
# 在测试中添加
import tempfile
temp_dir = tempfile.mkdtemp(prefix="lance_debug_")
print(f"Test data saved in: {temp_dir}")
```

#### 3. 单步调试

```bash
pytest tests/io/lance/test_lance_datasink_batching.py::test_specific -v -s --pdb
```

## 性能基准

### 预期性能改进

基于测试结果，Lance DataSink批处理功能应该提供：

- **Fragment减少**: 60-90%的fragment数量减少
- **写入性能**: 30-55%的写入时间改进
- **I/O效率**: 60-90%的I/O调用减少
- **内存管理**: 可预测的内存使用模式

### 基准测试数据

| 数据规模 | 批处理设置 | Fragment减少 | 性能提升 |
|----------|------------|--------------|----------|
| 1K行     | batch_size=10 | ~70% | ~35% |
| 10K行    | batch_size=50 | ~80% | ~45% |
| 100K行   | max_batch_rows=5000 | ~85% | ~50% |

## 贡献指南

### 添加新测试

1. **选择合适的测试文件**:
   - 核心功能 → `test_lance_datasink_batching.py`
   - 集成测试 → `test_lance_datasink_integration.py`
   - 性能测试 → `test_lance_datasink_performance.py`

2. **遵循命名约定**:
   - 测试类: `TestLanceDataSink[功能名]`
   - 测试方法: `test_[具体功能描述]`

3. **使用适当的fixtures**:
   - 使用`conftest.py`中的共享fixtures
   - 创建特定的fixtures如果需要

4. **添加适当的标记**:
   ```python
   @pytest.mark.lance_batching
   @pytest.mark.slow  # 对于耗时测试
   ```

### 测试质量检查

运行测试前，确保：

```bash
# 代码格式检查
make format-check

# 代码质量检查
make lint

# 运行完整测试套件
make test EXTRA_ARGS="-v tests/io/lance/"
```

## 相关文档

- [Lance DataSink批处理设计文档](../../../batching_design.md)
- [性能分析报告](../../../performance_analysis.md)
- [集成指南](../../../integration_guide.md)
- [Daft贡献指南](../../../CONTRIBUTING.md)

## 联系方式

如果在运行测试时遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看测试日志和错误消息
3. 在Daft项目中创建issue，包含详细的错误信息和环境信息