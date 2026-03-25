# A股系统使用指南

## 概述

本项目已完成从港股系统到A股系统的改造，支持上证指数、深证成指的预测分析。

## 核心模块

### 1. A股数据获取 (`data_services/a_share_finance.py`)
- 支持上证指数、深证成指、沪深300指数数据获取
- 支持A股个股数据获取
- 内置重试机制，保证数据获取稳定性

### 2. 指数预测模块
- `ml_services/sse_prediction.py` - 上证指数预测
- `ml_services/szse_prediction.py` - 深证成指预测

### 3. A股配置 (`config_a_share.py`)
- A股股票池（沪深300核心成分股）
- 板块映射和评分配置
- 指数映射配置

### 4. 邮件系统 (`a_share_email.py`)
- A股市场分析报告生成
- 交易信号生成
- 邮件通知功能

### 5. 综合预测 (`a_share_prediction.py`)
- 集成上证指数和深证成指预测
- 生成综合市场判断报告

## 快速开始

### 1. 测试A股系统

```bash
python3 test_a_share_system.py
```

### 2. 运行上证指数预测

```bash
python3 ml_services/sse_prediction.py
```

### 3. 运行深证成指预测

```bash
python3 ml_services/szse_prediction.py
```

### 4. 运行A股综合预测

```bash
# 预测上证指数
python3 a_share_prediction.py --indices sse

# 预测深证成指
python3 a_share_prediction.py --indices szse

# 预测所有指数
python3 a_share_prediction.py --all
```

### 5. 运行A股邮件系统

```bash
# 仅生成报告（不发送邮件）
python3 a_share_email.py

# 生成报告并发送邮件
python3 a_share_email.py --send-email

# 需要设置环境变量
export SMTP_USERNAME=your_email@qq.com
export SMTP_PASSWORD=your_password
export RECIPIENT_EMAIL=recipient@qq.com
```

## 配置说明

### A股股票池

`config_a_share.py` 中定义了29只A股核心成分股：

**银行股**: 工商银行、招商银行、中国平安、兴业银行、平安银行
**科技股**: 五粮液、宁德时代、海康威视、美的集团、伊利股份
**新能源股**: 比亚迪、亿纬锂能、隆基绿能
**消费股**: 贵州茅台、泸州老窖、山西汾酒
**医药股**: 恒瑞医药、长春高新、迈瑞医疗
**汽车股**: 广汽集团、长城汽车、长安汽车
**房地产股**: 万科A、保利发展、招商蛇口
**能源股**: 中国石油、中国石化、中国神华
**公用事业股**: 长江电力、宝钢股份、紫金矿业
**保险股**: 中国太保、新华保险
**券商股**: 中信证券、华泰证券
**通信股**: 中国联通、中兴通讯

### 指数映射

- **上证指数**: 000001
- **深证成指**: 399001
- **沪深300**: 000300
- **上证50**: 000016

## 输出文件

### 预测报告

预测报告保存在 `output/` 目录：

- `sse_prediction_report_*.json` - 上证指数预测报告
- `szse_prediction_report_*.json` - 深证成指预测报告
- `a_share_prediction_report_*.json` - 综合预测报告

### 特征数据

特征数据保存在 `data/` 目录：

- `sse_prediction_features_*.csv` - 上证指数特征
- `szse_prediction_features_*.csv` - 深证成指特征

## 预测结果解读

### 预测得分范围

- **≥ 0.65**: 强烈看涨 🟢🟢🟢
- **0.55 - 0.64**: 看涨 🟢🟢
- **0.50 - 0.54**: 中性偏涨 🟢
- **0.45 - 0.49**: 中性偏跌 🔴
- **0.35 - 0.44**: 看跌 🔴🔴
- **< 0.35**: 强烈看跌 🔴🔴🔴

### 关键因素分析

预测系统基于19个技术指标进行加权评分：

1. **MA250** (15%权重) - 250日均线，长期趋势
2. **Volume_MA250** (12%权重) - 250日成交量均线
3. **MA120** (10%权重) - 120日均线，中期趋势
4. **RS_Signal**系列 (8%-2.5%权重) - 相对强度信号
5. **Trend**系列 (5%-3%权重) - 趋势信号
6. **Volatility_120d** (4%权重) - 120日波动率

## 依赖库

```
akshare        # A股数据获取
pandas         # 数据处理
numpy          # 数值计算
requests       # HTTP请求
```

## 注意事项

1. **数据源**: 使用AkShare获取A股数据，请确保网络连接正常
2. **交易时间**: A股交易时间为 9:30-11:30, 13:00-15:00
3. **涨跌幅限制**: 主板±10%，创业板/科创板±20%
4. **预测建议**: 本系统仅供参考，不构成投资建议

## 系统架构

```
fortune/
├── data_services/
│   └── a_share_finance.py      [NEW] A股数据获取
├── ml_services/
│   ├── sse_prediction.py       [NEW] 上证指数预测
│   └── szse_prediction.py      [NEW] 深证成指预测
├── config_a_share.py            [NEW] A股配置
├── a_share_email.py             [NEW] A股邮件系统
├── a_share_prediction.py        [NEW] A股综合预测
└── test_a_share_system.py       [NEW] 系统测试脚本
```

## 对比：港股系统 vs A股系统

| 功能 | 港股系统 | A股系统 |
|------|----------|---------|
| 数据源 | 腾讯财经 + yfinance | AkShare |
| 指数预测 | 恒生指数 (^HSI) | 上证/深证/沪深300 |
| 股票池 | 58只港股 | 29只A股核心 |
| 配置文件 | config.py | config_a_share.py |
| 邮件系统 | hsi_email.py | a_share_email.py |
| 预测脚本 | hsi_prediction.py | a_share_prediction.py |

## 下一步计划

1. [ ] 扩展A股个股预测和回测功能
2. [ ] 添加板块轮动分析
3. [ ] 集成ML模型训练功能
4. [ ] 添加实时数据推送功能
5. [ ] 完善邮件模板和报告格式

## 联系方式

如有问题或建议，请提交Issue或Pull Request。