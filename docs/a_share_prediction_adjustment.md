# A股预测系统调整说明

## 调整时间
2026-03-26

## 调整背景
通过对比分析发现，A股预测系统与港股预测系统在特征标准化参数上存在差异，导致A股预测得分过高（0.9686 vs 0.5512），无法准确反映市场情绪。

## 主要调整

### 1. 特征标准化参数统一

**调整前:**
- MA标准化: `np.tanh(value / 3500)`
- 波动率: `(value - 0.01) / 0.025`

**调整后（与港股HSI保持一致）:**
- MA标准化: `np.tanh(value / 50000)`
- 波动率: `(value - 0.02) / 0.03`

### 2. 影响文件
- `ml_services/sse_prediction.py` - 上证指数预测
- `ml_services/szse_prediction.py` - 深证成指预测

## 对比结果

### 调整前
- 上证指数预测得分: 0.9686 (强烈看涨)
- 深证成指预测得分: 0.9368 (强烈看涨)
- 恒生指数预测得分: 0.5512 (看涨)
- 差异: 0.4174

### 调整后（预期）
- 上证指数预测得分: ~0.55-0.65 (与港股相近)
- 深证成指预测得分: ~0.55-0.65 (与港股相近)
- 恒生指数预测得分: 0.5512 (看涨)
- 差异: <0.10

## 特征配置对比

| 特征名称 | 港股权重 | A股权重 | 状态 |
|----------|----------|---------|------|
| MA250 | 15.00% | 15.00% | ✅ 一致 |
| Volume_MA250 | 12.00% | 12.00% | ✅ 一致 |
| MA120 | 10.00% | 10.00% | ✅ 一致 |
| 60d_RS_Signal_MA250 | 8.00% | 8.00% | ✅ 一致 |
| 60d_RS_Signal_Volume_MA250 | 6.00% | 6.00% | ✅ 一致 |
| 60d_Trend_MA250 | 5.00% | 5.00% | ✅ 一致 |
| ... | ... | ... | ... |

## 数据源差异

### 港股 (HSI)
- ✅ 恒生指数 (^HSI) - yfinance
- ✅ 美国10年期国债收益率 (^TNX) - yfinance
- ✅ VIX恐慌指数 (^VIX) - yfinance

### A股 (SSE/SZSE)
- ✅ 上证指数 (000001) - AkShare
- ✅ 深证成指 (399001) - AkShare
- ❌ 缺少美股相关数据
- ❌ 缺少VIX恐慌指数

## 后续优化建议

### 1. 添加外部市场数据
```python
# 建议添加到A股预测系统
import yfinance as yf

# 获取美股数据
sp500 = yf.Ticker("^GSPC").history(period="2y")
nasdaq = yf.Ticker("^IXIC").history(period="2y")

# 获取VIX数据
vix = yf.Ticker("^VIX").history(period="2y")

# 获取美债收益率
us_yield = yf.Ticker("^TNX").history(period="2y")
```

### 2. 添加北向资金数据
```python
# 北向资金流向是A股重要指标
from akshare import stock_em_hsgt_north_net_flow_in_em

hsgt_data = stock_em_hsgt_north_net_flow_in_em(symbol="北向资金")
```

### 3. 添加外部影响因子
- 美股涨跌幅
- VIX恐慌指数
- 美债收益率变化
- 汇率波动（人民币/美元）

## 测试验证

运行对比脚本验证调整效果：
```bash
python3 compare_prediction_systems.py
```

运行A股预测系统：
```bash
python3 ml_services/sse_prediction.py
python3 ml_services/szse_prediction.py
```

## 注意事项

1. **参数调整影响**：调整后的标准化参数可能导致预测得分与之前有较大差异，这是正常的
2. **回测验证**：建议使用历史数据进行回测，验证调整后的模型准确性
3. **持续监控**：需要持续监控预测结果，根据实际表现进行微调

## 版本信息
- 调整前版本: A股预测系统 v1.0
- 调整后版本: A股预测系统 v1.1 (与港股HSI策略对齐)