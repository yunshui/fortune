#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股vs港股预测对比分析
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print("=" * 100)
print(f"{'A股 vs 港股预测系统对比分析':^100}")
print("=" * 100)
print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 1. 特征重要性对比
print("=" * 100)
print("1. 特征重要性配置对比")
print("=" * 100)

# 港股特征配置
HSI_FEATURES = {
    'MA250': {'weight': 0.1500, 'direction': 1},
    'Volume_MA250': {'weight': 0.1200, 'direction': 1},
    'MA120': {'weight': 0.1000, 'direction': 1},
    '60d_RS_Signal_MA250': {'weight': 0.0800, 'direction': 1},
    '60d_RS_Signal_Volume_MA250': {'weight': 0.0600, 'direction': 1},
    '20d_RS_Signal_MA250': {'weight': 0.0400, 'direction': 1},
    '10d_RS_Signal_MA250': {'weight': 0.0350, 'direction': 1},
    '5d_RS_Signal_MA250': {'weight': 0.0300, 'direction': 1},
    '3d_RS_Signal_MA250': {'weight': 0.0250, 'direction': 1},
    '60d_Trend_MA250': {'weight': 0.0500, 'direction': 1},
    '20d_Trend_MA250': {'weight': 0.0450, 'direction': 1},
    '10d_Trend_MA250': {'weight': 0.0400, 'direction': 1},
    '5d_Trend_MA250': {'weight': 0.0350, 'direction': 1},
    '3d_Trend_MA250': {'weight': 0.0300, 'direction': 1},
    '60d_Trend_Volume_MA250': {'weight': 0.0450, 'direction': 1},
    '20d_Trend_Volume_MA250': {'weight': 0.0350, 'direction': 1},
    'Volatility_120d': {'weight': 0.0400, 'direction': -1},
    '60d_Trend_MA120': {'weight': 0.0350, 'direction': 1},
    '20d_RS_Signal_Volume_MA250': {'weight': 0.0300, 'direction': 1},
}

# A股特征配置（从SSE预测器读取）
try:
    from ml_services.sse_prediction import SSE_Predictor
    SSE_FEATURES = SSE_Predictor.FEATURE_IMPORTANCE
except:
    SSE_FEATURES = {}

print("\n📊 特征数量对比:")
print(f"  港股 (HSI): {len(HSI_FEATURES)} 个特征")
print(f"  A股 (SSE): {len(SSE_FEATURES)} 个特征")

print("\n📈 特征权重对比 (前10位):")
hsi_sorted = sorted(HSI_FEATURES.items(), key=lambda x: -x[1]['weight'])[:10]
sse_sorted = sorted(SSE_FEATURES.items(), key=lambda x: -x[1]['weight'])[:10]

print(f"\n{'特征名称':<30} {'港股权重':<12} {'A股权重':<12} {'差异':<10}")
print("-" * 70)

for i in range(10):
    hsi_name, hsi_config = hsi_sorted[i]
    sse_name, sse_config = sse_sorted[i] if i < len(sse_sorted) else (None, None)

    hsi_weight = f"{hsi_config['weight']*100:.2f}%"
    sse_weight = f"{sse_config['weight']*100:.2f}%" if sse_config else "N/A"
    diff = "相同" if hsi_name == sse_name else "不同"

    print(f"{hsi_name:<30} {hsi_weight:<12} {sse_weight:<12} {diff:<10}")

# 2. 数据源对比
print("\n" + "=" * 100)
print("2. 数据源对比")
print("=" * 100)

print("\n📊 港股 (HSI) 数据源:")
print("  ✓ 恒生指数 (^HSI) - yfinance")
print("  ✓ 美国10年期国债收益率 (^TNX) - yfinance")
print("  ✓ VIX恐慌指数 (^VIX) - yfinance")

print("\n📊 A股 (SSE) 数据源:")
print("  ✓ 上证指数 (000001) - AkShare")
print("  ✗ 缺少美股相关数据")
print("  ✗ 缺少VIX恐慌指数")

# 3. 特征标准化对比
print("\n" + "=" * 100)
print("3. 特征标准化参数对比")
print("=" * 100)

print("\n📊 港股 (HSI) 标准化参数:")
print("  MA标准化: np.tanh(value / 50000)")
print("  波动率: (value - 0.02) / 0.03, 范围[-1, 1]")
print("  VIX: (value - 20) / 30")

print("\n📊 A股 (SSE) 标准化参数:")
print("  MA标准化: np.tanh(value / 3500)")
print("  波动率: (value - 0.01) / 0.025, 范围[-1, 1]")

# 4. 运行预测对比
print("\n" + "=" * 100)
print("4. 实时预测对比")
print("=" * 100)

try:
    # 运行港股预测
    print("\n🔹 运行港股预测...")
    from hsi_prediction import HSI_Predictor
    hsi_predictor = HSI_Predictor()
    hsi_predictor.fetch_data()
    hsi_predictor.calculate_features()
    hsi_score, hsi_details = hsi_predictor.calculate_prediction_score()
    hsi_trend = hsi_predictor.interpret_score(hsi_score)[0]

    # 运行A股预测
    print("\n🔹 运行A股预测...")
    from ml_services.sse_prediction import SSE_Predictor
    sse_predictor = SSE_Predictor()
    sse_predictor.fetch_data()
    sse_predictor.calculate_features()
    sse_score, sse_details = sse_predictor.calculate_prediction_score()
    sse_trend = sse_predictor.interpret_score(sse_score)[0]

    # 显示对比结果
    print("\n" + "=" * 100)
    print("5. 预测结果对比")
    print("=" * 100)

    print(f"\n{'指数':<15} {'预测得分':<15} {'预测趋势':<15} {'相对强弱':<15}")
    print("-" * 70)
    print(f"{'恒生指数':<15} {hsi_score:<15.4f} {hsi_trend:<15} {'较高' if hsi_score > sse_score else '较低'}")
    print(f"{'上证指数':<15} {sse_score:<15.4f} {sse_trend:<15} {'较高' if sse_score > hsi_score else '较低'}")

    print(f"\n📊 得分差异: {abs(hsi_score - sse_score):.4f}")
    if abs(hsi_score - sse_score) < 0.05:
        print("  两个市场预测趋势基本一致")
    elif hsi_score > sse_score:
        print("  港股相对更强")
    else:
        print("  A股相对更强")

except Exception as e:
    print(f"\n❌ 预测对比失败: {e}")

# 6. 建议
print("\n" + "=" * 100)
print("6. 优化建议")
print("=" * 100)

print("\n🔧 建议为A股预测添加:")
print("  1. 美股指数数据 (标普500 SPX, 纳斯达克 IXIC)")
print("  2. VIX恐慌指数")
print("  3. 美债收益率数据")
print("  4. 北向资金数据")
print("  5. 新增外部市场影响因子")

print("\n🔧 建议调整特征标准化:")
print("  1. 根据A股历史数据重新校准MA标准化参数")
print("  2. 调整波动率标准化范围")
print("  3. 添加外部因子标准化方法")

print("\n" + "=" * 100)
print("✅ 对比分析完成")
print("=" * 100)