#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股系统快速测试脚本
测试所有A股功能模块
"""

import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print("=" * 80)
print("A股系统测试")
print("=" * 80)

# 测试1: A股数据获取
print("\n[测试1] A股数据获取模块")
print("-" * 80)
try:
    from data_services.a_share_finance import get_sse_index_data, get_szse_index_data

    sse_data = get_sse_index_data(period_days=10)
    if sse_data is not None:
        print(f"✅ 上证指数数据获取成功，最新收盘价: {sse_data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ 上证指数数据获取失败")

    szse_data = get_szse_index_data(period_days=10)
    if szse_data is not None:
        print(f"✅ 深证成指数据获取成功，最新收盘价: {szse_data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ 深证成指数据获取失败")

except Exception as e:
    print(f"❌ 测试失败: {e}")

# 测试2: 上证指数预测
print("\n[测试2] 上证指数预测")
print("-" * 80)
try:
    from ml_services.sse_prediction import SSE_Predictor

    predictor = SSE_Predictor()
    report = predictor.generate_report()

    if report:
        print(f"✅ 上证指数预测完成")
        print(f"   预测得分: {report['prediction_score']:.4f}")
        print(f"   预测趋势: {report['prediction_trend']}")
    else:
        print("❌ 上证指数预测失败")

except Exception as e:
    print(f"❌ 测试失败: {e}")

# 测试3: 深证成指预测
print("\n[测试3] 深证成指预测")
print("-" * 80)
try:
    from ml_services.szse_prediction import SZSE_Predictor

    predictor = SZSE_Predictor()
    report = predictor.generate_report()

    if report:
        print(f"✅ 深证成指预测完成")
        print(f"   预测得分: {report['prediction_score']:.4f}")
        print(f"   预测趋势: {report['prediction_trend']}")
    else:
        print("❌ 深证成指预测失败")

except Exception as e:
    print(f"❌ 测试失败: {e}")

# 测试4: A股配置
print("\n[测试4] A股配置模块")
print("-" * 80)
try:
    from config_a_share import WATCHLIST, A_SHARE_STOCK_MAPPING, INDEX_MAPPING

    print(f"✅ A股配置加载成功")
    print(f"   自选股数量: {len(WATCHLIST)}")
    print(f"   股票池数量: {len(A_SHARE_STOCK_MAPPING)}")
    print(f"   指数配置: {', '.join(INDEX_MAPPING.keys())}")

except Exception as e:
    print(f"❌ 测试失败: {e}")

# 测试5: 综合预测
print("\n[测试5] A股综合预测")
print("-" * 80)
try:
    from a_share_prediction import ASharePredictionSystem

    system = ASharePredictionSystem()
    summary = system.run(indices=['sse', 'szse'])

    if summary:
        print(f"✅ 综合预测完成")
        if 'overall_trend' in summary:
            print(f"   综合趋势: {summary['overall_trend']}")
    else:
        print("❌ 综合预测失败")

except Exception as e:
    print(f"❌ 测试失败: {e}")

# 测试6: 邮件系统（仅测试数据获取，不发送邮件）
print("\n[测试6] A股邮件系统")
print("-" * 80)
try:
    from a_share_email import AShareEmailSystem

    system = AShareEmailSystem()
    system.run(send_email=False)  # 不发送邮件，仅测试

    print(f"✅ 邮件系统测试完成（未发送邮件）")

except Exception as e:
    print(f"❌ 测试失败: {e}")

print("\n" + "=" * 80)
print("所有测试完成")
print("=" * 80)