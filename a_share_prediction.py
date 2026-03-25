#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合预测脚本
集成上证指数、深证成指预测，以及A股个股预测
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 获取项目目录
data_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'output')

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 导入预测模块
try:
    from ml_services.sse_prediction import SSE_Predictor
    SSE_PREDICTION_AVAILABLE = True
except ImportError:
    SSE_PREDICTION_AVAILABLE = False
    print("⚠️ 上证指数预测模块不可用")

try:
    from ml_services.szse_prediction import SZSE_Predictor
    SZSE_PREDICTION_AVAILABLE = True
except ImportError:
    SZSE_PREDICTION_AVAILABLE = False
    print("⚠️ 深证成指预测模块不可用")

# 导入配置
try:
    from config_a_share import WATCHLIST, INDEX_MAPPING
except ImportError:
    print("⚠️ A股配置模块不可用")
    WATCHLIST = {}
    INDEX_MAPPING = {}


class ASharePredictionSystem:
    """A股综合预测系统"""

    def __init__(self):
        self.predictors = {}
        self.reports = {}

    def predict_sse(self):
        """预测上证指数"""
        if not SSE_PREDICTION_AVAILABLE:
            print("⚠️ 上证指数预测模块不可用，跳过")
            return None

        try:
            print("\n" + "=" * 100)
            print("上证指数预测")
            print("=" * 100)

            predictor = SSE_Predictor()
            report = predictor.generate_report()

            if report:
                self.predictors['sse'] = predictor
                self.reports['sse'] = report

            return report

        except Exception as e:
            print(f"❌ 上证指数预测失败: {e}")
            return None

    def predict_szse(self):
        """预测深证成指"""
        if not SZSE_PREDICTION_AVAILABLE:
            print("⚠️ 深证成指预测模块不可用，跳过")
            return None

        try:
            print("\n" + "=" * 100)
            print("深证成指预测")
            print("=" * 100)

            predictor = SZSE_Predictor()
            report = predictor.generate_report()

            if report:
                self.predictors['szse'] = predictor
                self.reports['szse'] = report

            return report

        except Exception as e:
            print(f"❌ 深证成指预测失败: {e}")
            return None

    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "=" * 100)
        print("A股市场综合预测报告")
        print("=" * 100)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 汇总预测结果
        summary = {
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predictions': {}
        }

        # 上证指数预测
        if 'sse' in self.reports:
            sse_report = self.reports['sse']
            summary['predictions']['sse'] = {
                'index_name': '上证指数',
                'index_code': '000001',
                'current_price': sse_report['current_price'],
                'price_change': sse_report['price_change'],
                'prediction_score': sse_report['prediction_score'],
                'prediction_trend': sse_report['prediction_trend']
            }

            print(f"\n📊 上证指数 (000001)")
            print(f"  当前价格: {sse_report['current_price']:.2f}")
            print(f"  价格变化: {sse_report['price_change']:+.2f}%")
            print(f"  预测得分: {sse_report['prediction_score']:.4f}")
            print(f"  预测趋势: {sse_report['prediction_trend']}")

        # 深证成指预测
        if 'szse' in self.reports:
            szse_report = self.reports['szse']
            summary['predictions']['szse'] = {
                'index_name': '深证成指',
                'index_code': '399001',
                'current_price': szse_report['current_price'],
                'price_change': szse_report['price_change'],
                'prediction_score': szse_report['prediction_score'],
                'prediction_trend': szse_report['prediction_trend']
            }

            print(f"\n📊 深证成指 (399001)")
            print(f"  当前价格: {szse_report['current_price']:.2f}")
            print(f"  价格变化: {szse_report['price_change']:+.2f}%")
            print(f"  预测得分: {szse_report['prediction_score']:.4f}")
            print(f"  预测趋势: {szse_report['prediction_trend']}")

        # 综合判断
        print("\n" + "=" * 100)
        print("📈 综合市场判断")

        if 'sse' in summary['predictions'] and 'szse' in summary['predictions']:
            sse_score = summary['predictions']['sse']['prediction_score']
            szse_score = summary['predictions']['szse']['prediction_score']
            avg_score = (sse_score + szse_score) / 2

            summary['average_score'] = avg_score

            if avg_score >= 0.65:
                overall_trend = "强烈看涨"
                trend_emoji = "🟢🟢🟢"
            elif avg_score >= 0.55:
                overall_trend = "看涨"
                trend_emoji = "🟢🟢"
            elif avg_score >= 0.50:
                overall_trend = "中性偏涨"
                trend_emoji = "🟢"
            elif avg_score >= 0.45:
                overall_trend = "中性偏跌"
                trend_emoji = "🔴"
            elif avg_score >= 0.35:
                overall_trend = "看跌"
                trend_emoji = "🔴🔴"
            else:
                overall_trend = "强烈看跌"
                trend_emoji = "🔴🔴🔴"

            summary['overall_trend'] = overall_trend

            print(f"  综合得分: {avg_score:.4f}")
            print(f"  综合趋势: {overall_trend} {trend_emoji}")

        # 保存JSON报告
        report_file = os.path.join(output_dir, f'a_share_prediction_report_{timestamp}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 综合报告已保存: {report_file}")

        print("\n" + "=" * 100)
        print("✅ A股综合预测完成")
        print("=" * 100)

        return summary

    def run(self, indices=['sse', 'szse']):
        """运行综合预测"""
        print("=" * 100)
        print("A股综合预测系统")
        print("=" * 100)
        print(f"📅 分析日期: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"📊 预测指数: {', '.join([INDEX_MAPPING.get(idx, {}).get('name', idx) for idx in indices])}")

        # 预测各个指数
        for index in indices:
            if index == 'sse':
                self.predict_sse()
            elif index == 'szse':
                self.predict_szse()

        # 生成综合报告
        return self.generate_comprehensive_report()


def main():
    parser = argparse.ArgumentParser(description='A股综合预测系统')
    parser.add_argument('--indices', type=str, default='sse,szse',
                        help='预测的指数，逗号分隔（sse,szse,csi300）')
    parser.add_argument('--all', action='store_true', help='预测所有支持的指数')

    args = parser.parse_args()

    # 确定预测指数列表
    if args.all:
        indices = ['sse', 'szse']  # 目前只支持这两个
    else:
        indices = [idx.strip().lower() for idx in args.indices.split(',')]

    # 创建预测系统
    system = ASharePredictionSystem()

    # 运行预测
    summary = system.run(indices=indices)

    # 输出结果
    if summary:
        print("\n预测摘要:")
        for index_name, pred in summary.get('predictions', {}).items():
            print(f"  {pred['index_name']} ({pred['index_code']}): {pred['prediction_trend']}")
        if 'overall_trend' in summary:
            print(f"  综合趋势: {summary['overall_trend']}")


if __name__ == '__main__':
    main()