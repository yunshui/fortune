#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股市场邮件通知系统
基于技术分析指标生成买卖信号，只在有交易信号时发送邮件

参考 hsi_email.py，适配A股市场
"""

import os
import warnings

# 抑制 pkg_resources 弃用警告
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:py_mini_racer.*'
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module='py_mini_racer')

import smtplib
import json
import argparse
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# 导入技术分析工具
try:
    from data_services.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析模块不可用")

# 导入A股数据获取模块
try:
    from data_services.a_share_finance import (
        get_sse_index_data,
        get_szse_index_data,
        get_csi300_index_data,
        get_a_stock_data
    )
    A_SHARE_DATA_AVAILABLE = True
except ImportError:
    A_SHARE_DATA_AVAILABLE = False
    print("⚠️ A股数据获取模块不可用")

# 从A股配置导入股票列表
try:
    from config_a_share import WATCHLIST, INDEX_MAPPING
    STOCK_LIST = WATCHLIST
    TOTAL_STOCKS_COUNT = len(WATCHLIST)
except ImportError:
    print("⚠️ A股配置模块不可用，使用默认配置")
    STOCK_LIST = {
        "600519.SH": "贵州茅台",
        "000001.SZ": "平安银行",
    }
    TOTAL_STOCKS_COUNT = len(STOCK_LIST)


class AShareEmailSystem:
    """A股市场邮件通知系统"""

    # 根据投资风格和计算窗口确定历史数据长度
    DATA_PERIOD_CONFIG = {
        'ultra_short_term': '6mo',    # 超短线：6个月数据（约125个交易日）
        'short_term': '1y',           # 波段交易：1年数据（约247个交易日）
        'medium_long_term': '2y',      # 中长期投资：2年数据（约493个交易日）
    }

    def __init__(self, stock_list=None, investment_style='short_term',
                 min_risk_level=5, max_positions=10,
                 enable_diversification=True, enable_sector_rotation=True,
                 smtp_server='smtp.qq.com', smtp_port=465,
                 smtp_username=None, smtp_password=None, recipient_email=None):
        """
        初始化A股邮件系统

        参数:
        - stock_list: 股票代码列表
        - investment_style: 投资风格
        - min_risk_level: 最低风险等级（0-10）
        - max_positions: 最大持仓数量
        - enable_diversification: 启用分散投资
        - enable_sector_rotation: 启用板块轮动
        - smtp_server: SMTP服务器
        - smtp_port: SMTP端口
        - smtp_username: SMTP用户名
        - smtp_password: SMTP密码
        - recipient_email: 收件人邮箱
        """
        self.stock_list = stock_list or list(STOCK_LIST.keys())
        self.investment_style = investment_style
        self.min_risk_level = min_risk_level
        self.max_positions = max_positions
        self.enable_diversification = enable_diversification
        self.enable_sector_rotation = enable_sector_rotation
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.recipient_email = recipient_email

        # 技术分析器
        self.technical_analyzer = TechnicalAnalyzer() if TECHNICAL_ANALYSIS_AVAILABLE else None

        # 缓存数据
        self.cache = {
            'sse_data': None,
            'szse_data': None,
            'stock_data': {},
            'last_update': None
        }

        # 配置数据周期
        self.period_days = {
            'ultra_short_term': 120,
            'short_term': 240,
            'medium_long_term': 480
        }.get(investment_style, 240)

    def get_market_data(self):
        """获取市场数据"""
        print("📊 获取A股市场数据...")

        try:
            # 获取上证指数数据
            self.cache['sse_data'] = get_sse_index_data(period_days=self.period_days)

            # 获取深证成指数据
            self.cache['szse_data'] = get_szse_index_data(period_days=self.period_days)

            # 获取个股数据
            for stock_code in self.stock_list:
                try:
                    self.cache['stock_data'][stock_code] = get_a_stock_data(stock_code, period_days=self.period_days)
                    time.sleep(0.5)  # 避免请求过快
                except Exception as e:
                    print(f"⚠️ 获取 {stock_code} 数据失败: {e}")
                    continue

            self.cache['last_update'] = datetime.now()
            print(f"✅ 市场数据获取完成")

        except Exception as e:
            print(f"❌ 获取市场数据失败: {e}")

    def generate_trading_signals(self):
        """生成交易信号"""
        print("🎯 生成交易信号...")

        signals = []

        for stock_code in self.stock_list:
            stock_name = STOCK_LIST.get(stock_code, stock_code)
            stock_data = self.cache['stock_data'].get(stock_code)

            if stock_data is None or len(stock_data) < 20:
                continue

            try:
                # 计算技术指标
                if self.technical_analyzer:
                    # 这里可以添加技术分析逻辑
                    pass

                # 简单示例：基于移动平均线生成信号
                latest = stock_data.iloc[-1]
                ma5 = stock_data['Close'].rolling(5).mean().iloc[-1]
                ma20 = stock_data['Close'].rolling(20).mean().iloc[-1]

                # 买入信号：短期均线上穿长期均线
                if ma5 > ma20:
                    signals.append({
                        'code': stock_code,
                        'name': stock_name,
                        'signal': 'BUY',
                        'price': latest['Close'],
                        'ma5': ma5,
                        'ma20': ma20,
                        'reason': '短期均线上穿长期均线'
                    })

                # 卖出信号：短期均线下穿长期均线
                elif ma5 < ma20:
                    signals.append({
                        'code': stock_code,
                        'name': stock_name,
                        'signal': 'SELL',
                        'price': latest['Close'],
                        'ma5': ma5,
                        'ma20': ma20,
                        'reason': '短期均线下穿长期均线'
                    })

            except Exception as e:
                print(f"⚠️ 生成 {stock_code} 信号失败: {e}")
                continue

        return signals

    def generate_report_content(self, signals):
        """生成报告内容"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 获取指数数据
        sse_data = self.cache.get('sse_data')
        szse_data = self.cache.get('szse_data')

        sse_close = sse_data['Close'].iloc[-1] if sse_data is not None else 0
        sse_change = ((sse_data['Close'].iloc[-1] - sse_data['Close'].iloc[-2]) / sse_data['Close'].iloc[-2] * 100) if sse_data is not None and len(sse_data) > 1 else 0

        szse_close = szse_data['Close'].iloc[-1] if szse_data is not None else 0
        szse_change = ((szse_data['Close'].iloc[-1] - szse_data['Close'].iloc[-2]) / szse_data['Close'].iloc[-2] * 100) if szse_data is not None and len(szse_data) > 1 else 0

        # 获取预测数据
        from ml_services.sse_prediction import SSE_Predictor
        from ml_services.szse_prediction import SZSE_Predictor

        # 运行预测
        sse_predictor = SSE_Predictor()
        sse_predictor.fetch_data()
        sse_predictor.calculate_features()
        sse_score, _ = sse_predictor.calculate_prediction_score()
        sse_trend = sse_predictor.interpret_score(sse_score)[0]

        szse_predictor = SZSE_Predictor()
        szse_predictor.fetch_data()
        szse_predictor.calculate_features()
        szse_score, _ = szse_predictor.calculate_prediction_score()
        szse_trend = szse_predictor.interpret_score(szse_score)[0]

        # 综合判断
        avg_score = (sse_score + szse_score) / 2
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

        # 生成HTML报告
        content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股市场分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a73e8;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 10px;
        }}
        .market-summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .index {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .index-name {{
            font-weight: bold;
            color: #1a73e8;
        }}
        .index-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .index-change {{
            font-size: 18px;
        }}
        .up {{
            color: #e53935;
        }}
        .down {{
            color: #43a047;
        }}
        .signals-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .signals-table th,
        .signals-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .signals-table th {{
            background-color: #1a73e8;
            color: #fff;
        }}
        .signals-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .buy {{
            color: #e53935;
            font-weight: bold;
        }}
        .sell {{
            color: #43a047;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🇨🇳 A股市场分析报告</h1>

        <div class="market-summary">
            <h2>📊 市场概况</h2>
            <div class="index">
                <div class="index-name">上证指数</div>
                <div class="index-value">{sse_close:.2f}</div>
                <div class="index-change {'up' if sse_change > 0 else 'down'}">
                    {sse_change:+.2f}%
                </div>
            </div>
            <div class="index">
                <div class="index-name">深证成指</div>
                <div class="index-value">{szse_close:.2f}</div>
                <div class="index-change {'up' if szse_change > 0 else 'down'}">
                    {szse_change:+.2f}%
                </div>
            </div>
        </div>

        <div class="market-summary" style="background-color: #e3f2fd; border-left: 4px solid #1a73e8;">
            <h2 style="margin-top: 0;">📈 市场预测</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="background-color: #bbdefb;">
                    <th style="padding: 10px; text-align: left;">指数</th>
                    <th style="padding: 10px; text-align: left;">预测得分</th>
                    <th style="padding: 10px; text-align: left;">预测趋势</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">上证指数 (000001)</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{sse_score:.4f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: #e53935; font-weight: bold;">{sse_trend}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">深证成指 (399001)</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{szse_score:.4f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: #e53935; font-weight: bold;">{szse_trend}</td>
                </tr>
                <tr style="background-color: #bbdefb;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">综合判断</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{avg_score:.4f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-size: 18px; font-weight: bold;">{overall_trend} {trend_emoji}</td>
                </tr>
            </table>
        </div>

        <h2>🎯 交易信号</h2>
"""

        if signals:
            content += """
        <table class="signals-table">
            <thead>
                <tr>
                    <th>股票代码</th>
                    <th>股票名称</th>
                    <th>信号</th>
                    <th>当前价格</th>
                    <th>理由</th>
                </tr>
            </thead>
            <tbody>
"""

            for signal in signals:
                signal_class = 'buy' if signal['signal'] == 'BUY' else 'sell'
                signal_text = '买入' if signal['signal'] == 'BUY' else '卖出'

                content += f"""
                <tr>
                    <td>{signal['code']}</td>
                    <td>{signal['name']}</td>
                    <td class="{signal_class}">{signal_text}</td>
                    <td>{signal['price']:.2f}</td>
                    <td>{signal['reason']}</td>
                </tr>
"""

            content += """
            </tbody>
        </table>
"""
        else:
            content += """
        <p>📌 当前无交易信号</p>
"""

        content += f"""
        <div class="footer">
            <p>生成时间: {now}</p>
            <p>本报告仅供参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>
"""

        return content

    def send_email(self, content):
        """发送邮件"""
        if not all([self.smtp_username, self.smtp_password, self.recipient_email]):
            print("❌ 邮件配置不完整，跳过邮件发送")
            print("请设置环境变量或参数: smtp_username, smtp_password, recipient_email")
            return False

        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"A股市场分析报告 - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = self.smtp_username
            msg['To'] = self.recipient_email

            # 添加HTML内容
            html_part = MIMEText(content, 'html', 'utf-8')
            msg.attach(html_part)

            # 连接SMTP服务器
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"✅ 邮件已发送至 {self.recipient_email}")
            return True

        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")
            return False

    def run(self, send_email=False):
        """运行分析"""
        print("=" * 80)
        print("A股市场邮件通知系统")
        print("=" * 80)

        # 获取市场数据
        self.get_market_data()

        # 生成交易信号
        signals = self.generate_trading_signals()

        # 生成报告
        content = self.generate_report_content(signals)

        # 打印信号
        if signals:
            print(f"\n🎯 发现 {len(signals)} 个交易信号:")
            for signal in signals:
                print(f"  - {signal['code']} {signal['name']}: {signal['signal']} @ {signal['price']:.2f}")
        else:
            print("\n📌 当前无交易信号")

        # 发送邮件
        if send_email:
            self.send_email(content)

        print("\n" + "=" * 80)
        print("✅ 分析完成")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='A股市场邮件通知系统')
    parser.add_argument('--send-email', action='store_true', help='发送邮件')
    parser.add_argument('--investment-style', type=str, default='short_term',
                        choices=['ultra_short_term', 'short_term', 'medium_long_term'],
                        help='投资风格')
    parser.add_argument('--smtp-server', type=str, default='smtp.qq.com', help='SMTP服务器')
    parser.add_argument('--smtp-port', type=int, default=465, help='SMTP端口')
    parser.add_argument('--smtp-username', type=str, help='SMTP用户名')
    parser.add_argument('--smtp-password', type=str, help='SMTP密码')
    parser.add_argument('--recipient-email', type=str, help='收件人邮箱')

    args = parser.parse_args()

    # 从环境变量读取配置
    smtp_username = args.smtp_username or os.getenv('EMAIL_ADDRESS')
    smtp_password = args.smtp_password or os.getenv('EMAIL_AUTHCODE')
    recipient_email = args.recipient_email or os.getenv('RECIPIENT_EMAIL')

    # 创建邮件系统
    system = AShareEmailSystem(
        investment_style=args.investment_style,
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        recipient_email=recipient_email
    )

    # 运行分析
    system.run(send_email=args.send_email)


if __name__ == '__main__':
    main()