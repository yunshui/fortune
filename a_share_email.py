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

# 导入A股预测器
try:
    from ml_services.sse_prediction import SSE_Predictor
    from ml_services.szse_prediction import SZSE_Predictor
except ImportError:
    print("⚠️ A股预测模块不可用")


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
        """生成交易信号（包含详细技术分析）"""
        print("🎯 生成交易信号...")

        signals = []

        for stock_code in self.stock_list:
            stock_name = STOCK_LIST.get(stock_code, stock_code)
            stock_data = self.cache['stock_data'].get(stock_code)

            if stock_data is None or len(stock_data) < 20:
                continue

            try:
                # 计算技术指标
                df = stock_data.copy()

                # 移动平均线
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()

                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # MACD
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

                # KDJ
                low_9 = df['Low'].rolling(window=9).min()
                high_9 = df['High'].rolling(window=9).max()
                rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
                df['K'] = rsv.ewm(com=2, adjust=False).mean()
                df['D'] = df['K'].ewm(com=2, adjust=False).mean()
                df['J'] = 3 * df['K'] - 2 * df['D']

                # 布林带
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                df['BB_Std'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
                df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

                # ATR
                df['High_Low'] = df['High'] - df['Low']
                df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
                df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
                df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
                df['ATR'] = df['TR'].rolling(window=14).mean()

                # 支撑阻力
                df['Support_20d'] = df['Low'].rolling(window=20).min()
                df['Resistance_20d'] = df['High'].rolling(window=20).max()

                latest = df.iloc[-1]
                ma5 = latest['MA5']
                ma10 = latest['MA10']
                ma20 = latest['MA20']
                ma60 = latest['MA60']
                rsi = latest['RSI']
                macd = latest['MACD']
                macd_signal = latest['MACD_Signal']
                k = latest['K']
                d = latest['D']
                j = latest['J']
                bb_position = latest['BB_Position']
                bb_lower = latest['BB_Lower']
                bb_upper = latest['BB_Upper']
                atr = latest['ATR']
                support_20d = latest['Support_20d']
                resistance_20d = latest['Resistance_20d']

                # 判断信号和详细理由
                signal = 'HOLD'
                reason = ''
                tech_analysis = []

                # 均线分析
                if ma5 > ma10 > ma20:
                    signal = 'BUY'
                    reason = '短期均线多头排列，趋势向上'
                    tech_analysis.append('MA5>MA10>MA20多头排列')
                elif ma5 < ma10 < ma20:
                    signal = 'SELL'
                    reason = '短期均线空头排列，趋势向下'
                    tech_analysis.append('MA5<MA10<MA20空头排列')
                elif ma5 > ma20 and ma10 < ma20:
                    signal = 'BUY'
                    reason = '短期均线拐头向上'
                    tech_analysis.append('MA5上穿MA20')
                elif ma5 < ma20 and ma10 > ma20:
                    signal = 'SELL'
                    reason = '短期均线拐头向下'
                    tech_analysis.append('MA5下穿MA20')

                # RSI分析
                if pd.notna(rsi):
                    if rsi < 30:
                        if signal != 'SELL':
                            tech_analysis.append(f'RSI={rsi:.1f}超卖，反弹信号')
                    elif rsi > 70:
                        if signal != 'BUY':
                            tech_analysis.append(f'RSI={rsi:.1f}超买，回调风险')
                    else:
                        tech_analysis.append(f'RSI={rsi:.1f}正常')

                # MACD分析
                if pd.notna(macd) and pd.notna(macd_signal):
                    if macd > macd_signal:
                        tech_analysis.append(f'MACD金叉({macd:.2f}>({macd_signal:.2f})')
                    else:
                        tech_analysis.append(f'MACD死叉({macd:.2f}<{macd_signal:.2f})')

                # KDJ分析
                if pd.notna(k) and pd.notna(d):
                    if k > d:
                        tech_analysis.append(f'KDJ金叉(K={k:.1f}>D={d:.1f})')
                        if k > 80:
                            tech_analysis.append('KDJ超买')
                    else:
                        tech_analysis.append(f'KDJ死叉(K={k:.1f}<D={d:.1f})')
                        if k < 20:
                            tech_analysis.append('KDJ超卖')

                # 布林带分析
                if pd.notna(bb_position):
                    if bb_position > 0.8:
                        tech_analysis.append(f'接近布林带上轨({bb_position*100:.0f}%)')
                    elif bb_position < 0.2:
                        tech_analysis.append(f'接近布林带下轨({bb_position*100:.0f}%)')
                    else:
                        tech_analysis.append(f'布林带中位({bb_position*100:.0f}%)')

                # ATR波动分析
                if pd.notna(atr) and pd.notna(latest['Close']):
                    atr_pct = (atr / latest['Close']) * 100
                    tech_analysis.append(f'ATR波动率{atr_pct:.2f}%')

                # 支撑阻力分析
                if pd.notna(support_20d) and pd.notna(resistance_20d):
                    support_dist = ((latest['Close'] - support_20d) / support_20d) * 100
                    resistance_dist = ((resistance_20d - latest['Close']) / latest['Close']) * 100
                    tech_analysis.append(f'支撑{support_20d:.2f}({support_dist:+.1f}%)')
                    tech_analysis.append(f'阻力{resistance_20d:.2f}({resistance_dist:+.1f}%)')

                # 成交量分析
                volume = latest['Volume']
                vol_str = f"{volume/100000000:.1f}亿" if volume > 0 else "N/A"

                signals.append({
                    'code': stock_code,
                    'name': stock_name,
                    'signal': signal,
                    'price': latest['Close'],
                    'reason': reason,
                    'tech_analysis': tech_analysis,
                    'ma5': ma5,
                    'ma10': ma10,
                    'ma20': ma20,
                    'ma60': ma60,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'k': k,
                    'd': d,
                    'j': j,
                    'bb_position': bb_position,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'atr': atr,
                    'support_20d': support_20d,
                    'resistance_20d': resistance_20d,
                    'volume': volume
                })

            except Exception as e:
                print(f"⚠️ 生成 {stock_code} 信号失败: {e}")
                continue

        return signals

    def generate_report_content(self, signals):
        """生成报告内容（包含详细技术指标）"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 获取指数数据
        sse_data = self.cache.get('sse_data')
        szse_data = self.cache.get('szse_data')

        sse_close = sse_data['Close'].iloc[-1] if sse_data is not None else 0
        sse_change = ((sse_data['Close'].iloc[-1] - sse_data['Close'].iloc[-2]) / sse_data['Close'].iloc[-2] * 100) if sse_data is not None and len(sse_data) > 1 else 0

        szse_close = szse_data['Close'].iloc[-1] if szse_data is not None else 0
        szse_change = ((szse_data['Close'].iloc[-1] - szse_data['Close'].iloc[-2]) / szse_data['Close'].iloc[-2] * 100) if szse_data is not None and len(szse_data) > 1 else 0

        # 计算指数技术指标
        sse_predictor = SSE_Predictor()
        sse_predictor.fetch_data()
        sse_df = sse_predictor.calculate_technical_indicators(sse_predictor.sse_data)
        sse_latest = sse_df.iloc[-1]

        szse_predictor = SZSE_Predictor()
        szse_predictor.fetch_data()
        szse_df = szse_predictor.calculate_technical_indicators(szse_predictor.szse_data)
        szse_latest = szse_df.iloc[-1]

        # 运行预测
        sse_predictor.calculate_features()
        sse_score, _ = sse_predictor.calculate_prediction_score()
        sse_trend = sse_predictor.interpret_score(sse_score)[0]

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
            max-width: 900px;
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
        .stock-card {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .stock-header {{
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .stock-name {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .stock-price {{
            font-size: 20px;
            font-weight: bold;
        }}
        .tech-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 12px;
        }}
        .tech-table th,
        .tech-table td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }}
        .tech-table th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        .tech-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .buy {{
            color: #e53935;
            font-weight: bold;
        }}
        .sell {{
            color: #43a047;
            font-weight: bold;
        }}
        .analysis-points {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        .analysis-points li {{
            margin: 3px 0;
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
        <h3 style="text-align: center; color: #666; font-weight: normal; margin-top: -20px; margin-bottom: 20px;">当前预测日期: {now}</h3>

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
            <h2 style="margin-top: 0;">📊 市场技术指标</h2>
            <table class="tech-table" style="width: 100%; margin-top: 10px;">
                <tr style="background-color: #bbdefb;">
                    <th style="padding: 10px; text-align: left;">指标</th>
                    <th style="padding: 10px; text-align: left;">上证指数</th>
                    <th style="padding: 10px; text-align: left;">深证成指</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">RSI</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('RSI', 0):.1f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('RSI', 0):.1f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">MACD</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('MACD', 0):.2f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('MACD', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">KDJ-K</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('K', 0):.1f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('K', 0):.1f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">布林带位置</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('BB_Position', 0)*100:.1f}%</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('BB_Position', 0)*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">支撑位</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('Support_20d', 0):.2f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('Support_20d', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">阻力位</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{sse_latest.get('Resistance_20d', 0):.2f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{szse_latest.get('Resistance_20d', 0):.2f}</td>
                </tr>
            </table>
        </div>

        <div class="market-summary" style="background-color: #fff3e0; border-left: 4px solid #ff9800;">
            <h2 style="margin-top: 0;">📈 市场预测</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="background-color: #ffcc80;">
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
                <tr style="background-color: #ffcc80;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">综合判断</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{avg_score:.4f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-size: 18px; font-weight: bold;">{overall_trend} {trend_emoji}</td>
                </tr>
            </table>
        </div>
"""

        # 分离买入和卖出信号
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']

        # 排序：买入按价格从低到高，卖出按价格从高到低
        buy_signals_sorted = sorted(buy_signals, key=lambda x: x['price'])
        sell_signals_sorted = sorted(sell_signals, key=lambda x: -x['price'])

        if buy_signals:
            content += f"""
        <div class="market-summary" style="background-color: #e8f5e9; border-left: 4px solid #43a047;">
            <h2 style="margin-top: 0; color: #2e7d32;">🟢 强烈推荐买入 ({len(buy_signals)}只)</h2>
"""

            for signal in buy_signals_sorted:
                rsi_str = f"{signal['rsi']:.1f}" if pd.notna(signal['rsi']) else 'N/A'
                macd_str = f"{signal['macd']:.2f}" if pd.notna(signal['macd']) else 'N/A'
                bb_pos_str = f"{signal['bb_position']*100:.1f}%" if pd.notna(signal['bb_position']) else 'N/A'

                content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <span class="stock-name">{signal['code']} {signal['name']}</span>
                    <span class="stock-price buy">¥{signal['price']:.2f}</span>
                </div>
                <div style="font-size: 14px; margin-bottom: 10px;">
                    <strong>买入理由:</strong> {signal['reason']}
                </div>
                <table class="tech-table">
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>信号</th>
                    </tr>
                    <tr>
                        <td>MA5/MA20</td>
                        <td>{signal['ma5']:.2f} / {signal['ma20']:.2f}</td>
                        <td class="buy">多头</td>
                    </tr>
                    <tr>
                        <td>RSI</td>
                        <td>{rsi_str}</td>
                        <td>{'正常' if 30 <= signal['rsi'] <= 70 else '超买' if signal['rsi'] > 70 else '超卖' if pd.notna(signal['rsi']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>MACD</td>
                        <td>{macd_str}</td>
                        <td>{'金叉' if signal['macd'] > signal['macd_signal'] else '死叉' if pd.notna(signal['macd']) and pd.notna(signal['macd_signal']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>布林带位置</td>
                        <td>{bb_pos_str}</td>
                        <td>{'高位' if signal['bb_position'] > 0.8 else '低位' if signal['bb_position'] < 0.2 else '中位' if pd.notna(signal['bb_position']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>支撑位</td>
                        <td>{signal['support_20d']:.2f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>阻力位</td>
                        <td>{signal['resistance_20d']:.2f}</td>
                        <td>-</td>
                    </tr>
                </table>
                <div class="analysis-points">
                    <strong>技术分析:</strong>
                    <ul>
"""
                for point in signal['tech_analysis'][:5]:  # 显示前5个要点
                    content += f"<li>{point}</li>"
                content += """
                    </ul>
                </div>
            </div>
"""

            content += """
        </div>
"""

        if sell_signals:
            content += f"""
        <div class="market-summary" style="background-color: #ffebee; border-left: 4px solid #e53935;">
            <h2 style="margin-top: 0; color: #c62828;">🔴 推荐卖出 ({len(sell_signals)}只)</h2>
"""

            for signal in sell_signals_sorted:
                rsi_str = f"{signal['rsi']:.1f}" if pd.notna(signal['rsi']) else 'N/A'
                macd_str = f"{signal['macd']:.2f}" if pd.notna(signal['macd']) else 'N/A'
                bb_pos_str = f"{signal['bb_position']*100:.1f}%" if pd.notna(signal['bb_position']) else 'N/A'

                content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <span class="stock-name">{signal['code']} {signal['name']}</span>
                    <span class="stock-price sell">¥{signal['price']:.2f}</span>
                </div>
                <div style="font-size: 14px; margin-bottom: 10px;">
                    <strong>卖出理由:</strong> {signal['reason']}
                </div>
                <table class="tech-table">
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>信号</th>
                    </tr>
                    <tr>
                        <td>MA5/MA20</td>
                        <td>{signal['ma5']:.2f} / {signal['ma20']:.2f}</td>
                        <td class="sell">空头</td>
                    </tr>
                    <tr>
                        <td>RSI</td>
                        <td>{rsi_str}</td>
                        <td>{'正常' if 30 <= signal['rsi'] <= 70 else '超买' if signal['rsi'] > 70 else '超卖' if pd.notna(signal['rsi']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>MACD</td>
                        <td>{macd_str}</td>
                        <td>{'金叉' if signal['macd'] > signal['macd_signal'] else '死叉' if pd.notna(signal['macd']) and pd.notna(signal['macd_signal']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>布林带位置</td>
                        <td>{bb_pos_str}</td>
                        <td>{'高位' if signal['bb_position'] > 0.8 else '低位' if signal['bb_position'] < 0.2 else '中位' if pd.notna(signal['bb_position']) else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>支撑位</td>
                        <td>{signal['support_20d']:.2f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>阻力位</td>
                        <td>{signal['resistance_20d']:.2f}</td>
                        <td>-</td>
                    </tr>
                </table>
                <div class="analysis-points">
                    <strong>技术分析:</strong>
                    <ul>
"""
                for point in signal['tech_analysis'][:5]:  # 显示前5个要点
                    content += f"<li>{point}</li>"
                content += """
                    </ul>
                </div>
            </div>
"""

            content += """
        </div>
"""

        if not signals:
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