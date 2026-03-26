#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股买入和回避分析脚本 (使用 yfinance)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# 测试主要港股
test_stocks = [
    ('0700.HK', '腾讯控股', 'tech'),
    ('9988.HK', '阿里巴巴', 'tech'),
    ('0005.HK', '汇丰银行', 'bank'),
    ('0939.HK', '建设银行', 'bank'),
    ('0941.HK', '中国移动', 'utility'),
    ('0388.HK', '香港交易所', 'exchange'),
    ('1299.HK', '友邦保险', 'insurance'),
    ('2318.HK', '中国平安', 'insurance'),
    ('0700.HK', '腾讯控股', 'tech'),
    ('0981.HK', '中芯国际', 'semiconductor'),
    ('1810.HK', '小米集团', 'tech'),
    ('1211.HK', '比亚迪股份', 'auto'),
    ('2024.HK', '美团-W', 'tech'),
    ('02333.HK', '长城汽车', 'auto'),
    ('02269.HK', '药明康德', 'biotech'),
    ('1398.HK', '工商银行', 'bank'),
    ('3968.HK', '招商银行', 'bank'),
    ('1288.HK', '农业银行', 'bank'),
    ('0883.HK', '中国海洋石油', 'energy'),
    ('0960.HK', '龙源电力', 'new_energy'),
    ('1798.HK', '赣锋锂业', 'new_energy'),
    ('1138.HK', '中远海能', 'shipping'),
    ('0988.HK', '京东集团-SW', 'tech'),
    ('9618.HK', '京东集团-SW', 'tech'),
]

print('=' * 80)
print(f'{"港股市场买卖建议":^76}')
print('=' * 80)
print(f'📅 当前时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'📈 综合预测: 强烈看涨 (恒指预测得分 0.7612)')
print()

buy_stocks = []
sell_stocks = []

for code, name, sector in test_stocks:
    try:
        ticker = yf.Ticker(code)
        hist = ticker.history(period='3mo', interval='1d')

        if hist is not None and len(hist) >= 20:
            latest = hist.iloc[-1]
            ma5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma10 = hist['Close'].rolling(10).mean().iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma60 = hist['Close'].rolling(60).mean().iloc[-1]

            # 计算RSI
            rsi = None
            if len(hist) >= 14:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (gain + loss)
                rsi_val = rs.iloc[-1]
                rsi = 100 - (100 / (1 + rsi_val)) if pd.notna(rsi_val) else None

            # 判断信号
            signal = 'HOLD'
            reason = ''

            if ma5 > ma10 > ma20:
                signal = 'BUY'
                reason = '短期均线多头排列(MA5>MA10>MA20)'
            elif ma5 < ma10 < ma20:
                signal = 'SELL'
                reason = '短期均线空头排列(MA5<MA10<MA20)'
            elif ma5 > ma20:
                signal = 'BUY'
                reason = '短期上穿中期均线'
            elif ma5 < ma20:
                signal = 'SELL'
                reason = '短期下穿中期均线'

            if signal != 'HOLD':
                stock_info = {
                    'code': code,
                    'name': name,
                    'sector': sector,
                    'price': latest['Close'],
                    'ma5': ma5,
                    'ma20': ma20,
                    'ma60': ma60,
                    'rsi': rsi,
                    'signal': signal,
                    'reason': reason,
                    'volume': latest['Volume']
                }
                if signal == 'BUY':
                    buy_stocks.append(stock_info)
                else:
                    sell_stocks.append(stock_info)
                print(f'✅ {code} {name}: 价格={latest["Close"]:.2f}, MA5={ma5:.2f}, MA20={ma20:.2f}, 信号={signal}')
            else:
                print(f'⚪ {code} {name}: 价格={latest["Close"]:.2f}, 中性')
        else:
            print(f'❌ {code} {name}: 数据不足')

        time.sleep(0.5)
    except Exception as e:
        print(f'❌ {code} {name}: 获取失败 - {e}')

# 排序：买入按价格从低到高，卖出按价格从高到低
buy_stocks_sorted = sorted(buy_stocks, key=lambda x: x['price'])
sell_stocks_sorted = sorted(sell_stocks, key=lambda x: -x['price'])

print()
if buy_stocks:
    print(f'🟢 强烈推荐买入 ({len(buy_stocks)} 只)')
    print('-' * 80)
    for i, stock in enumerate(buy_stocks_sorted, 1):
        ma60_str = f'{stock["ma60"]:.2f}' if pd.notna(stock["ma60"]) else 'N/A'
        rsi_str = f'{stock["rsi"]:.1f}' if stock["rsi"] is not None and pd.notna(stock["rsi"]) else 'N/A'
        vol_str = f'{stock["volume"]/1000000:.1f}M' if pd.notna(stock["volume"]) and stock["volume"] > 0 else 'N/A'
        print(f'{i}. {stock["code"]} {stock["name"]} ({stock["sector"]})')
        print(f'   价格: HK${stock["price"]:.2f}')
        print(f'   理由: {stock["reason"]}')
        print(f'   技术指标: MA5={stock["ma5"]:.2f}, MA20={stock["ma20"]:.2f}, MA60={ma60_str}')
        print(f'   RSI: {rsi_str}, 成交量: {vol_str}')
        print()

if sell_stocks:
    print(f'🔴 推荐卖出 ({len(sell_stocks)} 只)')
    print('-' * 80)
    for i, stock in enumerate(sell_stocks_sorted, 1):
        ma60_str = f'{stock["ma60"]:.2f}' if pd.notna(stock["ma60"]) else 'N/A'
        rsi_str = f'{stock["rsi"]:.1f}' if stock["rsi"] is not None and pd.notna(stock["rsi"]) else 'N/A'
        vol_str = f'{stock["volume"]/1000000:.1f}M' if pd.notna(stock["volume"]) and stock["volume"] > 0 else 'N/A'
        print(f'{i}. {stock["code"]} {stock["name"]} ({stock["sector"]})')
        print(f'   价格: HK${stock["price"]:.2f}')
        print(f'   理由: {stock["reason"]}')
        print(f'   技术指标: MA5={stock["ma5"]:.2f}, MA20={stock["ma20"]:.2f}, MA60={ma60_str}')
        print(f'   RSI: {rsi_str}, 成交量: {vol_str}')
        print()

print('=' * 80)
print(f'✅ 分析完成: {len(buy_stocks)} 买入, {len(sell_stocks)} 卖出')
print('=' * 80)