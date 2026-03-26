#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深证成指涨跌预测脚本

基于模型特征重要性，使用加权评分模型预测深证成指短期走势
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录并添加到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

data_dir = os.path.join(project_root, 'data')
output_dir = os.path.join(project_root, 'output')

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 导入A股数据获取模块
from data_services.a_share_finance import get_szse_index_data


class SZSE_Predictor:
    """深证成指预测器"""

    # 特征重要性配置（权重、影响方向）
    # 参考HSI预测器的特征配置，适用于A股市场
    FEATURE_IMPORTANCE = {
        # 长期移动平均线相关（权重最高）
        'MA250': {'weight': 0.1500, 'direction': 1},  # 250日均线，长期趋势支撑
        'Volume_MA250': {'weight': 0.1200, 'direction': 1},  # 250日成交量均线，长期流动性
        'MA120': {'weight': 0.1000, 'direction': 1},  # 120日均线，中期趋势

        # 多周期相对强度信号（RS_Signal）
        '60d_RS_Signal_MA250': {'weight': 0.0800, 'direction': 1},  # 60日相对强度信号
        '60d_RS_Signal_Volume_MA250': {'weight': 0.0600, 'direction': 1},  # 成交量相对强度
        '20d_RS_Signal_MA250': {'weight': 0.0400, 'direction': 1},  # 20日相对强度
        '10d_RS_Signal_MA250': {'weight': 0.0350, 'direction': 1},  # 10日相对强度
        '5d_RS_Signal_MA250': {'weight': 0.0300, 'direction': 1},  # 5日相对强度
        '3d_RS_Signal_MA250': {'weight': 0.0250, 'direction': 1},  # 3日相对强度

        # 多周期趋势（Trend）
        '60d_Trend_MA250': {'weight': 0.0500, 'direction': 1},  # 60日趋势
        '20d_Trend_MA250': {'weight': 0.0450, 'direction': 1},  # 20日趋势
        '10d_Trend_MA250': {'weight': 0.0400, 'direction': 1},  # 10日趋势
        '5d_Trend_MA250': {'weight': 0.0350, 'direction': 1},  # 5日趋势
        '3d_Trend_MA250': {'weight': 0.0300, 'direction': 1},  # 3日趋势

        # 成交量趋势
        '60d_Trend_Volume_MA250': {'weight': 0.0450, 'direction': 1},  # 60日成交量趋势
        '20d_Trend_Volume_MA250': {'weight': 0.0350, 'direction': 1},  # 20日成交量趋势

        # 波动率
        'Volatility_120d': {'weight': 0.0400, 'direction': -1},  # 120日波动率，高波动率不利

        # 中期均线趋势
        '60d_Trend_MA120': {'weight': 0.0350, 'direction': 1},  # 60日MA120趋势

        # 成交量相对强度
        '20d_RS_Signal_Volume_MA250': {'weight': 0.0300, 'direction': 1},  # 20日成交量相对强度
    }

    def __init__(self):
        self.szse_data = None
        self.features = {}

    def fetch_data(self):
        """获取所需数据"""
        print("📊 正在获取数据...")

        # 获取深证成指数据（2年数据以确保MA250等长期指标有足够数据）
        print("  - 深证成指数据...")
        self.szse_data = get_szse_index_data(period_days=730)

        if self.szse_data is None or self.szse_data.empty:
            raise ValueError("深证成指数据获取失败")

        print(f"  ✅ 数据获取完成（深成指：{len(self.szse_data)} 条）")

    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()

        # 移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        df['MA250'] = df['Close'].rolling(window=250).mean()

        # MA250斜率（趋势强度）
        df['MA250_Slope'] = df['MA250'].diff()

        # RSI (相对强弱指标)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (指数平滑异同移动平均线)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # KDJ (随机指标)
        low_9 = df['Low'].rolling(window=9).min()
        high_9 = df['High'].rolling(window=9).max()
        rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # 布林带 (Bollinger Bands)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # 收益率
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_20d'] = df['Close'].pct_change(20)
        df['Return_60d'] = df['Close'].pct_change(60)

        # 成交量相关
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA250'] = df['Volume'].rolling(window=250).mean()
        df['Turnover_Std_20'] = df['Volume'].rolling(window=20).std()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA250']

        # OBV（能量潮指标）
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # ATR（平均真实波幅）
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_MA'] = df['ATR'].rolling(window=20).mean()
        df['ATR_MA120'] = df['ATR'].rolling(window=120).mean()

        # 波动率
        df['Volatility'] = df['Return_1d'].rolling(window=20).std()
        df['Vol_Std_20'] = df['Volatility'].rolling(window=20).std()

        # VWAP（成交量加权平均价）
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

        # 支撑阻力位
        df['Resistance_120d'] = df['High'].rolling(window=120).max()
        df['Support_120d'] = df['Low'].rolling(window=120).min()
        df['Distance_Support_120d'] = (df['Close'] - df['Support_120d']) / df['Support_120d']

        # 相对强弱信号
        df['RS_Signal_MA250_Slope'] = df['Close'] / df['MA250'] - 1

        return df

    def calculate_features(self):
        """计算所有特征"""
        print("🔧 正在计算特征...")

        szse_df = self.calculate_technical_indicators(self.szse_data)

        # 计算多周期指标
        periods = [3, 5, 10, 20, 60]

        # 计算多周期收益率
        for period in periods:
            if len(szse_df) >= period:
                return_col = f'Return_{period}d'
                szse_df[return_col] = szse_df['Close'].pct_change(period)

                # 计算趋势方向（1=上涨，0=下跌）
                trend_col = f'{period}d_Trend'
                szse_df[trend_col] = (szse_df[return_col] > 0).astype(int)

                # 计算相对强度信号（基于收益率和MA250）
                rs_signal_col = f'{period}d_RS_Signal'
                szse_df[rs_signal_col] = (szse_df[return_col] > 0).astype(int)

        # 计算多周期MA250趋势
        for period in periods:
            if len(szse_df) >= period:
                trend_col = f'{period}d_Trend_MA250'
                szse_df[trend_col] = (szse_df['MA250'].diff(period) > 0).astype(int)

                # 计算MA250的相对强度信号
                rs_signal_col = f'{period}d_RS_Signal_MA250'
                szse_df[rs_signal_col] = (szse_df['Close'] > szse_df['MA250']).astype(int)

        # 计算多周期Volume_MA250趋势
        for period in periods:
            if len(szse_df) >= period:
                trend_col = f'{period}d_Trend_Volume_MA250'
                szse_df[trend_col] = (szse_df['Volume_MA250'].diff(period) > 0).astype(int)

                # 计算Volume_MA250的相对强度信号
                rs_signal_col = f'{period}d_RS_Signal_Volume_MA250'
                szse_df[rs_signal_col] = (szse_df['Volume'] > szse_df['Volume_MA250']).astype(int)

        # 计算多周期MA120趋势
        for period in periods:
            if len(szse_df) >= period:
                trend_col = f'{period}d_Trend_MA120'
                szse_df[trend_col] = (szse_df['MA120'].diff(period) > 0).astype(int)

        # 计算120日波动率
        szse_df['Volatility_120d'] = szse_df['Return_1d'].rolling(window=120).std()

        # 获取最新数据（最近一天）
        latest_szse = szse_df.iloc[-1]

        # 安全获取特征值，处理NaN情况
        def safe_get(series, default=0):
            if pd.isna(series):
                return default
            return series

        # 计算特征值
        self.features = {
            # 长期移动平均线相关
            'MA250': safe_get(latest_szse.get('MA250', latest_szse['Close'])),
            'Volume_MA250': safe_get(latest_szse.get('Volume_MA250', latest_szse['Volume'])),
            'MA120': safe_get(latest_szse.get('MA120', latest_szse['Close'])),

            # 多周期相对强度信号（RS_Signal）
            '60d_RS_Signal_MA250': safe_get(latest_szse.get('60d_RS_Signal_MA250', 0), 0),
            '60d_RS_Signal_Volume_MA250': safe_get(latest_szse.get('60d_RS_Signal_Volume_MA250', 0), 0),
            '20d_RS_Signal_MA250': safe_get(latest_szse.get('20d_RS_Signal_MA250', 0), 0),
            '10d_RS_Signal_MA250': safe_get(latest_szse.get('10d_RS_Signal_MA250', 0), 0),
            '5d_RS_Signal_MA250': safe_get(latest_szse.get('5d_RS_Signal_MA250', 0), 0),
            '3d_RS_Signal_MA250': safe_get(latest_szse.get('3d_RS_Signal_MA250', 0), 0),

            # 多周期趋势（Trend）
            '60d_Trend_MA250': safe_get(latest_szse.get('60d_Trend_MA250', 0), 0),
            '20d_Trend_MA250': safe_get(latest_szse.get('20d_Trend_MA250', 0), 0),
            '10d_Trend_MA250': safe_get(latest_szse.get('10d_Trend_MA250', 0), 0),
            '5d_Trend_MA250': safe_get(latest_szse.get('5d_Trend_MA250', 0), 0),
            '3d_Trend_MA250': safe_get(latest_szse.get('3d_Trend_MA250', 0), 0),

            # 成交量趋势
            '60d_Trend_Volume_MA250': safe_get(latest_szse.get('60d_Trend_Volume_MA250', 0), 0),
            '20d_Trend_Volume_MA250': safe_get(latest_szse.get('20d_Trend_Volume_MA250', 0), 0),

            # 波动率
            'Volatility_120d': safe_get(latest_szse.get('Volatility_120d', 0), 0),

            # 中期均线趋势
            '60d_Trend_MA120': safe_get(latest_szse.get('60d_Trend_MA120', 0), 0),

            # 成交量相对强度
            '20d_RS_Signal_Volume_MA250': safe_get(latest_szse.get('20d_RS_Signal_Volume_MA250', 0), 0),
        }

        print(f"  ✅ 特征计算完成（{len(self.features)} 个特征）")

    def normalize_feature(self, feature_name, value):
        """特征标准化（与港股HSI预测系统保持一致）"""
        # RS_Signal和Trend特征通常是0-1的二元值，直接映射到[-1, 1]
        if 'RS_Signal' in feature_name or 'Trend' in feature_name:
            # 将0-1映射到[-1, 1]：0 -> -1, 1 -> 1
            return value * 2 - 1

        # 如果是收益率类特征，使用固定范围标准化
        elif 'Return' in feature_name or 'Yield' in feature_name:
            # 标准化到[-1, 1]区间，假设收益率在[-0.2, 0.2]范围内
            return np.clip(value / 0.2, -1, 1)

        # MA相关特征，使用与港股一致的标准化参数
        elif 'MA' in feature_name:
            if pd.isna(value):
                return 0
            # 与港股HSI保持一致的标准化参数
            return np.tanh(value / 50000)  # 假设MA值在50000左右

        # 波动率特征 - 与港股HSI保持一致
        elif 'Volatility' in feature_name:
            # 波动率通常在0.01-0.05之间
            if pd.isna(value):
                return 0
            # 与港股HSI保持一致的标准化参数
            return np.clip((value - 0.02) / 0.03, -1, 1)

        # Level特征
        elif 'Level' in feature_name:
            # 指数水平通常在8000-12000之间，标准化到[0, 1]
            if pd.isna(value):
                return 0
            return (value - 10000) / 2000  # 10000为中位数

        # Slope特征
        elif 'Slope' in feature_name:
            # 斜率通常很小，放大处理
            if pd.isna(value):
                return 0
            return np.clip(value * 100, -1, 1)

        else:
            # 其他特征使用简单的相对标准化
            if pd.isna(value):
                return 0
            return np.tanh(value / (abs(value) + 1))  # 使用tanh函数标准化

    def calculate_prediction_score(self):
        """计算预测得分"""
        print("📈 正在计算预测得分...")

        weighted_score = 0
        feature_details = []

        for feature_name, feature_value in self.features.items():
            if pd.isna(feature_value):
                continue

            # 获取特征配置
            config = self.FEATURE_IMPORTANCE.get(feature_name, {'weight': 0.05, 'direction': 1})
            weight = config['weight']
            direction = config['direction']

            # 标准化特征值
            normalized_value = self.normalize_feature(feature_name, feature_value)

            # 计算加权贡献
            contribution = normalized_value * weight * direction
            weighted_score += contribution

            feature_details.append({
                'feature': feature_name,
                'value': feature_value,
                'normalized': normalized_value,
                'weight': weight,
                'direction': direction,
                'contribution': contribution
            })

        # 标准化得分到[0, 1]区间
        # 得分 > 0.5 表示看涨，< 0.5 表示看跌
        prediction_score = (weighted_score + 1) / 2  # 映射到[0, 1]
        prediction_score = np.clip(prediction_score, 0, 1)

        print(f"  ✅ 预测得分计算完成：{prediction_score:.4f}")

        return prediction_score, feature_details

    def interpret_score(self, score):
        """解读预测得分"""
        if score >= 0.65:
            return "强烈看涨", "🟢"
        elif score >= 0.55:
            return "看涨", "🟢"
        elif score >= 0.50:
            return "中性偏涨", "🟡"
        elif score >= 0.45:
            return "中性偏跌", "🟡"
        elif score >= 0.35:
            return "看跌", "🔴"
        else:
            return "强烈看跌", "🔴"

    def generate_report(self):
        """生成预测报告"""
        try:
            # 获取数据
            self.fetch_data()

            # 计算特征
            self.calculate_features()

            # 计算预测得分
            prediction_score, feature_details = self.calculate_prediction_score()

            # 解读得分
            prediction_trend, trend_emoji = self.interpret_score(prediction_score)

            # 获取当前价格和日期
            current_price = self.szse_data['Close'].iloc[-1]
            previous_price = self.szse_data['Close'].iloc[-2]
            price_change = ((current_price - previous_price) / previous_price) * 100
            current_date = self.szse_data.index[-1].strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存JSON报告
            report = {
                'timestamp': timestamp,
                'prediction_date': current_date,
                'index_name': '深证成指',
                'index_code': '399001',
                'current_price': float(current_price),
                'price_change': float(price_change),
                'prediction_score': float(prediction_score),
                'prediction_trend': prediction_trend,
                'features': self.features,
                'feature_details': feature_details
            }

            report_file = os.path.join(output_dir, f'szse_prediction_report_{timestamp}.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON报告已保存: {report_file}")

            # 保存特征CSV
            features_df = pd.DataFrame([self.features])
            features_csv = os.path.join(data_dir, f'szse_prediction_features_{timestamp}.csv')
            features_df.to_csv(features_csv, index=False, encoding='utf-8-sig')
            print(f"💾 特征数据已保存: {features_csv}")

            # 打印报告
            self.print_report(current_price, price_change, prediction_score, prediction_trend, feature_details)

            return report

        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_report(self, current_price, price_change, prediction_score, prediction_trend, feature_details):
        """打印预测报告"""
        print("=" * 100)
        print("                                   深证成指涨跌预测系统                                    ")
        print("                                 基于特征重要性加权评分模型                                  ")
        print("=" * 100)

        current_date = self.szse_data.index[-1].strftime('%Y-%m-%d')
        print(f"📅 分析日期：{current_date}")
        print(f"📊 深成指收盘：{current_price:.2f}（{price_change:+.2f}%）")
        print(f"📈 预测得分：{prediction_score:.4f}")
        print(f"🎯 预测趋势：{prediction_trend}")

        # 获取技术指标数据
        szse_df = self.calculate_technical_indicators(self.szse_data)
        latest_szse = szse_df.iloc[-1]

        # 显示技术指标详情
        print("=" * 100)
        print("                                     技术指标详情                                      ")
        print("=" * 100)

        # 移动平均线
        ma5 = latest_szse.get('MA5', latest_szse['Close'])
        ma10 = latest_szse.get('MA10', latest_szse['Close'])
        ma20 = latest_szse.get('MA20', latest_szse['Close'])
        ma60 = latest_szse.get('MA60', latest_szse['Close'])
        ma120 = latest_szse.get('MA120', latest_szse['Close'])
        ma250 = latest_szse.get('MA250', latest_szse['Close'])

        print(f"\n📍 移动平均线 (MA):")
        print(f"  MA5:   {ma5:.2f} 点  {'↑' if current_price > ma5 else '↓'}")
        print(f"  MA10:  {ma10:.2f} 点  {'↑' if current_price > ma10 else '↓'}")
        print(f"  MA20:  {ma20:.2f} 点  {'↑' if current_price > ma20 else '↓'}")
        print(f"  MA60:  {ma60:.2f} 点  {'↑' if current_price > ma60 else '↓'}")
        print(f"  MA120: {ma120:.2f} 点  {'↑' if current_price > ma120 else '↓'}")
        print(f"  MA250: {ma250:.2f} 点  {'↑' if current_price > ma250 else '↓'}")

        # RSI指标
        rsi = latest_szse.get('RSI')
        if pd.notna(rsi):
            rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常"
            print(f"\n📊 RSI相对强弱指标: {rsi:.1f} ({rsi_signal})")
        else:
            print(f"\n📊 RSI相对强弱指标: 数据不足")

        # MACD指标
        macd = latest_szse.get('MACD')
        macd_signal = latest_szse.get('MACD_Signal')
        macd_hist = latest_szse.get('MACD_Histogram')

        if pd.notna(macd) and pd.notna(macd_signal):
            macd_trend = "金叉" if macd > macd_signal else "死叉"
            print(f"📊 MACD指标: {macd:.2f} (信号线: {macd_signal:.2f}, {macd_trend})")
            if pd.notna(macd_hist):
                macd_hist_signal = "看多" if macd_hist > 0 else "看空"
                print(f"  MACD柱状图: {macd_hist:.2f} ({macd_hist_signal})")
        else:
            print(f"📊 MACD指标: 数据不足")

        # KDJ指标
        k = latest_szse.get('K')
        d = latest_szse.get('D')
        j = latest_szse.get('J')

        if pd.notna(k) and pd.notna(d):
            kdj_signal = "金叉" if k > d else "死叉"
            kdj_level = "超买" if k > 80 else "超卖" if k < 20 else "正常"
            print(f"📊 KDJ指标: K={k:.1f}, D={d:.1f} ({kdj_signal}, {kdj_level})")
            if pd.notna(j):
                print(f"  J值: {j:.1f}")
        else:
            print(f"📊 KDJ指标: 数据不足")

        # 布林带
        bb_upper = latest_szse.get('BB_Upper')
        bb_middle = latest_szse.get('BB_Middle')
        bb_lower = latest_szse.get('BB_Lower')
        bb_position = latest_szse.get('BB_Position')

        if pd.notna(bb_upper) and pd.notna(bb_middle) and pd.notna(bb_lower):
            bb_signal = "上轨" if current_price > bb_upper else "下轨" if current_price < bb_lower else "中轨"
            print(f"📊 布林带 (BB): 上轨={bb_upper:.2f}, 中轨={bb_middle:.2f}, 下轨={bb_lower:.2f}")
            print(f"  当前位置: {bb_signal}")
            if pd.notna(bb_position):
                bb_pos_pct = bb_position * 100
                print(f"  百分位: {bb_pos_pct:.1f}%")
        else:
            print(f"📊 布林带 (BB): 数据不足")

        # 成交量
        volume = latest_szse.get('Volume', 0)
        volume_ma20 = latest_szse.get('Volume_MA250', 0)
        volume_ratio = latest_szse.get('Volume_Ratio')

        print(f"\n📊 成交量:")
        print(f"  当前成交量: {volume:,.0f} 手")
        print(f"  均量: {volume_ma20:,.0f} 手")
        if pd.notna(volume_ratio):
            print(f"  量比: {volume_ratio:.2f}")

        # ATR (真实波幅)
        atr = latest_szse.get('ATR')
        if pd.notna(atr):
            atr_pct = (atr / current_price) * 100
            print(f"\n📊 ATR (平均真实波幅): {atr:.2f} ({atr_pct:.2f}%)")

        # 波动率
        volatility_120d = latest_szse.get('Volatility_120d', 0) * 100
        print(f"📊 120日波动率: {volatility_120d:.2f}%")

        # 支撑阻力位
        resistance = latest_szse.get('Resistance_120d', 0)
        support = latest_szse.get('Support_120d', 0)

        if pd.notna(resistance) and pd.notna(support):
            print(f"\n📊 支撑阻力位:")
            print(f"  120日阻力位: {resistance:.2f} 点 (距离: {((resistance - current_price)/current_price)*100:+.2f}%)")
            print(f"  120日支撑位: {support:.2f} 点 (距离: {((support - current_price)/current_price)*100:+.2f}%)")

        # 按权重排序特征
        sorted_features = sorted(
            feature_details,
            key=lambda x: abs(x['contribution']),
            reverse=True
        )

        print("\n" + "=" * 100)
        print("                              关键因素分析（按权重排序）                                   ")
        print("=" * 100)

        print(f"{'特征':<30} {'当前值':>15} {'标准化':>10} {'权重':>10} {'方向':>8} {'贡献度':>12}")
        print("-" * 100)

        for feat in sorted_features[:10]:  # 显示前10个
            direction_str = "正面" if feat['direction'] > 0 else "负面"
            print(f"{feat['feature']:<30} {feat['value']:>15.4f} {feat['normalized']:>10.4f} {feat['weight']*100:>9.1f}% {direction_str:>8} {feat['contribution']:+.6f}")

        # 统计正面和负面因素
        positive_features = [f for f in feature_details if f['contribution'] > 0]
        negative_features = [f for f in feature_details if f['contribution'] < 0]
        positive_score = sum(f['contribution'] for f in positive_features)
        negative_score = sum(abs(f['contribution']) for f in negative_features)

        print()
        print(f"📊 因素汇总：")
        print(f"  - 正面因素贡献：{positive_score:+.6f}（{len(positive_features)} 个）")
        print(f"  - 负面因素贡献：{-negative_score:.6f}（{len(negative_features)} 个）")
        print(f"  - 综合得分：{positive_score - negative_score:+.6f}")

        print("=" * 100)
        print("                                     关键市场指标                                      ")
        print("=" * 100)

        volume_ma250 = self.features.get('Volume_MA250', 0)

        print(f"250日均线（MA250）：{ma250:.2f} 点")
        print(f"250日成交量均值：{volume_ma250:,.0f} 手")
        print(f"120日均线（MA120）：{ma120:.2f} 点")
        print(f"120日波动率：{volatility_120d:.2f}%")

        print("=" * 100)


def main():
    """主函数"""
    predictor = SZSE_Predictor()
    report = predictor.generate_report()

    if report:
        print()
        print("=" * 100)
        print("✅ 预测完成")
        print("=" * 100)
    else:
        print()
        print("=" * 100)
        print("❌ 预测失败")
        print("=" * 100)


if __name__ == '__main__':
    main()