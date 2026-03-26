#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据获取模块
使用AkShare获取上证指数、深证成指和A股个股数据
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import warnings
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')


# 创建带有重试机制的session
def create_session():
    """创建带有重试机制的requests session"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# 设置AkShare使用自定义session
try:
    ak.set_session(create_session())
except:
    pass


def get_sse_index_data(period_days=730, retry_count=3):
    """
    获取上证指数数据

    Args:
        period_days (int): 获取数据的天数，默认730天（约2年）
        retry_count (int): 重试次数

    Returns:
        pandas.DataFrame: 包含上证指数数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    for attempt in range(retry_count):
        try:
            # 方法1: 使用stock_zh_index_daily获取上证指数
            print(f"  尝试获取上证指数数据 (第{attempt + 1}/{retry_count}次)...")
            df = ak.stock_zh_index_daily(symbol="sh000001")

            if df.empty:
                raise ValueError("返回数据为空")

            # 重命名列以保持一致性
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'amount': 'Amount'
            })

            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df['Date'])

            # 按日期排序
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)

            # 限制数据天数
            if len(df) > period_days:
                df = df.tail(period_days)

            print(f"  ✅ 上证指数数据获取成功，共 {len(df)} 条")
            return df

        except Exception as e:
            print(f"  ⚠️ 上证指数数据获取失败 (第{attempt + 1}次): {e}")
            if attempt < retry_count - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                # 最后一次尝试使用备用方法
                try:
                    print(f"  尝试备用方法获取上证指数...")
                    df = ak.index_zh_a_hist(symbol="000001", period="daily")

                    if not df.empty:
                        df = df.rename(columns={
                            '日期': 'Date',
                            '开盘': 'Open',
                            '收盘': 'Close',
                            '最高': 'High',
                            '最低': 'Low',
                            '成交量': 'Volume',
                            '成交额': 'Amount'
                        })
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date')
                        df.set_index('Date', inplace=True)

                        if len(df) > period_days:
                            df = df.tail(period_days)

                        print(f"  ✅ 备用方法成功，共 {len(df)} 条")
                        return df
                except:
                    pass

    print("❌ 上证指数数据获取失败")
    return None


def get_szse_index_data(period_days=730, retry_count=3):
    """
    获取深证成指数据

    Args:
        period_days (int): 获取数据的天数，默认730天（约2年）
        retry_count (int): 重试次数

    Returns:
        pandas.DataFrame: 包含深证成指数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    for attempt in range(retry_count):
        try:
            # 使用stock_zh_index_daily获取深证成指
            print(f"  尝试获取深证成指数据 (第{attempt + 1}/{retry_count}次)...")
            df = ak.stock_zh_index_daily(symbol="sz399001")

            if df.empty:
                raise ValueError("返回数据为空")

            # 重命名列以保持一致性
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'amount': 'Amount'
            })

            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df['Date'])

            # 按日期排序
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)

            # 限制数据天数
            if len(df) > period_days:
                df = df.tail(period_days)

            print(f"  ✅ 深证成指数据获取成功，共 {len(df)} 条")
            return df

        except Exception as e:
            print(f"  ⚠️ 深证成指数据获取失败 (第{attempt + 1}次): {e}")
            if attempt < retry_count - 1:
                time.sleep(2)

    print("❌ 深证成指数据获取失败")
    return None


def get_sse_index_realtime():
    """
    获取上证指数实时行情

    Returns:
        dict: 包含上证指数实时行情的字典，包括代码、名称、最新价、涨跌额、涨跌幅、今开、最高、最低、昨收
    """
    try:
        realtime_data = ak.stock_zh_index_spot_em()
        sse_row = realtime_data[realtime_data['代码'] == '000001']

        if not sse_row.empty:
            row = sse_row.iloc[0]
            return {
                'code': row['代码'],
                'name': row['名称'],
                'current_price': row['最新价'],
                'change': row['涨跌额'],
                'change_pct': row['涨跌幅'],
                'open': row['今开'],
                'high': row['最高'],
                'low': row['最低'],
                'pre_close': row['昨收'],
                'volume': row.get('成交量', 0),
                'amount': row.get('成交额', 0)
            }
        return None

    except Exception as e:
        print(f"获取上证指数实时行情失败: {e}")
        return None


def get_szse_index_realtime():
    """
    获取深证成指实时行情

    Returns:
        dict: 包含深证成指实时行情的字典
    """
    try:
        realtime_data = ak.stock_zh_index_spot_em()
        szse_row = realtime_data[realtime_data['代码'] == '399001']

        if not szse_row.empty:
            row = szse_row.iloc[0]
            return {
                'code': row['代码'],
                'name': row['名称'],
                'current_price': row['最新价'],
                'change': row['涨跌额'],
                'change_pct': row['涨跌幅'],
                'open': row['今开'],
                'high': row['最高'],
                'low': row['最低'],
                'pre_close': row['昨收'],
                'volume': row.get('成交量', 0),
                'amount': row.get('成交额', 0)
            }
        return None

    except Exception as e:
        print(f"获取深证成指实时行情失败: {e}")
        return None


def get_a_stock_realtime(stock_code):
    """
    获取A股个股实时行情

    Args:
        stock_code (str): 股票代码（如 "600519" 或 "600519.SH"）

    Returns:
        dict: 包含个股实时行情的字典
    """
    try:
        # 标准化股票代码
        clean_code = stock_code.replace('.SH', '').replace('.SZ', '')

        realtime_data = ak.stock_zh_a_spot_em()
        stock_row = realtime_data[realtime_data['代码'] == clean_code]

        if not stock_row.empty:
            row = stock_row.iloc[0]
            return {
                'code': row['代码'],
                'name': row['名称'],
                'current_price': row['最新价'],
                'change': row['涨跌额'],
                'change_pct': row['涨跌幅'],
                'open': row['今开'],
                'high': row['最高'],
                'low': row['最低'],
                'pre_close': row['昨收'],
                'volume': row['成交量'],
                'amount': row['成交额'],
                'amplitude': row['振幅'],
                'pe': row.get('市盈率-动态', 0),
                'market_cap': row.get('总市值', 0)
            }
        return None

    except Exception as e:
        print(f"获取股票 {stock_code} 实时行情失败: {e}")
        return None


def get_csi300_index_data(period_days=730):
    """
    获取沪深300指数数据

    Args:
        period_days (int): 获取数据的天数，默认730天（约2年）

    Returns:
        pandas.DataFrame: 包含沪深300指数数据的DataFrame
    """
    try:
        # AkShare获取沪深300指数（000300）
        df = ak.index_zh_a_hist(symbol="000300", period="daily")

        if df.empty:
            print("获取沪深300指数数据失败：返回数据为空")
            return None

        # 重命名列以保持一致性
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount'
        })

        # 确保日期格式正确
        df['Date'] = pd.to_datetime(df['Date'])

        # 按日期排序
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        # 限制数据天数
        if len(df) > period_days:
            df = df.tail(period_days)

        return df

    except Exception as e:
        print(f"获取沪深300指数数据失败: {e}")
        return None


def get_a_stock_data(stock_code, period_days=90, retry_count=3):
    """
    获取A股个股数据

    Args:
        stock_code (str): 股票代码
            - 上交所：如 "600519" (贵州茅台) 或 "601398.SH"
            - 深交所：如 "000858" (五粮液) 或 "000858.SZ"
        period_days (int): 获取数据的天数，默认90天
        retry_count (int): 重试次数

    Returns:
        pandas.DataFrame: 包含个股数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    # 标准化股票代码（移除交易所后缀）
    clean_code = stock_code.replace('.SH', '').replace('.SZ', '')

    # 判断交易所并添加前缀
    exchange = 'sh' if clean_code.startswith('6') else 'sz'
    symbol_with_exchange = f"{exchange}{clean_code}"

    for attempt in range(retry_count):
        try:
            # 使用stock_zh_a_daily方法（更稳定）
            df = ak.stock_zh_a_daily(symbol=symbol_with_exchange, adjust='qfq')

            if df.empty:
                raise ValueError("返回数据为空")

            # 重命名列以保持一致性
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # 添加股票代码列
            df['Code'] = stock_code

            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df.index)

            # 重置索引并设置Date为索引
            df.reset_index(drop=True, inplace=True)
            df.set_index('Date', inplace=True)

            # 按日期排序
            df = df.sort_values('Date')

            # 限制数据天数
            if len(df) > period_days:
                df = df.tail(period_days)

            return df

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"获取股票 {stock_code} 数据失败: {e}")
                return None


def get_a_stock_info(stock_code):
    """
    获取A股个股基本信息

    Args:
        stock_code (str): 股票代码

    Returns:
        dict: 包含个股基本信息的字典
    """
    try:
        # 标准化股票代码
        clean_code = stock_code.replace('.SH', '').replace('.SZ', '')

        # AkShare获取个股基本信息
        info = ak.stock_individual_info_em(symbol=clean_code)

        if info.empty:
            print(f"获取股票 {stock_code} 信息失败：返回数据为空")
            return None

        # 转换为字典
        info_dict = dict(zip(info['item'], info['value']))

        return info_dict

    except Exception as e:
        print(f"获取股票 {stock_code} 信息失败: {e}")
        return None


def get_a_stock_list(market='all'):
    """
    获取A股股票列表

    Args:
        market (str): 市场类型
            - 'all': 全部A股
            - 'sh': 上交所
            - 'sz': 深交所

    Returns:
        pandas.DataFrame: 包含股票列表的DataFrame
    """
    try:
        if market == 'all':
            # 获取全部A股列表
            df = ak.stock_info_a_code_name()
        elif market == 'sh':
            df = ak.stock_sh_a_spot_em()
        elif market == 'sz':
            df = ak.stock_sz_a_spot_em()
        else:
            print(f"不支持的市场类型: {market}")
            return None

        if df.empty:
            print(f"获取{market}市场股票列表失败：返回数据为空")
            return None

        return df

    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return None


def get_sector_data(sector_code, period_days=730):
    """
    获取板块数据

    Args:
        sector_code (str): 板块代码
            - 如 'new' (新能源)、'tmt' (TMT) 等
        period_days (int): 获取数据的天数

    Returns:
        pandas.DataFrame: 包含板块数据的DataFrame
    """
    try:
        # AkShare获取板块数据（使用东方财富概念板块）
        df = ak.stock_board_concept_cons_em(symbol=sector_code)

        if df.empty:
            print(f"获取板块 {sector_code} 数据失败：返回数据为空")
            return None

        return df

    except Exception as e:
        print(f"获取板块数据失败: {e}")
        return None


def get_market_breadth(market='all'):
    """
    获取市场广度数据（涨跌家数、涨停跌停等）

    Args:
        market (str): 市场类型

    Returns:
        dict: 市场广度数据
    """
    try:
        # 获取A股实时数据
        if market == 'all':
            df = ak.stock_zh_a_spot_em()
        elif market == 'sh':
            df = ak.stock_sh_a_spot_em()
        elif market == 'sz':
            df = ak.stock_sz_a_spot_em()
        else:
            print(f"不支持的市场类型: {market}")
            return None

        if df.empty:
            return None

        # 计算市场广度
        total = len(df)
        up = len(df[df['涨跌幅'] > 0])
        down = len(df[df['涨跌幅'] < 0])
        flat = len(df[df['涨跌幅'] == 0])

        # 涨停跌停（主板10%，创业板/科创板20%）
        limit_up = len(df[df['涨跌幅'] >= 9.9])
        limit_down = len(df[df['涨跌幅'] <= -9.9])

        return {
            'total': total,
            'up': up,
            'down': down,
            'flat': flat,
            'up_ratio': up / total if total > 0 else 0,
            'limit_up': limit_up,
            'limit_down': limit_down
        }

    except Exception as e:
        print(f"获取市场广度数据失败: {e}")
        return None


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("A股数据获取模块测试")
    print("=" * 80)

    # 测试上证指数
    print("\n1. 测试上证指数数据获取...")
    sse_data = get_sse_index_data(period_days=30)
    if sse_data is not None:
        print(f"✅ 获取成功，共 {len(sse_data)} 条数据")
        print(sse_data.head())
        print(f"最新收盘价: {sse_data['Close'].iloc[-1]:.2f}")

    # 测试深证成指
    print("\n2. 测试深证成指数据获取...")
    szse_data = get_szse_index_data(period_days=30)
    if szse_data is not None:
        print(f"✅ 获取成功，共 {len(szse_data)} 条数据")
        print(szse_data.head())
        print(f"最新收盘价: {szse_data['Close'].iloc[-1]:.2f}")

    # 测试沪深300
    print("\n3. 测试沪深300指数数据获取...")
    csi300_data = get_csi300_index_data(period_days=30)
    if csi300_data is not None:
        print(f"✅ 获取成功，共 {len(csi300_data)} 条数据")
        print(csi300_data.head())
        print(f"最新收盘价: {csi300_data['Close'].iloc[-1]:.2f}")

    # 测试个股数据
    print("\n4. 测试个股数据获取（贵州茅台 600519）...")
    stock_data = get_a_stock_data('600519', period_days=30)
    if stock_data is not None:
        print(f"✅ 获取成功，共 {len(stock_data)} 条数据")
        print(stock_data.head())
        print(f"最新收盘价: {stock_data['Close'].iloc[-1]:.2f}")

    # 测试个股信息
    print("\n5. 测试个股信息获取（贵州茅台 600519）...")
    stock_info = get_a_stock_info('600519')
    if stock_info:
        print(f"✅ 获取成功")
        print(f"股票名称: {stock_info.get('股票简称', 'N/A')}")
        print(f"总市值: {stock_info.get('总市值', 'N/A')}")
        print(f"市盈率: {stock_info.get('市盈率-动态', 'N/A')}")

    # 测试市场广度
    print("\n6. 测试市场广度数据获取...")
    breadth = get_market_breadth()
    if breadth:
        print(f"✅ 获取成功")
        print(f"总股票数: {breadth['total']}")
        print(f"上涨家数: {breadth['up']} ({breadth['up_ratio']:.2%})")
        print(f"下跌家数: {breadth['down']}")
        print(f"涨停家数: {breadth['limit_up']}")
        print(f"跌停家数: {breadth['limit_down']}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)