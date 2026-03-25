#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股市场配置文件
包含A股股票列表、板块映射、评分配置等
"""

# A股股票映射（沪深300核心成分股）
# 包含股票名称、板块代码、类型、评分（防御性、成长性、周期性、流动性、风险）
A_SHARE_STOCK_MAPPING = {
    # 银行股 (bank)
    '601398.SH': {'sector': 'bank', 'name': '工商银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20},
    '600036.SH': {'sector': 'bank', 'name': '招商银行', 'type': 'bank', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 80, 'risk': 25},
    '601318.SH': {'sector': 'bank', 'name': '中国平安', 'type': 'bank', 'defensive': 80, 'growth': 45, 'cyclical': 30, 'liquidity': 75, 'risk': 30},
    '601166.SH': {'sector': 'bank', 'name': '兴业银行', 'type': 'bank', 'defensive': 85, 'growth': 35, 'cyclical': 25, 'liquidity': 75, 'risk': 25},
    '000001.SZ': {'sector': 'bank', 'name': '平安银行', 'type': 'bank', 'defensive': 80, 'growth': 40, 'cyclical': 30, 'liquidity': 70, 'risk': 30},

    # 科技股 (tech)
    '000858.SZ': {'sector': 'tech', 'name': '五粮液', 'type': 'tech', 'defensive': 40, 'growth': 80, 'cyclical': 30, 'liquidity': 85, 'risk': 60},
    '300750.SZ': {'sector': 'tech', 'name': '宁德时代', 'type': 'tech', 'defensive': 30, 'growth': 85, 'cyclical': 50, 'liquidity': 80, 'risk': 70},
    '002415.SZ': {'sector': 'tech', 'name': '海康威视', 'type': 'tech', 'defensive': 45, 'growth': 70, 'cyclical': 40, 'liquidity': 75, 'risk': 55},
    '000333.SZ': {'sector': 'tech', 'name': '美的集团', 'type': 'tech', 'defensive': 50, 'growth': 65, 'cyclical': 45, 'liquidity': 80, 'risk': 50},
    '600887.SH': {'sector': 'tech', 'name': '伊利股份', 'type': 'tech', 'defensive': 55, 'growth': 50, 'cyclical': 40, 'liquidity': 70, 'risk': 45},

    # 半导体股 (semiconductor)
    '600584.SH': {'sector': 'semiconductor', 'name': '长电科技', 'type': 'semiconductor', 'defensive': 30, 'growth': 80, 'cyclical': 70, 'liquidity': 70, 'risk': 75},
    '688981.SH': {'sector': 'semiconductor', 'name': '中芯国际-U', 'type': 'semiconductor', 'defensive': 25, 'growth': 85, 'cyclical': 75, 'liquidity': 65, 'risk': 80},
    '002371.SZ': {'sector': 'semiconductor', 'name': '北方华创', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 60, 'risk': 75},

    # 新能源股 (new_energy)
    '002594.SZ': {'sector': 'new_energy', 'name': '比亚迪', 'type': 'new_energy', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 80, 'risk': 70},
    '300014.SZ': {'sector': 'new_energy', 'name': '亿纬锂能', 'type': 'new_energy', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 65, 'risk': 75},
    '601012.SH': {'sector': 'new_energy', 'name': '隆基绿能', 'type': 'new_energy', 'defensive': 20, 'growth': 75, 'cyclical': 75, 'liquidity': 60, 'risk': 80},

    # 消费股 (consumer)
    '600519.SH': {'sector': 'consumer', 'name': '贵州茅台', 'type': 'consumer', 'defensive': 60, 'growth': 40, 'cyclical': 30, 'liquidity': 90, 'risk': 40},
    '000568.SZ': {'sector': 'consumer', 'name': '泸州老窖', 'type': 'consumer', 'defensive': 50, 'growth': 50, 'cyclical': 40, 'liquidity': 70, 'risk': 50},
    '600809.SH': {'sector': 'consumer', 'name': '山西汾酒', 'type': 'consumer', 'defensive': 45, 'growth': 60, 'cyclical': 45, 'liquidity': 65, 'risk': 55},

    # 医药股 (pharmaceutical)
    '600276.SH': {'sector': 'pharmaceutical', 'name': '恒瑞医药', 'type': 'pharmaceutical', 'defensive': 50, 'growth': 65, 'cyclical': 50, 'liquidity': 70, 'risk': 60},
    '000661.SZ': {'sector': 'pharmaceutical', 'name': '长春高新', 'type': 'pharmaceutical', 'defensive': 45, 'growth': 70, 'cyclical': 55, 'liquidity': 65, 'risk': 65},
    '300760.SZ': {'sector': 'pharmaceutical', 'name': '迈瑞医疗', 'type': 'pharmaceutical', 'defensive': 55, 'growth': 60, 'cyclical': 45, 'liquidity': 75, 'risk': 50},

    # 汽车股 (auto)
    '601238.SH': {'sector': 'auto', 'name': '广汽集团', 'type': 'auto', 'defensive': 35, 'growth': 60, 'cyclical': 65, 'liquidity': 65, 'risk': 70},
    '601633.SH': {'sector': 'auto', 'name': '长城汽车', 'type': 'auto', 'defensive': 30, 'growth': 65, 'cyclical': 70, 'liquidity': 60, 'risk': 75},
    '000625.SZ': {'sector': 'auto', 'name': '长安汽车', 'type': 'auto', 'defensive': 30, 'growth': 70, 'cyclical': 75, 'liquidity': 65, 'risk': 80},

    # 房地产股 (real_estate)
    '000002.SZ': {'sector': 'real_estate', 'name': '万科A', 'type': 'real_estate', 'defensive': 30, 'growth': 30, 'cyclical': 90, 'liquidity': 70, 'risk': 80},
    '600048.SH': {'sector': 'real_estate', 'name': '保利发展', 'type': 'real_estate', 'defensive': 35, 'growth': 35, 'cyclical': 85, 'liquidity': 65, 'risk': 75},
    '001979.SZ': {'sector': 'real_estate', 'name': '招商蛇口', 'type': 'real_estate', 'defensive': 30, 'growth': 40, 'cyclical': 80, 'liquidity': 60, 'risk': 70},

    # 能源股 (energy)
    '601857.SH': {'sector': 'energy', 'name': '中国石油', 'type': 'energy', 'defensive': 70, 'growth': 25, 'cyclical': 80, 'liquidity': 75, 'risk': 50},
    '600028.SH': {'sector': 'energy', 'name': '中国石化', 'type': 'energy', 'defensive': 75, 'growth': 20, 'cyclical': 75, 'liquidity': 70, 'risk': 45},
    '601088.SH': {'sector': 'energy', 'name': '中国神华', 'type': 'energy', 'defensive': 80, 'growth': 25, 'cyclical': 70, 'liquidity': 65, 'risk': 40},

    # 公用事业股 (utility)
    '600900.SH': {'sector': 'utility', 'name': '长江电力', 'type': 'utility', 'defensive': 95, 'growth': 20, 'cyclical': 10, 'liquidity': 75, 'risk': 10},
    '600019.SH': {'sector': 'utility', 'name': '宝钢股份', 'type': 'utility', 'defensive': 85, 'growth': 25, 'cyclical': 20, 'liquidity': 70, 'risk': 20},
    '601899.SH': {'sector': 'utility', 'name': '紫金矿业', 'type': 'utility', 'defensive': 65, 'growth': 45, 'cyclical': 60, 'liquidity': 80, 'risk': 50},

    # 保险股 (insurance)
    '601601.SH': {'sector': 'insurance', 'name': '中国太保', 'type': 'insurance', 'defensive': 80, 'growth': 40, 'cyclical': 25, 'liquidity': 65, 'risk': 35},
    '601336.SH': {'sector': 'insurance', 'name': '新华保险', 'type': 'insurance', 'defensive': 80, 'growth': 35, 'cyclical': 30, 'liquidity': 60, 'risk': 40},

    # 券商股 (broker)
    '600030.SH': {'sector': 'broker', 'name': '中信证券', 'type': 'broker', 'defensive': 55, 'growth': 55, 'cyclical': 70, 'liquidity': 70, 'risk': 60},
    '601688.SH': {'sector': 'broker', 'name': '华泰证券', 'type': 'broker', 'defensive': 50, 'growth': 60, 'cyclical': 75, 'liquidity': 65, 'risk': 65},

    # 通信股 (telecom)
    '600050.SH': {'sector': 'telecom', 'name': '中国联通', 'type': 'telecom', 'defensive': 85, 'growth': 25, 'cyclical': 20, 'liquidity': 65, 'risk': 20},
    '000063.SZ': {'sector': 'telecom', 'name': '中兴通讯', 'type': 'telecom', 'defensive': 50, 'growth': 65, 'cyclical': 45, 'liquidity': 70, 'risk': 55},

    # 白酒股 (liquor)
    '000596.SZ': {'sector': 'liquor', 'name': '古井贡酒', 'type': 'liquor', 'defensive': 45, 'growth': 55, 'cyclical': 40, 'liquidity': 60, 'risk': 55},
    '603589.SH': {'sector': 'liquor', 'name': '口子窖', 'type': 'liquor', 'defensive': 40, 'growth': 50, 'cyclical': 45, 'liquidity': 55, 'risk': 60},
}

# 板块名称映射（统一中文名称）
SECTOR_NAME_MAPPING = {
    'bank': '银行股',
    'tech': '科技股',
    'semiconductor': '半导体股',
    'new_energy': '新能源股',
    'consumer': '消费股',
    'pharmaceutical': '医药股',
    'auto': '汽车股',
    'real_estate': '房地产股',
    'energy': '能源股',
    'utility': '公用事业股',
    'insurance': '保险股',
    'broker': '券商股',
    'telecom': '通信股',
    'liquor': '白酒股',
}

# A股自选股列表（从 A_SHARE_STOCK_MAPPING 中提取）
WATCHLIST = {
    "600519.SH": "贵州茅台",
    "600036.SH": "招商银行",
    "000001.SZ": "平安银行",
    "002594.SZ": "比亚迪",
    "300750.SZ": "宁德时代",
    "000858.SZ": "五粮液",
    "600030.SH": "中信证券",
    "601398.SH": "工商银行",
    "000333.SZ": "美的集团",
    "600276.SH": "恒瑞医药",
    "600900.SH": "长江电力",
    "601857.SH": "中国石油",
    "601318.SH": "中国平安",
    "601012.SH": "隆基绿能",
    "601633.SH": "长城汽车",
    "000568.SZ": "泸州老窖",
    "002415.SZ": "海康威视",
    "600019.SH": "宝钢股份",
    "600887.SH": "伊利股份",
    "601088.SH": "中国神华",
    "600584.SH": "长电科技",
    "300760.SZ": "迈瑞医疗",
    "601899.SH": "紫金矿业",
    "601166.SH": "兴业银行",
    "601601.SH": "中国太保",
    "600050.SH": "中国联通",
    "601688.SH": "华泰证券",
    "601238.SH": "广汽集团",
    "000002.SZ": "万科A",
    "688981.SH": "中芯国际-U",
}

# 指数映射
INDEX_MAPPING = {
    'sse': {
        'name': '上证指数',
        'code': '000001',
        'source': 'akshare',
        'data_func': 'get_sse_index_data',
        'description': '上海证券交易所综合指数'
    },
    'szse': {
        'name': '深证成指',
        'code': '399001',
        'source': 'akshare',
        'data_func': 'get_szse_index_data',
        'description': '深圳证券交易所成分指数'
    },
    'csi300': {
        'name': '沪深300',
        'code': '000300',
        'source': 'akshare',
        'data_func': 'get_csi300_index_data',
        'description': '沪深300指数'
    },
    'sz50': {
        'name': '上证50',
        'code': '000016',
        'source': 'akshare',
        'data_func': 'get_sz50_index_data',
        'description': '上证50指数'
    },
}

# 交易所配置
EXCHANGE_CONFIG = {
    'SH': {
        'name': '上海证券交易所',
        'code': 'SH',
        'trading_hours': {'morning': ('09:30', '11:30'), 'afternoon': ('13:00', '15:00')},
        'price_limit': 0.10,  # ±10%
    },
    'SZ': {
        'name': '深圳证券交易所',
        'code': 'SZ',
        'trading_hours': {'morning': ('09:30', '11:30'), 'afternoon': ('13:00', '15:00')},
        'price_limit': {'main': 0.10, 'startup': 0.20},  # 主板±10%，创业板/科创板±20%
    },
}

# A股市场特征
MARKET_FEATURES = {
    'trading_days_per_year': 242,  # A股每年约242个交易日
    'currency': 'CNY',  # 人民币
    'timezone': 'Asia/Shanghai',
    'settlement': 'T+1',  # T+1交割
    'price_tick': 0.01,  # 最小变动单位
    'board_lot': 100,  # 最小交易单位（手）
}

# A股风险控制参数
RISK_CONTROL = {
    'max_single_position': 0.20,  # 单只股票最大仓位20%
    'max_sector_position': 0.30,  # 单个板块最大仓位30%
    'max_daily_loss': 0.05,  # 单日最大亏损5%
    'max_drawdown': 0.15,  # 最大回撤15%
    'stop_loss_ratio': 0.08,  # 止损比例8%
    'take_profit_ratio': 0.20,  # 止盈比例20%
}