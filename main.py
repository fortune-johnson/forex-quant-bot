"""
FOREX QUANT v7.5 - ULTIMATE TRADING INTELLIGENCE SYSTEM
═══════════════════════════════════════════════════════════════════════════
Institutional-Grade Metrics · VPIN · Kyle's Lambda · Flow Toxicity
Self-Learning ML · Historical Backtesting · Auto Alerts · Liquidity Zones
Smart Money Detection · Session Advisor · Pattern Memory Engine

pip install python-telegram-bot aiohttp matplotlib numpy scipy scikit-learn
python forex_quant_v75.py
"""

import asyncio
import aiohttp
import logging
import sys
import platform
import io
import os
import json
import gc
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import TimedOut, NetworkError
import warnings
warnings.filterwarnings('ignore')
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

OANDA_API_KEY     = os.environ.get("OANDA_API_KEY",    "67eae51bea4d1ddbc5899613f6660977-ac693290039d6b19fe0634712ee248a4")
OANDA_ACCOUNT_ID  = os.environ.get("OANDA_ACCOUNT_ID", "101-001-27452070-001")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN",   "7123896226:AAFLeyCPnfJjJgakBH8twdSDPyznu69ZQa4")
ENVIRONMENT       = os.environ.get("OANDA_ENV",        "practice")

PREDICTIONS_FILE         = "predictions_v75.json"
ACTIVE_PREDICTIONS_FILE  = "active_predictions_v75.json"
ML_MODEL_FILE            = "ml_model_v75.pkl"
HISTORICAL_BT_FILE       = "historical_backtest_v75.json"
PATTERN_MEMORY_FILE      = "pattern_memory_v75.json"
USER_SUBSCRIPTIONS_FILE  = "user_subscriptions_v75.json"
GLOBAL_STATS_FILE        = "global_stats_v75.json"

API_BASE = ("https://api-fxpractice.oanda.com" if ENVIRONMENT == "practice"
            else "https://api-fxtrade.oanda.com")
HEADERS  = {"Authorization": f"Bearer {OANDA_API_KEY}",
            "Accept-Datetime-Format": "RFC3339"}

CURRENCIES = ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"]

FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "NZD_USD", "USD_CHF", "EUR_GBP", "EUR_JPY", "EUR_AUD",
    "EUR_CAD", "EUR_NZD", "EUR_CHF", "GBP_JPY", "GBP_AUD",
    "GBP_CAD", "GBP_NZD", "GBP_CHF", "AUD_JPY", "AUD_CAD",
    "AUD_NZD", "AUD_CHF", "CAD_JPY", "CAD_CHF", "NZD_JPY",
    "NZD_CHF", "CHF_JPY", "NZD_CAD"
]

METALS  = ["XAU_USD", "XAG_USD"]
INDICES = ["NAS100_USD"]
ALL_PAIRS = FOREX_PAIRS + METALS + INDICES




ASSET_CATEGORIES = {
    "forex_major": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"],
    "forex_cross": [
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_AUD", "GBP_AUD", "AUD_JPY", "EUR_CAD",
        "GBP_CAD", "AUD_CAD", "EUR_NZD", "GBP_NZD", "AUD_NZD", "NZD_JPY", "CAD_JPY",
        "EUR_CHF", "GBP_CHF", "AUD_CHF", "NZD_CHF", "CAD_CHF", "CHF_JPY", "NZD_CAD"
    ],
    "metals":  ["XAU_USD", "XAG_USD"],
    "indices": ["NAS100_USD"]
}

TIMEFRAMES = {
    "M5": {"label": "5 Minutes", "candles": 200, "emoji": "⚡"},
    "H1": {"label": "1 Hour",    "candles": 100, "emoji": "🕐"},
    "H4": {"label": "4 Hours",   "candles": 75,  "emoji": "📊"}
}

SESSIONS = {
    "TOKYO":   {"start": 0,  "end": 9,  "emoji": "🗼",
                "pairs": ["USD_JPY", "AUD_JPY", "NZD_JPY", "EUR_JPY", "GBP_JPY"]},
    "LONDON":  {"start": 7,  "end": 16, "emoji": "🏦",
                "pairs": ["EUR_USD", "GBP_USD", "EUR_GBP", "USD_CHF", "EUR_CHF"]},
    "NEW_YORK":{"start": 13, "end": 22, "emoji": "🗽",
                "pairs": ["EUR_USD", "GBP_USD", "USD_CAD", "USD_JPY", "XAU_USD"]},
    "OVERLAP": {"start": 13, "end": 16, "emoji": "⚡",
                "pairs": ["EUR_USD", "GBP_USD", "EUR_GBP", "XAU_USD"]}
}

MIN_CONFLUENCE_FACTORS   = 4
MIN_IMBALANCE_PCT        = 10
MIN_CONFIDENCE           = 55
MONITORING_INTERVAL      = 30
ALERT_SCAN_INTERVAL      = 300
ML_MIN_SAMPLES           = 20
STRENGTH_CACHE_TTL       = 300
HISTORICAL_BT_CANDLES    = 500
MIN_ALERT_CONFIDENCE     = 72
MIN_ALERT_INST_SCORE     = 60
MIN_ALERT_FACTORS        = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("ForexQuant")


# ════════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ════════════════════════════════════════════════════════════════

class SetupQuality(Enum):
    A_PLUS   = "A+ (Strong)"
    A        = "A (Good)"
    B        = "B (Moderate)"
    C        = "C (Weak)"
    NO_TRADE = "No Clear Setup"

class SessionType(Enum):
    TOKYO     = "Tokyo"
    LONDON    = "London"
    NEW_YORK  = "New York"
    OVERLAP   = "Session Overlap"
    OFF_HOURS = "Off Hours"


@dataclass
class Candle:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_abs(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def buy_volume(self) -> float:
        if self.range == 0:
            return self.volume * 0.5
        body_pct = self.body_abs / self.range
        return (self.volume * (0.5 + body_pct * 0.5)
                if self.is_bullish
                else self.volume * (0.5 - body_pct * 0.5))

    @property
    def sell_volume(self) -> float:
        return self.volume - self.buy_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass
class OrderFlowData:
    buy_volume: float = 0; sell_volume: float = 0; total_volume: float = 0
    cvd: float = 0; delta: float = 0; buy_pct: float = 50; sell_pct: float = 50
    imbalance: float = 0; price: float = 0; vwap: float = 0
    delta_momentum: float = 0; volume_trend: float = 0
    recent_delta_5: float = 0; recent_delta_10: float = 0; recent_delta_20: float = 0
    cvd_slope: float = 0; volume_acceleration: float = 0
    buying_climax: bool = False; selling_climax: bool = False


@dataclass
class AdvancedOrderFlowData:
    vpin: float = 0; vpin_level: str = "LOW"
    kyle_lambda: float = 0; kyle_lambda_interpretation: str = ""
    toxicity: float = 0; toxicity_level: str = "LOW"
    amihud_illiquidity: float = 0; liquidity_level: str = "NORMAL"
    market_depth_imbalance: float = 0; depth_bias: str = "NEUTRAL"
    iceberg_orders: Dict = field(default_factory=dict); iceberg_count: int = 0
    smart_money_activity: str = "LOW"; informed_trader_signal: str = "NEUTRAL"
    effective_spread: float = 0; realized_spread: float = 0
    price_impact: float = 0; trade_arrival_rate: float = 0
    volume_clustering: float = 0; absorption_ratio: float = 0
    aggressor_side: str = "NEUTRAL"; institutional_flow_score: float = 0
    roll_spread: float = 0


@dataclass
class LiquidityZone:
    price: float
    zone_type: str
    strength: float
    description: str
    upper: float = 0.0
    lower: float = 0.0
    tested: bool = False
    created_at: str = ""


@dataclass
class VolumeProfile:
    poc: float = 0; vah: float = 0; val: float = 0
    hvn: List[float] = field(default_factory=list)
    lvn: List[float] = field(default_factory=list)
    total_volume: float = 0


@dataclass
class OrderBookData:
    price: float = 0; total_longs: float = 0; total_shorts: float = 0
    net_imbalance: float = 0; pending_delta: float = 0
    breakout_bias: str = "BALANCED"; long_pct: float = 50; short_pct: float = 50
    pressure_direction: str = "NEUTRAL"


@dataclass
class PositionBookData:
    price: float = 0; long_pct: float = 50; short_pct: float = 50
    skew: float = 0; contrarian_signal: str = "NEUTRAL"
    underwater_longs: float = 0; underwater_shorts: float = 0
    trapped_longs_pct: float = 0; trapped_shorts_pct: float = 0
    total_underwater: float = 0; crowded_trade_index: float = 0
    pain_threshold: float = 0; squeeze_potential: str = "LOW"


@dataclass
class CurrencyStrength:
    currency: str = ""; strength: float = 0; trend: str = "NEUTRAL"; rank: int = 0


@dataclass
class ConfluenceFactor:
    name: str; direction: str; strength: float; description: str


@dataclass
class PredictionResult:
    has_setup: bool; direction: str; quality: SetupQuality
    confidence: float; factors: List[ConfluenceFactor]
    bullish_count: int; bearish_count: int; neutral_count: int
    reasons: List[str]


@dataclass
class QuantPrediction:
    prediction_id: str; pair: str; timeframe: str; timestamp: str
    current_price: float; direction: str; target_price: float
    invalidation_price: float; confidence: float; quality: str
    reasons: List[str]; key_levels: List[Dict]; factors_aligned: int
    features: List[float]
    status: str = "ACTIVE"; outcome: Optional[str] = None
    hit_time: Optional[str] = None; chat_id: Optional[int] = None
    ml_confidence: float = 0.0; ml_used: bool = False
    pips_gained: float = 0.0


@dataclass
class PatternRecord:
    pattern_id: str; pair: str; timeframe: str; timestamp: str
    setup_conditions: Dict; direction: str; outcome: str
    pips: float; confidence: float; quality: str
    institutional_score: float; vpin: float; session: str


@dataclass
class HistoricalBacktestResult:
    pair: str; timeframe: str; period_days: int
    total_signals: int; wins: int; losses: int
    win_rate: float; avg_pips: float
    best_session: str; worst_session: str; best_quality: str
    quality_win_rates: Dict; session_win_rates: Dict
    confidence_accuracy: Dict; ml_accuracy: float
    last_run: str; patterns_learned: int



class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")
    def log_message(self, format, *args):
        pass  # suppress logs

def start_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()

# ════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_pip_info(instrument: str) -> Tuple[float, int]:
    if "JPY" in instrument:   return 0.01, 3
    elif "XAU" in instrument: return 0.1, 2
    elif "XAG" in instrument: return 0.01, 3
    elif "NAS" in instrument: return 1.0, 1
    else:                     return 0.0001, 5

def pip_value(instrument: str) -> float:
    return get_pip_info(instrument)[0]

def format_price(value: float, instrument: str) -> str:
    return f"{value:.{get_pip_info(instrument)[1]}f}"

def pips_diff(instrument: str, diff: float) -> float:
    return abs(diff) / pip_value(instrument)

def format_number(value: float) -> str:
    if abs(value) >= 1e6: return f"{value/1e6:.2f}M"
    if abs(value) >= 1e3: return f"{value/1e3:.1f}K"
    return f"{value:.0f}"

def format_signed(value: float) -> str:
    if abs(value) >= 1e6: return f"{value/1e6:+.2f}M"
    if abs(value) >= 1e3: return f"{value/1e3:+.1f}K"
    return f"{value:+.0f}"

def format_pct(value: float) -> str:
    return f"{value:+.1f}%"

def format_scientific(value: float, precision: int = 2) -> str:
    if abs(value) < 0.0001 or abs(value) > 10000:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"

def progress_bar(pct: float, length: int = 10) -> str:
    filled = int(max(0, min(pct, 100)) / 100 * length)
    return "█" * filled + "░" * (length - filled)

def escape_html(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def pair_to_display(instrument: str) -> str:
    return instrument.replace("_", "/")

def get_asset_emoji(instrument: str) -> str:
    if "XAU" in instrument: return "🥇"
    elif "XAG" in instrument: return "🥈"
    elif "NAS" in instrument: return "📈"
    return "💱"

def get_current_session() -> Tuple[str, SessionType, float]:
    now  = datetime.now(timezone.utc)
    hour = now.hour
    if 13 <= hour < 16:
        return "London-NY Overlap", SessionType.OVERLAP,   16 - hour - now.minute / 60
    if 7 <= hour < 9:
        return "Tokyo-London Overlap", SessionType.OVERLAP, 9 - hour - now.minute / 60
    if 0 <= hour < 9:
        return "Tokyo/Asian",     SessionType.TOKYO,     9  - hour - now.minute / 60
    if 7 <= hour < 16:
        return "London/European", SessionType.LONDON,    16 - hour - now.minute / 60
    if 13 <= hour < 22:
        return "New York",        SessionType.NEW_YORK,  22 - hour - now.minute / 60
    return "Off Hours", SessionType.OFF_HOURS, 0.0

def get_session_name(hour: int) -> str:
    if 13 <= hour < 16: return "OVERLAP"
    if 0 <= hour < 9:   return "TOKYO"
    if 7 <= hour < 16:  return "LONDON"
    if 13 <= hour < 22: return "NEW_YORK"
    return "OFF_HOURS"

def parse_pair_from_callback(data: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    remainder = data[len(prefix):]
    for pair in sorted(ALL_PAIRS, key=len, reverse=True):
        if remainder.startswith(pair + "_"):
            return pair, remainder[len(pair) + 1:]
    return None, None

def parse_pair_tf(data: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse pair and timeframe from callback data."""
    remainder = data[len(prefix):]
    for pair in sorted(ALL_PAIRS, key=len, reverse=True):
        if remainder.startswith(pair + "_"):
            tf = remainder[len(pair) + 1:]
            if tf in TIMEFRAMES:
                return pair, tf
            return None, None
    return None, None

def load_json_file(filepath: str, default: Any) -> Any:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading {filepath}: {e}")
    return default

def save_json_file(filepath: str, data: Any):
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.error(f"Error saving {filepath}: {e}")

# ── Short aliases used throughout the code ───────────────────────
def load_json(filepath: str, default: Any) -> Any:
    return load_json_file(filepath, default)

def save_json(filepath: str, data: Any):
    save_json_file(filepath, data)

def fmt_price(value: float, instrument: str) -> str:
    return format_price(value, instrument)

def fmt_num(value: float) -> str:
    return format_number(value)

def fmt_signed(value: float) -> str:
    return format_signed(value)

def fmt_pct(value: float) -> str:
    return format_pct(value)

def fmt_sci(value: float, precision: int = 2) -> str:
    return format_scientific(value, precision)

def pbar(pct: float, length: int = 10) -> str:
    return progress_bar(pct, length)

def esc(text: str) -> str:
    return escape_html(text)

def pair_display(instrument: str) -> str:
    return pair_to_display(instrument)

def asset_emoji(instrument: str) -> str:
    return get_asset_emoji(instrument)

def session_name(hour: int) -> str:
    return get_session_name(hour)

def current_session() -> Tuple[str, float]:
    name, _, hours = get_current_session()
    return name, hours

def pips(instrument: str, diff: float) -> float:
    return pips_diff(instrument, diff)

def pip_val(instrument: str) -> float:
    return pip_value(instrument)

# ── Async safe helpers ───────────────────────────────────────────
async def safe_answer_callback(query) -> bool:
    try:
        await asyncio.wait_for(query.answer(), timeout=5.0)
        return True
    except Exception:
        return False

async def safe_delete_message(message) -> bool:
    try:
        await asyncio.wait_for(message.delete(), timeout=5.0)
        return True
    except Exception:
        return False

async def safe_send_message(bot, chat_id: int, text: str, **kwargs) -> Optional[object]:
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                bot.send_message(chat_id=chat_id, text=text, **kwargs),
                timeout=30.0)
        except (TimedOut, NetworkError):
            if attempt < 2: await asyncio.sleep(1)
        except Exception as e:
            log.error(f"Send message error: {e}")
            return None
    return None

async def safe_send_photo(bot, chat_id: int, photo, **kwargs) -> Optional[object]:
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                bot.send_photo(chat_id=chat_id, photo=photo, **kwargs),
                timeout=60.0)
        except (TimedOut, NetworkError):
            if attempt < 2: await asyncio.sleep(1)
        except Exception as e:
            log.error(f"Send photo error: {e}")
            return None
    return None

# Short aliases for the helpers
async def cb_answer(query) -> bool:
    return await safe_answer_callback(query)

async def del_msg(message) -> bool:
    return await safe_delete_message(message)

async def send_msg(bot, chat_id: int, text: str, **kwargs) -> Optional[object]:
    return await safe_send_message(bot, chat_id, text, **kwargs)

async def send_photo(bot, chat_id: int, photo, **kwargs) -> Optional[object]:
    return await safe_send_photo(bot, chat_id, photo, **kwargs)


# ════════════════════════════════════════════════════════════════
#  OANDA API
# ════════════════════════════════════════════════════════════════

async def fetch_candles(session: aiohttp.ClientSession, instrument: str,
                        granularity: str, count: int) -> List[Candle]:
    url    = f"{API_BASE}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        async with session.get(url, headers=HEADERS, params=params,
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            candles = []
            for c in data.get("candles", []):
                if "mid" in c:
                    m = c["mid"]
                    candles.append(Candle(
                        time=c.get("time", ""),
                        open=float(m["o"]), high=float(m["h"]),
                        low=float(m["l"]),  close=float(m["c"]),
                        volume=max(float(c.get("volume", 1)), 1),
                        complete=c.get("complete", True)
                    ))
            return candles
    except Exception as e:
        log.error(f"Fetch candles error {instrument}: {e}")
        return []

async def fetch_current_price(session: aiohttp.ClientSession,
                              instrument: str) -> Optional[float]:
    url    = f"{API_BASE}/v3/instruments/{instrument}/candles"
    params = {"granularity": "M1", "count": 1, "price": "M"}
    try:
        async with session.get(url, headers=HEADERS, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200: return None
            data = await resp.json()
            candles = data.get("candles", [])
            if candles and "mid" in candles[0]:
                return float(candles[0]["mid"]["c"])
    except Exception as e:
        log.error(f"Fetch price error {instrument}: {e}")
    return None

async def fetch_order_book(session: aiohttp.ClientSession,
                           instrument: str) -> Optional[Dict]:
    url = f"{API_BASE}/v3/instruments/{instrument}/orderBook"
    try:
        async with session.get(url, headers=HEADERS,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200: return None
            return (await resp.json()).get("orderBook")
    except Exception:
        return None

async def fetch_position_book(session: aiohttp.ClientSession,
                              instrument: str) -> Optional[Dict]:
    url = f"{API_BASE}/v3/instruments/{instrument}/positionBook"
    try:
        async with session.get(url, headers=HEADERS,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200: return None
            return (await resp.json()).get("positionBook")
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  CURRENCY STRENGTH CACHE
# ════════════════════════════════════════════════════════════════

async def calculate_currency_strength(
        session: aiohttp.ClientSession,
        timeframe: str = "H1",
        periods: int = 24) -> Dict[str, CurrencyStrength]:
    changes: Dict[str, List[float]] = {c: [] for c in CURRENCIES}
    for pair in FOREX_PAIRS[:14]:
        try:
            candles = await fetch_candles(session, pair, timeframe, periods + 1)
            if len(candles) >= 2:
                parts = pair.split("_")
                if len(parts) == 2:
                    base, quote = parts
                    op, cl = candles[0].open, candles[-1].close
                    if op > 0:
                        pct = ((cl - op) / op) * 100
                        if base  in changes: changes[base].append(pct)
                        if quote in changes: changes[quote].append(-pct)
            await asyncio.sleep(0.02)
        except Exception:
            continue
    strengths = {}
    for currency in CURRENCIES:
        avg   = (sum(changes[currency]) / len(changes[currency])
                 if changes[currency] else 0.0)
        trend = ("STRONG" if avg > 0.1 else
                 "WEAK"   if avg < -0.1 else "NEUTRAL")
        strengths[currency] = CurrencyStrength(
            currency=currency, strength=avg, trend=trend, rank=0)
    for i, cs in enumerate(
            sorted(strengths.values(), key=lambda x: x.strength, reverse=True)):
        strengths[cs.currency].rank = i + 1
    return strengths


class CurrencyStrengthCache:
    def __init__(self, ttl_seconds: int = STRENGTH_CACHE_TTL):
        self.cache: Optional[Dict] = None
        self.last_update: Optional[datetime] = None
        self.ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, session: aiohttp.ClientSession) -> Dict:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if (self.cache is None or self.last_update is None or
                    (now - self.last_update).total_seconds() > self.ttl):
                self.cache = await calculate_currency_strength(session, "H1", 24)
                self.last_update = now
            return self.cache

STRENGTH_CACHE = CurrencyStrengthCache()


# ════════════════════════════════════════════════════════════════
#  LIQUIDITY ZONE DETECTOR
# ════════════════════════════════════════════════════════════════

class LiquidityZoneDetector:

    def detect_all(self, candles: List[Candle],
                   instrument: str) -> List[LiquidityZone]:
        zones = []
        if len(candles) < 20:
            return zones
        zones.extend(self._detect_demand_supply(candles, instrument))
        zones.extend(self._detect_equal_levels(candles, instrument))
        zones.extend(self._detect_fair_value_gaps(candles, instrument))
        zones.extend(self._detect_order_blocks(candles, instrument))
        zones = self._merge_nearby_zones(zones, instrument)
        return sorted(zones, key=lambda z: z.strength, reverse=True)[:12]

    # keep 'detect' as alias
    def detect(self, candles: List[Candle], instrument: str) -> List[LiquidityZone]:
        return self.detect_all(candles, instrument)

    def _detect_demand_supply(self, candles: List[Candle],
                               instrument: str) -> List[LiquidityZone]:
        zones = []
        pip   = pip_value(instrument)
        for i in range(2, len(candles) - 2):
            c = candles[i]
            if c.range > 0 and c.body_abs / c.range > 0.6:
                prev = candles[i - 1]
                if c.is_bullish and not prev.is_bullish:
                    zone_low  = min(prev.low, c.low)
                    zone_high = max(prev.high, c.open)
                    strength  = min(100, (c.body_abs / pip) * 0.5)
                    zones.append(LiquidityZone(
                        price=zone_low + (zone_high - zone_low) / 2,
                        zone_type="DEMAND", strength=strength,
                        description=(f"Demand zone — institutional buying base at "
                                     f"{format_price(zone_low, instrument)}"),
                        upper=zone_high, lower=zone_low, created_at=c.time))
                elif not c.is_bullish and prev.is_bullish:
                    zone_low  = min(prev.low, c.open)
                    zone_high = max(prev.high, c.high)
                    strength  = min(100, (c.body_abs / pip) * 0.5)
                    zones.append(LiquidityZone(
                        price=zone_low + (zone_high - zone_low) / 2,
                        zone_type="SUPPLY", strength=strength,
                        description=(f"Supply zone — institutional selling base at "
                                     f"{format_price(zone_high, instrument)}"),
                        upper=zone_high, lower=zone_low, created_at=c.time))
        return zones

    def _detect_equal_levels(self, candles: List[Candle],
                              instrument: str) -> List[LiquidityZone]:
        zones  = []
        pip    = pip_value(instrument)
        thresh = pip * 3
        highs  = [(i, c.high) for i, c in enumerate(candles)]
        lows   = [(i, c.low)  for i, c in enumerate(candles)]
        for i in range(len(highs)):
            cluster = [highs[j] for j in range(i + 1, min(i + 20, len(highs)))
                       if abs(highs[j][1] - highs[i][1]) <= thresh]
            if len(cluster) >= 2:
                avg_price = sum(h[1] for h in cluster) / len(cluster)
                zones.append(LiquidityZone(
                    price=avg_price, zone_type="EQUAL_HIGHS",
                    strength=min(100, len(cluster) * 25),
                    description=(f"Equal highs liquidity pool at "
                                 f"{format_price(avg_price, instrument)} "
                                 f"— stop-hunts likely"),
                    upper=avg_price + thresh, lower=avg_price - thresh,
                    created_at=candles[i].time))
                break
        for i in range(len(lows)):
            cluster = [lows[j] for j in range(i + 1, min(i + 20, len(lows)))
                       if abs(lows[j][1] - lows[i][1]) <= thresh]
            if len(cluster) >= 2:
                avg_price = sum(l[1] for l in cluster) / len(cluster)
                zones.append(LiquidityZone(
                    price=avg_price, zone_type="EQUAL_LOWS",
                    strength=min(100, len(cluster) * 25),
                    description=(f"Equal lows liquidity pool at "
                                 f"{format_price(avg_price, instrument)} "
                                 f"— stop-hunts likely"),
                    upper=avg_price + thresh, lower=avg_price - thresh,
                    created_at=candles[i].time))
                break
        return zones

    def _detect_fair_value_gaps(self, candles: List[Candle],
                                 instrument: str) -> List[LiquidityZone]:
        zones = []
        for i in range(1, len(candles) - 1):
            prev = candles[i - 1]
            curr = candles[i]
            nxt  = candles[i + 1]
            if nxt.low > prev.high:
                gap_size = nxt.low - prev.high
                strength = min(100, gap_size / pip_value(instrument) * 2)
                if strength > 20:
                    mid = prev.high + gap_size / 2
                    zones.append(LiquidityZone(
                        price=mid, zone_type="FAIR_VALUE_GAP", strength=strength,
                        description=(f"Bullish Fair Value Gap "
                                     f"{format_price(prev.high, instrument)}-"
                                     f"{format_price(nxt.low, instrument)} "
                                     f"— price likely to fill"),
                        upper=nxt.low, lower=prev.high, created_at=curr.time))
            elif nxt.high < prev.low:
                gap_size = prev.low - nxt.high
                strength = min(100, gap_size / pip_value(instrument) * 2)
                if strength > 20:
                    mid = nxt.high + gap_size / 2
                    zones.append(LiquidityZone(
                        price=mid, zone_type="FAIR_VALUE_GAP", strength=strength,
                        description=(f"Bearish Fair Value Gap "
                                     f"{format_price(nxt.high, instrument)}-"
                                     f"{format_price(prev.low, instrument)} "
                                     f"— price likely to fill"),
                        upper=prev.low, lower=nxt.high, created_at=curr.time))
        return zones[:4]

    def _detect_order_blocks(self, candles: List[Candle],
                              instrument: str) -> List[LiquidityZone]:
        zones = []
        for i in range(1, len(candles) - 3):
            c    = candles[i]
            nxt1 = candles[i + 1]
            nxt2 = candles[i + 2]
            if (not c.is_bullish and nxt1.is_bullish and nxt2.is_bullish
                    and nxt1.body_abs > c.body_abs * 1.5):
                zones.append(LiquidityZone(
                    price=c.low + (c.high - c.low) / 2,
                    zone_type="DEMAND",
                    strength=min(100, nxt1.body_abs / pip_value(instrument) * 0.3),
                    description=(f"Bullish order block at "
                                 f"{format_price(c.low, instrument)}-"
                                 f"{format_price(c.high, instrument)}"),
                    upper=c.high, lower=c.low, created_at=c.time))
            elif (c.is_bullish and not nxt1.is_bullish and not nxt2.is_bullish
                  and nxt1.body_abs > c.body_abs * 1.5):
                zones.append(LiquidityZone(
                    price=c.low + (c.high - c.low) / 2,
                    zone_type="SUPPLY",
                    strength=min(100, nxt1.body_abs / pip_value(instrument) * 0.3),
                    description=(f"Bearish order block at "
                                 f"{format_price(c.low, instrument)}-"
                                 f"{format_price(c.high, instrument)}"),
                    upper=c.high, lower=c.low, created_at=c.time))
        return zones[:4]

    def _merge_nearby_zones(self, zones: List[LiquidityZone],
                            instrument: str) -> List[LiquidityZone]:
        if not zones:
            return zones
        pip    = pip_value(instrument)
        thresh = pip * 10
        merged = []
        used   = set()
        for i, z in enumerate(zones):
            if i in used:
                continue
            group = [z]
            for j, z2 in enumerate(zones[i+1:], i+1):
                if j not in used and abs(z2.price - z.price) < thresh:
                    group.append(z2)
                    used.add(j)
            best = max(group, key=lambda x: x.strength)
            merged.append(best)
            used.add(i)
        return merged

    def format_zones_message(self, zones: List[LiquidityZone],
                             current_price: float, instrument: str) -> str:
        if not zones:
            return "No significant liquidity zones detected."
        msg   = ""
        above = [z for z in zones if z.price > current_price]
        below = [z for z in zones if z.price <= current_price]
        if above:
            msg += "<b>🔴 Liquidity Above Price (Resistance):</b>\n"
            for z in sorted(above, key=lambda x: x.price)[:4]:
                icon = ("🧊" if z.zone_type == "EQUAL_HIGHS" else
                        "⬆️" if z.zone_type == "SUPPLY" else
                        "🌀" if z.zone_type == "FAIR_VALUE_GAP" else "📦")
                dist = pips_diff(instrument, z.price - current_price)
                msg += (f"{icon} {format_price(z.price, instrument)} "
                        f"[{z.zone_type.replace('_',' ')}] "
                        f"+{dist:.0f} pips | Str: {z.strength:.0f}%\n"
                        f"   <i>{z.description}</i>\n")
        if below:
            msg += "\n<b>🟢 Liquidity Below Price (Support):</b>\n"
            for z in sorted(below, key=lambda x: x.price, reverse=True)[:4]:
                icon = ("🧊" if z.zone_type == "EQUAL_LOWS" else
                        "⬇️" if z.zone_type == "DEMAND" else
                        "🌀" if z.zone_type == "FAIR_VALUE_GAP" else "📦")
                dist = pips_diff(instrument, current_price - z.price)
                msg += (f"{icon} {format_price(z.price, instrument)} "
                        f"[{z.zone_type.replace('_',' ')}] "
                        f"-{dist:.0f} pips | Str: {z.strength:.0f}%\n"
                        f"   <i>{z.description}</i>\n")
        return msg


LIQUIDITY_DETECTOR = LiquidityZoneDetector()


# ════════════════════════════════════════════════════════════════
#  ADVANCED ORDER FLOW ENGINE
# ════════════════════════════════════════════════════════════════

class AdvancedOrderFlow:

    def analyze_all(self, candles: List[Candle],
                    order_book: Optional[Dict] = None) -> AdvancedOrderFlowData:
        if not candles or len(candles) < 20:
            return AdvancedOrderFlowData()
        vpin            = self._vpin_academic(candles)
        kyle_lambda     = self._kyle_lambda(candles)
        toxicity        = self._toxicity(candles)
        amihud          = self._amihud(candles)
        roll            = self._roll_spread(candles)
        depth_imbalance = 0.0
        iceberg_orders  = {}
        if order_book:
            depth_imbalance = self._depth_imbalance(order_book)
            iceberg_orders  = self._detect_icebergs(order_book)
        absorption   = self._absorption_ratio(candles)
        aggressor    = self._aggressor_side(candles)
        clustering   = self._volume_clustering(candles)
        arrival_rate = self._arrival_rate(candles)
        eff_spread   = self._effective_spread(candles)
        real_spread  = self._realized_spread(candles)
        price_impact = self._price_impact(candles)
        vpin_level   = self._interp_vpin(vpin)
        tox_level    = self._interp_toxicity(toxicity)
        liq_level    = self._interp_liquidity(amihud)
        depth_bias   = self._interp_depth(depth_imbalance)
        kyle_interp  = self._interp_kyle(kyle_lambda)
        smart_money  = self._smart_money_score(vpin, toxicity, kyle_lambda,
                                                depth_imbalance)
        informed_sig = self._informed_signal(candles, vpin, toxicity, aggressor)
        inst_score   = self._institutional_score(vpin, kyle_lambda, toxicity,
                                                  absorption, clustering)
        return AdvancedOrderFlowData(
            vpin=vpin, vpin_level=vpin_level,
            kyle_lambda=kyle_lambda,
            kyle_lambda_interpretation=kyle_interp,
            toxicity=toxicity, toxicity_level=tox_level,
            amihud_illiquidity=amihud, liquidity_level=liq_level,
            market_depth_imbalance=depth_imbalance, depth_bias=depth_bias,
            iceberg_orders=iceberg_orders, iceberg_count=len(iceberg_orders),
            smart_money_activity=smart_money,
            informed_trader_signal=informed_sig,
            effective_spread=eff_spread, realized_spread=real_spread,
            price_impact=price_impact, trade_arrival_rate=arrival_rate,
            volume_clustering=clustering, absorption_ratio=absorption,
            aggressor_side=aggressor,
            institutional_flow_score=inst_score,
            roll_spread=roll
        )

    # keep 'analyze' as alias
    def analyze(self, candles: List[Candle],
                order_book: Optional[Dict] = None) -> AdvancedOrderFlowData:
        return self.analyze_all(candles, order_book)

    def _vpin_academic(self, candles: List[Candle],
                        num_buckets: int = 50) -> float:
        tv = sum(c.volume for c in candles)
        bs = tv / num_buckets
        if bs == 0: return 0.0
        bb, sb, cb, cs, cv = [], [], 0.0, 0.0, 0.0
        for c in candles:
            cb += c.buy_volume; cs += c.sell_volume; cv += c.volume
            while cv >= bs:
                r = bs / cv
                bb.append(cb * r); sb.append(cs * r)
                cb -= cb * r; cs -= cs * r; cv -= bs
        if not bb: return 0.0
        return min(1.0, sum(abs(b - s) for b, s in zip(bb, sb)) / (len(bb) * bs))

    def _kyle_lambda(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        pc = [c.close - c.open for c in candles]
        sv = [c.delta for c in candles]
        vv = np.var(sv)
        if vv == 0: return 0.0
        try:
            cov = np.cov(pc, sv)
            return float(cov[0][1] / vv) if cov.shape == (2, 2) else 0.0
        except Exception:
            return 0.0

    def _toxicity(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        scores = []
        for i in range(1, len(candles)):
            p = candles[i - 1]; c = candles[i]
            exp = (p.delta / p.volume * p.range) if p.volume > 0 else 0
            if p.range > 0:
                scores.append(
                    min(1.0, abs((c.close - p.close) - exp) / p.range))
        return float(np.mean(scores)) if scores else 0.0

    def _amihud(self, candles: List[Candle]) -> float:
        if len(candles) < 5: return 0.0
        illiq = []
        for i in range(1, len(candles)):
            pc, cc, vol = (candles[i-1].close, candles[i].close,
                           candles[i].volume)
            if pc > 0 and vol > 0:
                illiq.append(abs((cc - pc) / pc) / vol)
        return float(np.mean(illiq) * 1e9) if illiq else 0.0

    def _roll_spread(self, candles: List[Candle]) -> float:
        if len(candles) < 20: return 0.0
        ch = [candles[i].close - candles[i-1].close
              for i in range(1, len(candles))]
        if len(ch) < 2: return 0.0
        m   = float(np.mean(ch))
        cov = float(np.mean(
            [(ch[i]-m)*(ch[i-1]-m) for i in range(1, len(ch))]))
        return float(2 * np.sqrt(-cov)) if cov < 0 else 0.0

    def _detect_icebergs(self, order_book: Optional[Dict]) -> Dict:
        if not order_book or "buckets" not in order_book: return {}
        fm = {}
        for b in order_book["buckets"]:
            p = float(b["price"])
            t = (float(b.get("longCountPercent", 0)) +
                 float(b.get("shortCountPercent", 0)))
            fm[p] = fm.get(p, 0) + t
        if not fm: return {}
        vals = list(fm.values())
        avg  = float(np.mean(vals))
        std  = float(np.std(vals)) if len(vals) > 1 else 0.0
        return {p: f for p, f in fm.items() if f > avg + 2 * std}

    def _depth_imbalance(self, order_book: Optional[Dict]) -> float:
        if not order_book or "buckets" not in order_book: return 0.0
        bp = float(order_book.get("price", 0))
        bid = ask = 0.0
        for b in order_book["buckets"]:
            p = float(b["price"])
            t = (float(b.get("longCountPercent", 0)) +
                 float(b.get("shortCountPercent", 0)))
            bid += t if p < bp else 0
            ask += t if p >= bp else 0
        td = bid + ask
        return max(-1.0, min(1.0, (bid - ask) / td)) if td > 0 else 0.0

    def _effective_spread(self, candles: List[Candle]) -> float:
        sp = [(c.high - c.low) / c.close * 100
              for c in candles if c.close > 0]
        return float(np.mean(sp)) if sp else 0.0

    def _realized_spread(self, candles: List[Candle], h: int = 5) -> float:
        if len(candles) < h + 5: return 0.0
        rs = []
        for i in range(h, len(candles)):
            im = (candles[i-h].high + candles[i-h].low) / 2
            fm = (candles[i].high + candles[i].low) / 2
            d  = 1 if candles[i-h].is_bullish else -1
            if im > 0: rs.append((fm - im) / im * 100 * d)
        return float(np.mean(rs)) if rs else 0.0

    def _price_impact(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        imp = [abs(candles[i].close - candles[i-5].close) /
               candles[i-5].close * 100
               for i in range(5, len(candles))
               if candles[i-5].close > 0]
        return float(np.mean(imp)) if imp else 0.0

    def _arrival_rate(self, candles: List[Candle]) -> float:
        if not candles: return 0.0
        vols = [c.volume for c in candles]
        avg  = float(np.mean(vols))
        std  = float(np.std(vols)) if len(vols) > 1 else 1.0
        rec  = float(np.mean(vols[-5:])) if len(vols) >= 5 else avg
        return float((rec - avg) / std) if std > 0 else 0.0

    def _volume_clustering(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        lo = min(c.low for c in candles)
        hi = max(c.high for c in candles)
        rng = hi - lo
        if rng == 0: return 0.0
        nz = 10; zs = rng / nz; zv = [0.0] * nz
        for c in candles:
            idx = min(int(((c.high + c.low) / 2 - lo) / zs), nz - 1)
            zv[idx] += c.volume
        tv = sum(zv)
        if tv == 0: return 0.0
        sh = [v / tv for v in zv]
        hf = sum(s**2 for s in sh)
        mn = 1.0 / nz
        return float(max(0.0, min(1.0, (hf - mn) / (1.0 - mn))))

    def _absorption_ratio(self, candles: List[Candle]) -> float:
        if not candles: return 1.0
        ba = sa = 0.0
        for c in candles:
            d = c.range + 0.0001
            ba += c.lower_wick * c.volume / d
            sa += c.upper_wick * c.volume / d
        if sa == 0: return 2.0 if ba > 0 else 1.0
        return float(ba / sa)

    def _aggressor_side(self, candles: List[Candle]) -> str:
        if len(candles) < 5: return "NEUTRAL"
        bv = bev = 0.0
        for c in candles[-10:]:
            if c.range > 0 and c.body_abs / c.range > 0.6:
                if c.is_bullish: bv += c.volume
                else:            bev += c.volume
        t = bv + bev
        if t == 0: return "NEUTRAL"
        bp = bv / t
        if bp > 0.6: return "BUYERS"
        if bp < 0.4: return "SELLERS"
        return "NEUTRAL"

    def _interp_vpin(self, v: float) -> str:
        if v >= 0.7:  return "VERY HIGH 🔥"
        if v >= 0.5:  return "HIGH ⚠️"
        if v >= 0.3:  return "MODERATE"
        if v >= 0.15: return "LOW"
        return "MINIMAL"

    def _interp_toxicity(self, t: float) -> str:
        if t >= 0.7: return "VERY HIGH 🔥"
        if t >= 0.5: return "HIGH ⚠️"
        if t >= 0.3: return "MODERATE"
        return "LOW"

    def _interp_liquidity(self, a: float) -> str:
        if a >= 100: return "VERY ILLIQUID ⚠️"
        if a >= 50:  return "ILLIQUID"
        if a >= 20:  return "MODERATE"
        return "LIQUID ✅"

    def _interp_depth(self, d: float) -> str:
        if d >= 0.3:  return "STRONG BID PRESSURE 🟢"
        if d >= 0.1:  return "BID PRESSURE"
        if d <= -0.3: return "STRONG ASK PRESSURE 🔴"
        if d <= -0.1: return "ASK PRESSURE"
        return "BALANCED"

    def _interp_kyle(self, kl: float) -> str:
        a = abs(kl)
        if a >= 0.0001:  return "HIGH IMPACT (Informed trading or illiquid)"
        if a >= 0.00005: return "MODERATE IMPACT"
        if a >= 0.00001: return "LOW IMPACT"
        return "MINIMAL IMPACT (Efficient market)"

    def _smart_money_score(self, vpin, toxicity, kl, depth) -> str:
        s = 0
        if vpin >= 0.5:       s += 30
        elif vpin >= 0.3:     s += 15
        if toxicity >= 0.5:   s += 25
        elif toxicity >= 0.3: s += 12
        if abs(kl) >= 0.00005:  s += 25
        elif abs(kl) >= 0.00001: s += 12
        if abs(depth) >= 0.3:    s += 20
        elif abs(depth) >= 0.15: s += 10
        if s >= 70: return "VERY HIGH 🔥"
        if s >= 50: return "HIGH ⚠️"
        if s >= 30: return "MODERATE"
        if s >= 15: return "LOW"
        return "MINIMAL"

    def _informed_signal(self, candles, vpin, toxicity, aggressor) -> str:
        if vpin < 0.3 and toxicity < 0.3: return "NEUTRAL"
        cvd = sum(c.delta for c in candles[-10:])
        if aggressor == "BUYERS"  and cvd > 0: return "BULLISH 🟢"
        if aggressor == "SELLERS" and cvd < 0: return "BEARISH 🔴"
        if cvd > 0: return "LEANING BULLISH"
        if cvd < 0: return "LEANING BEARISH"
        return "NEUTRAL"

    def _institutional_score(self, vpin, kl, toxicity,
                              absorption, clustering) -> float:
        s  = min(25.0, vpin * 35)
        s += min(20.0, abs(kl) * 100000 * 20)
        s += min(20.0, toxicity * 30)
        s += min(15.0, clustering * 20)
        if absorption > 1.5 or absorption < 0.67:   s += 20
        elif absorption > 1.2 or absorption < 0.83: s += 10
        return min(100.0, s)


ADVANCED_OF = AdvancedOrderFlow()


# ════════════════════════════════════════════════════════════════
#  BASIC ORDER FLOW ENGINE
# ════════════════════════════════════════════════════════════════

class OrderFlowEngine:

    def analyze(self, candles: List[Candle]) -> OrderFlowData:
        if not candles or len(candles) < 20:
            return OrderFlowData()
        bv = sum(c.buy_volume for c in candles)
        sv = sum(c.sell_volume for c in candles)
        tv = bv + sv
        cvd    = sum(c.delta for c in candles)
        vwap_n = sum((c.high+c.low+c.close)/3*c.volume for c in candles)
        vwap_d = sum(c.volume for c in candles)
        vwap   = vwap_n / vwap_d if vwap_d > 0 else candles[-1].close
        rd5  = sum(c.delta for c in candles[-5:])
        rd10 = sum(c.delta for c in candles[-10:])
        rd20 = sum(c.delta for c in candles[-20:])
        d5f  = (sum(c.delta for c in candles[-10:-5])
                if len(candles) >= 10 else 0.0)
        mom  = rd5 - d5f
        cv   = []; r = 0.0
        for c in candles[-20:]:
            r += c.delta; cv.append(r)
        cvd_slope = ((sum(cv[10:])/max(1,len(cv)-10)) -
                     sum(cv[:10])/10) if len(cv) >= 10 else 0.0
        if len(candles) >= 20:
            v1 = sum(c.volume for c in candles[-20:-10]) / 10
            v2 = sum(c.volume for c in candles[-10:])    / 10
            vt = ((v2-v1)/v1*100) if v1 > 0 else 0.0
        else:
            vt = 0.0
        if len(candles) >= 15:
            vf = sum(c.volume for c in candles[-15:-10]) / 5
            vm = sum(c.volume for c in candles[-10:-5])  / 5
            vl = sum(c.volume for c in candles[-5:])     / 5
            va = (vl - vm) - (vm - vf)
        else:
            va = 0.0
        avg_v = tv / len(candles) if candles else 1.0
        rec_v = sum(c.volume for c in candles[-3:]) / 3
        bc    = rec_v > avg_v*2 and rd5 > 0 and mom < 0
        sc    = rec_v > avg_v*2 and rd5 < 0 and mom > 0
        return OrderFlowData(
            buy_volume=bv, sell_volume=sv, total_volume=tv,
            cvd=cvd, delta=rd20,
            buy_pct=(bv/tv*100) if tv > 0 else 50.0,
            sell_pct=(sv/tv*100) if tv > 0 else 50.0,
            imbalance=((bv-sv)/tv*100) if tv > 0 else 0.0,
            price=candles[-1].close, vwap=vwap,
            delta_momentum=mom, volume_trend=vt,
            recent_delta_5=rd5, recent_delta_10=rd10, recent_delta_20=rd20,
            cvd_slope=cvd_slope, volume_acceleration=va,
            buying_climax=bc, selling_climax=sc
        )

    def evaluate_confluence(self, order_flow: OrderFlowData,
                            volume_profile: Optional[VolumeProfile],
                            order_book: Optional[OrderBookData],
                            position_book: Optional[PositionBookData],
                            currency_strength: Optional[Dict],
                            advanced_flow: Optional[AdvancedOrderFlowData],
                            instrument: str,
                            current_price: float,
                            liquidity_zones: Optional[List[LiquidityZone]] = None
                            ) -> PredictionResult:
        factors = []
        tv = order_flow.total_volume

        def f(name, direction, strength, desc):
            factors.append(ConfluenceFactor(name, direction, strength, desc))

        # CVD
        if order_flow.cvd > 0:
            f("CVD", "BULLISH",
              min(100, abs(order_flow.cvd) / (tv/100+1) * 10),
              f"Cumulative buying pressure ({format_signed(order_flow.cvd)})")
        elif order_flow.cvd < 0:
            f("CVD", "BEARISH",
              min(100, abs(order_flow.cvd) / (tv/100+1) * 10),
              f"Cumulative selling pressure ({format_signed(order_flow.cvd)})")
        else:
            f("CVD", "NEUTRAL", 0, "CVD is flat")

        # Imbalance
        if order_flow.imbalance > MIN_IMBALANCE_PCT:
            f("Imbalance", "BULLISH",
              min(100, order_flow.imbalance * 3),
              f"Buyers dominating ({order_flow.imbalance:+.1f}%)")
        elif order_flow.imbalance < -MIN_IMBALANCE_PCT:
            f("Imbalance", "BEARISH",
              min(100, abs(order_flow.imbalance) * 3),
              f"Sellers dominating ({order_flow.imbalance:+.1f}%)")
        else:
            f("Imbalance", "NEUTRAL", 0, "Balanced order flow")

        # Delta Momentum
        mt = tv / 50 if tv > 0 else 1
        if order_flow.delta_momentum > mt:
            f("Delta Momentum", "BULLISH",
              min(100, order_flow.delta_momentum / mt * 30),
              "Buying pressure accelerating")
        elif order_flow.delta_momentum < -mt:
            f("Delta Momentum", "BEARISH",
              min(100, abs(order_flow.delta_momentum) / mt * 30),
              "Selling pressure accelerating")
        else:
            f("Delta Momentum", "NEUTRAL", 0, "No momentum acceleration")

        # CVD Slope
        st = tv / 100 if tv > 0 else 1
        if order_flow.cvd_slope > st:
            f("CVD Trend", "BULLISH",
              min(100, order_flow.cvd_slope / st * 20),
              "CVD trending upward")
        elif order_flow.cvd_slope < -st:
            f("CVD Trend", "BEARISH",
              min(100, abs(order_flow.cvd_slope) / st * 20),
              "CVD trending downward")
        else:
            f("CVD Trend", "NEUTRAL", 0, "CVD is flat")

        # Recent Delta
        tv100 = tv / 100 + 1
        if order_flow.recent_delta_5 > 0:
            f("Recent Flow", "BULLISH",
              min(100, order_flow.recent_delta_5 / tv100 * 15),
              "Recent candles show buying")
        elif order_flow.recent_delta_5 < 0:
            f("Recent Flow", "BEARISH",
              min(100, abs(order_flow.recent_delta_5) / tv100 * 15),
              "Recent candles show selling")
        else:
            f("Recent Flow", "NEUTRAL", 0, "Recent flow neutral")

        # Volume
        if order_flow.volume_trend > 15:
            d = "BULLISH" if order_flow.cvd > 0 else "BEARISH"
            f("Volume", d,
              min(100, order_flow.volume_trend * 2),
              f"Volume increasing ({order_flow.volume_trend:+.1f}%)")
        elif order_flow.volume_trend < -15:
            f("Volume", "NEUTRAL", 20,
              "Volume declining - weak conviction")
        else:
            f("Volume", "NEUTRAL", 0, "Volume stable")

        # VWAP
        vd = ((current_price - order_flow.vwap) / order_flow.vwap * 100
              if order_flow.vwap > 0 else 0)
        if vd > 0.05:
            f("VWAP", "BULLISH", min(100, vd * 50),
              "Price above VWAP (bullish bias)")
        elif vd < -0.05:
            f("VWAP", "BEARISH", min(100, abs(vd) * 50),
              "Price below VWAP (bearish bias)")
        else:
            f("VWAP", "NEUTRAL", 0, "Price at VWAP")

        # Position Book
        if position_book:
            if position_book.skew > 20:
                f("Contrarian", "BEARISH",
                  min(100, position_book.skew * 2),
                  f"Retail {position_book.long_pct:.0f}% long "
                  f"(contrarian short)")
            elif position_book.skew < -20:
                f("Contrarian", "BULLISH",
                  min(100, abs(position_book.skew) * 2),
                  f"Retail {position_book.short_pct:.0f}% short "
                  f"(contrarian long)")
            else:
                f("Contrarian", "NEUTRAL", 0,
                  "No extreme retail positioning")

        # Order Book
        if order_book:
            if order_book.breakout_bias == "UPWARD_PRESSURE":
                f("Order Book", "BULLISH", 60,
                  "Pending orders favour upside")
            elif order_book.breakout_bias == "DOWNWARD_PRESSURE":
                f("Order Book", "BEARISH", 60,
                  "Pending orders favour downside")
            else:
                f("Order Book", "NEUTRAL", 0, "Balanced order book")

        # Currency Strength
        if currency_strength and "_" in instrument:
            parts = instrument.split("_")
            if len(parts) == 2:
                bs = currency_strength.get(parts[0])
                qs = currency_strength.get(parts[1])
                if bs and qs:
                    diff = bs.strength - qs.strength
                    if diff > 0.1:
                        f("Currency Strength", "BULLISH",
                          min(100, diff * 30),
                          f"{parts[0]} stronger than {parts[1]}")
                    elif diff < -0.1:
                        f("Currency Strength", "BEARISH",
                          min(100, abs(diff) * 30),
                          f"{parts[1]} stronger than {parts[0]}")
                    else:
                        f("Currency Strength", "NEUTRAL", 0,
                          "Currencies equally strong")

        # Climax
        if order_flow.buying_climax:
            f("Climax Warning", "BEARISH", 40,
              "Buying climax - exhaustion risk")
        if order_flow.selling_climax:
            f("Climax Warning", "BULLISH", 40,
              "Selling climax - exhaustion risk")

        # Advanced Flow
        if advanced_flow:
            if advanced_flow.vpin >= 0.5:
                d = "BULLISH" if order_flow.cvd > 0 else "BEARISH"
                f("VPIN (Smart Money)", d,
                  min(100, advanced_flow.vpin * 80),
                  f"Smart money active ({advanced_flow.vpin_level})")
            if "BULLISH" in advanced_flow.informed_trader_signal:
                f("Informed Traders", "BULLISH", 70,
                  "Informed traders buying")
            elif "BEARISH" in advanced_flow.informed_trader_signal:
                f("Informed Traders", "BEARISH", 70,
                  "Informed traders selling")
            if advanced_flow.market_depth_imbalance > 0.2:
                f("Order Book Depth", "BULLISH",
                  min(100, advanced_flow.market_depth_imbalance * 150),
                  f"Strong bid support ({advanced_flow.depth_bias})")
            elif advanced_flow.market_depth_imbalance < -0.2:
                f("Order Book Depth", "BEARISH",
                  min(100, abs(advanced_flow.market_depth_imbalance) * 150),
                  f"Strong ask pressure ({advanced_flow.depth_bias})")
            if advanced_flow.aggressor_side == "BUYERS":
                f("Aggressor", "BULLISH", 60,
                  "Buyers aggressively taking offers")
            elif advanced_flow.aggressor_side == "SELLERS":
                f("Aggressor", "BEARISH", 60,
                  "Sellers aggressively hitting bids")
            if advanced_flow.institutional_flow_score >= 60:
                d = "BULLISH" if order_flow.cvd > 0 else "BEARISH"
                f("Institutional Flow", d,
                  advanced_flow.institutional_flow_score,
                  f"High institutional activity "
                  f"({advanced_flow.institutional_flow_score:.0f}%)")
            if advanced_flow.absorption_ratio > 1.5:
                f("Absorption", "BULLISH",
                  min(100, (advanced_flow.absorption_ratio - 1) * 80),
                  "Buying absorption detected")
            elif advanced_flow.absorption_ratio < 0.67:
                f("Absorption", "BEARISH",
                  min(100, (1 - advanced_flow.absorption_ratio) * 80),
                  "Selling absorption detected")
            if advanced_flow.toxicity >= 0.5:
                opp = "BEARISH" if order_flow.cvd > 0 else "BULLISH"
                f("Flow Toxicity", opp, 30,
                  f"High toxicity ({advanced_flow.toxicity_level}) "
                  f"- reversal risk")

        # Liquidity Zone alignment
        if liquidity_zones:
            nearby = [z for z in liquidity_zones
                      if pips_diff(instrument,
                                   abs(z.price - current_price)) < 50]
            demand = [z for z in nearby
                      if z.zone_type in ("DEMAND", "EQUAL_LOWS")
                      and z.price < current_price]
            supply = [z for z in nearby
                      if z.zone_type in ("SUPPLY", "EQUAL_HIGHS")
                      and z.price > current_price]
            if demand:
                f("Liquidity Zone", "BULLISH",
                  min(100, max(z.strength for z in demand)),
                  f"Price near demand zone / liquidity "
                  f"({format_price(demand[0].price, instrument)})")
            if supply:
                f("Liquidity Zone", "BEARISH",
                  min(100, max(z.strength for z in supply)),
                  f"Price near supply zone / liquidity "
                  f"({format_price(supply[0].price, instrument)})")

        bull_f = [x for x in factors if x.direction == "BULLISH"]
        bear_f = [x for x in factors if x.direction == "BEARISH"]
        neut_f = [x for x in factors if x.direction == "NEUTRAL"]
        bc, brc, nc = len(bull_f), len(bear_f), len(neut_f)
        mx = max(bc, brc)

        if mx >= 8:   quality = SetupQuality.A_PLUS
        elif mx >= 6: quality = SetupQuality.A
        elif mx >= 4: quality = SetupQuality.B
        elif mx >= 3: quality = SetupQuality.C
        else:         quality = SetupQuality.NO_TRADE

        if quality == SetupQuality.NO_TRADE:
            return PredictionResult(
                False, "NONE", quality, 0.0, factors, bc, brc, nc,
                ["Not enough factors aligned"])

        aligned  = bull_f if bc > brc else bear_f
        dir_val  = "LONG" if bc > brc else "SHORT"
        base     = 50 + len(aligned) * 4
        avg_str  = (sum(x.strength for x in aligned) / len(aligned)
                    if aligned else 0)
        conf_pen = (bc if dir_val == "SHORT" else brc) * 2
        bonus    = (avg_str / 10 +
                    (5 if advanced_flow and
                     advanced_flow.institutional_flow_score >= 50 else 0))
        confidence = min(95.0, max(35.0, base + bonus - conf_pen))
        reasons    = [x.description for x in aligned if x.strength > 30][:8]
        return PredictionResult(
            True, dir_val, quality, confidence, factors, bc, brc, nc, reasons)

    # Alias
    def confluence(self, *args, **kwargs) -> PredictionResult:
        return self.evaluate_confluence(*args, **kwargs)


OF_ENGINE = OrderFlowEngine()


# ════════════════════════════════════════════════════════════════
#  ANALYSIS HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def analyze_order_book(ob: Optional[Dict],
                       price: float) -> Optional[OrderBookData]:
    if not ob or "buckets" not in ob: return None
    bp = float(ob.get("price", price))
    al = bl = as_ = bs_ = 0.0
    for b in ob["buckets"]:
        p  = float(b["price"])
        lp = float(b["longCountPercent"])
        sp = float(b["shortCountPercent"])
        if p > bp: al += lp; as_ += sp
        else:      bl += lp; bs_ += sp
    tl, ts = al + bl, as_ + bs_
    ni      = tl - ts
    pd      = (al + as_) - (bl + bs_)
    bias    = ("UPWARD_PRESSURE"   if pd > 2 else
               "DOWNWARD_PRESSURE" if pd < -2 else "BALANCED")
    pdir    = ("BULLISH" if pd > 2 else
               "BEARISH" if pd < -2 else "NEUTRAL")
    tot     = tl + ts
    return OrderBookData(
        price=bp, total_longs=tl, total_shorts=ts,
        net_imbalance=ni, pending_delta=pd, breakout_bias=bias,
        long_pct=(tl/tot*100) if tot > 0 else 50.0,
        short_pct=(ts/tot*100) if tot > 0 else 50.0,
        pressure_direction=pdir)

def analyze_position_book(pb: Optional[Dict],
                          price: float) -> Optional[PositionBookData]:
    if not pb or "buckets" not in pb: return None
    bp = float(pb.get("price", price))
    tl = ts = trl = trs = 0.0
    for b in pb["buckets"]:
        p  = float(b["price"])
        lp = float(b["longCountPercent"])
        sp = float(b["shortCountPercent"])
        tl += lp; ts += sp
        if p > bp and ((p - bp) / bp * 100) > 0.5:
            trl += lp
        elif p < bp and p > 0 and ((bp - p) / p * 100) > 0.5:
            trs += sp
    tot  = tl + ts
    lp_  = (tl / tot * 100) if tot > 0 else 50.0
    sp_  = (ts / tot * 100) if tot > 0 else 50.0
    skew = lp_ - sp_
    tlp  = (trl / tot * 100) if tot > 0 else 0.0
    tsp  = (trs / tot * 100) if tot > 0 else 0.0
    tuw  = tlp + tsp
    ci   = abs(skew) * (tuw / 10)
    pain = bp * (1 + skew / 200)
    sqz  = ("HIGH"     if ci > 300 else
            "MODERATE" if ci > 150 else "LOW")
    ct   = ("BEARISH" if skew > 15 else
            "BULLISH" if skew < -15 else "NEUTRAL")
    return PositionBookData(
        price=bp, long_pct=lp_, short_pct=sp_, skew=skew,
        contrarian_signal=ct,
        underwater_longs=tl, underwater_shorts=ts,
        trapped_longs_pct=tlp, trapped_shorts_pct=tsp,
        total_underwater=tuw, crowded_trade_index=ci,
        pain_threshold=pain, squeeze_potential=sqz)

def calculate_volume_profile(candles: List[Candle],
                             num_levels: int = 24) -> Optional[VolumeProfile]:
    if not candles: return None
    hi, lo = (max(c.high for c in candles),
               min(c.low  for c in candles))
    rng = hi - lo
    if rng <= 0: return None
    step   = rng / num_levels
    levels = [{"price": lo + (i + 0.5) * step, "volume": 0.0}
              for i in range(num_levels)]
    for c in candles:
        li  = max(0, min(int((c.low  - lo) / step), num_levels - 1))
        hi2 = max(0, min(int((c.high - lo) / step), num_levels - 1))
        n   = hi2 - li + 1
        vpl = c.volume / n if n > 0 else c.volume
        for idx in range(li, hi2 + 1):
            levels[idx]["volume"] += vpl
    tv  = sum(lv["volume"] for lv in levels)
    poc = max(levels, key=lambda x: x["volume"])
    sl  = sorted(levels, key=lambda x: x["volume"], reverse=True)
    vav, vap = 0.0, []
    for lv in sl:
        vav += lv["volume"]; vap.append(lv["price"])
        if vav >= tv * 0.7: break
    vah = max(vap) if vap else hi
    val = min(vap) if vap else lo
    avg = tv / num_levels if num_levels > 0 else 1.0
    hvn = sorted([lv["price"] for lv in levels if lv["volume"] > avg * 1.5])
    lvn = sorted([lv["price"] for lv in levels if lv["volume"] < avg * 0.3])
    return VolumeProfile(poc=poc["price"], vah=vah, val=val,
                         hvn=hvn[:5], lvn=lvn[:5], total_volume=tv)

def identify_key_levels(candles: List[Candle],
                        volume_profile: Optional[VolumeProfile],
                        current_price: float,
                        instrument: str) -> List[Dict]:
    levels = []
    if not candles: return levels
    levels.append({"price": max(c.high for c in candles[-20:]),
                   "type": "RESISTANCE", "description": "Recent 20-bar high"})
    levels.append({"price": min(c.low  for c in candles[-20:]),
                   "type": "SUPPORT",    "description": "Recent 20-bar low"})
    if volume_profile:
        levels += [
            {"price": volume_profile.poc, "type": "POC",
             "description": "Point of Control"},
            {"price": volume_profile.vah, "type": "VAH",
             "description": "Value Area High"},
            {"price": volume_profile.val, "type": "VAL",
             "description": "Value Area Low"}
        ]
    ri = (0.5    if "JPY" in instrument else
          10.0   if "XAU" in instrument else
          100.0  if "NAS" in instrument else 0.005)
    levels.append({"price": round(current_price / ri) * ri,
                   "type": "ROUND", "description": "Psychological level"})
    levels.sort(key=lambda x: abs(x["price"] - current_price))
    return levels[:8]

def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return candles[-1].range if candles else 0.0001
    tr = [max(candles[i].high - candles[i].low,
              abs(candles[i].high - candles[i-1].close),
              abs(candles[i].low  - candles[i-1].close))
          for i in range(1, len(candles))]
    return sum(tr[-period:]) / period if tr else 0.0001


# ════════════════════════════════════════════════════════════════
#  ML LEARNING ENGINE
# ════════════════════════════════════════════════════════════════

class MLLearningEngine:

    def __init__(self):
        self.rf  = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=3, random_state=42, n_jobs=-1)
        self.gb  = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.08, random_state=42)
        self.lr  = LogisticRegression(
            max_iter=2000, random_state=42, C=1.0)
        self.scaler            = StandardScaler()
        self.is_trained        = False
        self.n_samples         = 0
        self.feature_importance: Dict[str, float] = {}
        self.accuracy_history:   List[float] = []
        self.win_rate_history:   List[float] = []
        self._load()

    def _load(self):
        if os.path.exists(ML_MODEL_FILE):
            try:
                with open(ML_MODEL_FILE, "rb") as f:
                    d = pickle.load(f)
                self.rf                 = d.get("rf",    self.rf)
                self.gb                 = d.get("gb",    self.gb)
                self.lr                 = d.get("lr",    self.lr)
                self.scaler             = d.get("scaler",self.scaler)
                self.is_trained         = d.get("is_trained",        False)
                self.n_samples          = d.get("n_samples",         0)
                self.feature_importance = d.get("feature_importance",{})
                self.accuracy_history   = d.get("accuracy_history",  [])
                self.win_rate_history   = d.get("win_rate_history",  [])
                log.info(f"ML loaded: {self.n_samples} samples, "
                         f"trained={self.is_trained}")
            except Exception as e:
                log.error(f"ML load error: {e}")

    def _save(self):
        try:
            with open(ML_MODEL_FILE, "wb") as f:
                pickle.dump({
                    "rf": self.rf, "gb": self.gb, "lr": self.lr,
                    "scaler": self.scaler,
                    "is_trained": self.is_trained,
                    "n_samples":  self.n_samples,
                    "feature_importance": self.feature_importance,
                    "accuracy_history":   self.accuracy_history,
                    "win_rate_history":   self.win_rate_history}, f)
        except Exception as e:
            log.error(f"ML save error: {e}")

    def extract_features(self, of: OrderFlowData,
                         af: Optional[AdvancedOrderFlowData],
                         vp: Optional[VolumeProfile],
                         ob: Optional[OrderBookData],
                         pb: Optional[PositionBookData],
                         cr: PredictionResult) -> List[float]:
        tv  = of.total_volume if of.total_volume > 0 else 1.0
        f0  = of.imbalance / 100
        f1  = min(1.0, abs(of.cvd) / tv)
        f2  = of.delta_momentum / tv
        f3  = of.volume_trend / 100
        f4  = of.recent_delta_5 / tv
        f5  = of.recent_delta_10 / tv
        f6  = of.cvd_slope / tv
        f7  = of.volume_acceleration / tv
        f8  = 1.0 if of.buying_climax  else 0.0
        f9  = 1.0 if of.selling_climax else 0.0
        f10 = ((of.price - of.vwap) / of.vwap * 100
               if of.vwap > 0 else 0.0)
        f11 = af.vpin                       if af else 0.0
        f12 = af.toxicity                   if af else 0.0
        f13 = min(1.0, af.amihud_illiquidity / 100) if af else 0.0
        f14 = af.market_depth_imbalance     if af else 0.0
        f15 = min(1.0, af.absorption_ratio / 3) if af else 0.33
        f16 = af.institutional_flow_score / 100 if af else 0.0
        f17 = min(1.0, max(-1.0, af.trade_arrival_rate / 3)) if af else 0.0
        f18 = af.volume_clustering          if af else 0.0
        f19 = (1.0  if af and af.aggressor_side == "BUYERS"  else
               -1.0 if af and af.aggressor_side == "SELLERS" else 0.0)
        f20 = min(1.0, af.iceberg_count / 10) if af else 0.0
        f21 = abs(af.kyle_lambda) * 100000  if af else 0.0
        f22 = af.roll_spread / 0.01         if af else 0.0
        if vp and vp.vah > vp.val:
            var = vp.vah - vp.val
            f23 = (of.price - vp.poc) / (var + 0.0001)
            f24 = (of.price - vp.val) / (var + 0.0001)
            f25 = 1.0 if vp.val <= of.price <= vp.vah else 0.0
        else:
            f23 = f24 = f25 = 0.0
        f26 = (ob.long_pct - ob.short_pct) / 100 if ob else 0.0
        f27 = ob.pending_delta / 10             if ob else 0.0
        f28 = pb.skew / 100                     if pb else 0.0
        f29 = pb.total_underwater / 100         if pb else 0.0
        f30 = pb.crowded_trade_index / 500      if pb else 0.0
        tf_ = (cr.bullish_count + cr.bearish_count + cr.neutral_count) or 1
        f31 = cr.bullish_count / tf_
        f32 = cr.bearish_count / tf_
        f33 = cr.confidence / 100
        f34 = (1.0 if cr.quality in (SetupQuality.A_PLUS, SetupQuality.A)
               else 0.5 if cr.quality == SetupQuality.B else 0.0)
        raw = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
               f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,
               f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34]
        return [0.0 if (np.isnan(v) or np.isinf(v)) else float(v)
                for v in raw]

    def train(self, predictions: List[Dict]) -> Dict:
        completed = [p for p in predictions
                     if p.get("outcome") in ("WIN", "LOSS")]
        self.n_samples = len(completed)
        if self.n_samples < ML_MIN_SAMPLES:
            return {"status": f"Need {ML_MIN_SAMPLES - self.n_samples} more",
                    "samples": self.n_samples}
        X, y = [], []
        for p in completed:
            feats = p.get("features", [])
            if len(feats) == 35:
                X.append(feats)
                y.append(1 if p["outcome"] == "WIN" else 0)
        if len(X) < ML_MIN_SAMPLES:
            return {"status": "Insufficient feature data", "samples": len(X)}
        Xa  = np.array(X, dtype=np.float32)
        ya  = np.array(y, dtype=np.int32)
        Xs  = self.scaler.fit_transform(Xa)
        self.rf.fit(Xs, ya)
        self.gb.fit(Xs, ya)
        self.lr.fit(Xs, ya)
        nsplit = min(5, len(ya))
        if nsplit >= 2:
            acc = float(np.mean(
                cross_val_score(self.rf, Xs, ya,
                                cv=nsplit, scoring="accuracy")))
        else:
            acc = float(accuracy_score(ya, self.rf.predict(Xs)))
        self.accuracy_history.append(acc)
        self.win_rate_history.append(sum(y) / len(y) if y else 0)
        self.is_trained = True
        fnames = [
            "imbalance","cvd_norm","delta_momentum","vol_trend",
            "recent_delta5","recent_delta10","cvd_slope","vol_acceleration",
            "buying_climax","selling_climax","vwap_dist","vpin","toxicity",
            "amihud","depth_imbalance","absorption","inst_score",
            "arrival_rate","vol_clustering","aggressor","iceberg",
            "kyle_lambda","roll_spread","vp_poc_dist","vp_val_dist",
            "in_value_area","ob_skew","ob_delta","pb_skew","pb_underwater",
            "crowded_idx","bull_pct","bear_pct","confidence","quality"
        ]
        if hasattr(self.rf, "feature_importances_"):
            self.feature_importance = {
                n: float(i)
                for n, i in zip(fnames, self.rf.feature_importances_)}
        self._save()
        log.info(f"ML retrained: {self.n_samples} samples, "
                 f"accuracy={acc:.2%}")
        return {"status": "trained",
                "samples": self.n_samples, "accuracy": acc}

    def predict_proba(self, features: List[float]) -> Tuple[float, str]:
        if not self.is_trained or len(features) != 35:
            return 0.5, "No ML data yet"
        try:
            X   = np.array([features], dtype=np.float32)
            Xs  = self.scaler.transform(X)
            rfp = self.rf.predict_proba(Xs)[0][1]
            gbp = self.gb.predict_proba(Xs)[0][1]
            lrp = self.lr.predict_proba(Xs)[0][1]
            ens = rfp * 0.50 + gbp * 0.35 + lrp * 0.15
            if ens >= 0.75:   lbl = "ML: VERY HIGH CONFIDENCE 🔥"
            elif ens >= 0.60: lbl = "ML: HIGH CONFIDENCE ✅"
            elif ens >= 0.50: lbl = "ML: MODERATE ⚡"
            elif ens >= 0.40: lbl = "ML: LOW ⚠️"
            else:              lbl = "ML: UNFAVORABLE 🔴"
            return float(ens), lbl
        except Exception as e:
            log.error(f"ML predict error: {e}")
            return 0.5, "ML error"

    # alias
    def predict(self, features: List[float]) -> Tuple[float, str]:
        return self.predict_proba(features)

    def get_stats(self) -> Dict:
        return {
            "is_trained":       self.is_trained,
            "n_samples":        self.n_samples,
            "accuracy_history": self.accuracy_history[-10:],
            "latest_accuracy":  (self.accuracy_history[-1]
                                 if self.accuracy_history else 0.0),
            "latest_acc":       (self.accuracy_history[-1]
                                 if self.accuracy_history else 0.0),
            "win_rate_history": self.win_rate_history[-10:],
            "top_features":     sorted(
                self.feature_importance.items(),
                key=lambda x: x[1], reverse=True)[:8]
        }

    # alias
    def stats(self) -> Dict:
        return self.get_stats()


ML_ENGINE = MLLearningEngine()


# ════════════════════════════════════════════════════════════════
#  PATTERN MEMORY ENGINE
# ════════════════════════════════════════════════════════════════

class PatternMemoryEngine:

    def __init__(self):
        self.patterns: List[Dict] = load_json_file(PATTERN_MEMORY_FILE, [])
        log.info(f"Pattern memory loaded: {len(self.patterns)} patterns")

    def _save(self):
        save_json_file(PATTERN_MEMORY_FILE, self.patterns[-500:])

    def record_pattern(self, pred: QuantPrediction,
                       of: OrderFlowData,
                       af: Optional[AdvancedOrderFlowData],
                       session: str):
        pattern = {
            "pattern_id":          pred.prediction_id,
            "pair":                pred.pair,
            "timeframe":           pred.timeframe,
            "timestamp":           pred.timestamp,
            "direction":           pred.direction,
            "outcome":             "PENDING",
            "pips":                0.0,
            "confidence":          pred.confidence,
            "quality":             pred.quality,
            "session":             session,
            "vpin":                af.vpin if af else 0.0,
            "toxicity":            af.toxicity if af else 0.0,
            "institutional_score": (af.institutional_flow_score
                                    if af else 0.0),
            "smart_money":         (af.smart_money_activity
                                    if af else "LOW"),
            "imbalance":           of.imbalance,
            "cvd_positive":        of.cvd > 0,
            "aggressor":           (af.aggressor_side
                                    if af else "NEUTRAL"),
            "iceberg_count":       af.iceberg_count if af else 0
        }
        self.patterns.append(pattern)
        self._save()

    # alias
    def record(self, pred: QuantPrediction,
               of: OrderFlowData,
               af: Optional[AdvancedOrderFlowData],
               session: str):
        self.record_pattern(pred, of, af, session)

    def update_outcome(self, prediction_id: str,
                       outcome: str, pips: float):
        for p in self.patterns:
            if p.get("pattern_id") == prediction_id:
                p["outcome"] = outcome
                p["pips"]    = pips
                break
        self._save()

    # alias
    def update(self, prediction_id: str, outcome: str, pips: float):
        self.update_outcome(prediction_id, outcome, pips)

    def find_similar(self, pair: str, direction: str,
                     af: Optional[AdvancedOrderFlowData],
                     of: Optional[OrderFlowData],
                     session: str) -> Dict:
        completed = [p for p in self.patterns
                     if p.get("outcome") in ("WIN", "LOSS")
                     and p.get("pair") == pair]
        if not completed:
            return {"found": False, "count": 0}
        scored = []
        for p in completed:
            score = 0
            if p.get("direction") == direction:             score += 30
            if p.get("session") == session:                 score += 20
            if af:
                vpin_diff = abs(p.get("vpin", 0) - af.vpin)
                if vpin_diff < 0.1:                         score += 20
                elif vpin_diff < 0.2:                       score += 10
                if p.get("aggressor") == af.aggressor_side: score += 15
                if p.get("smart_money") == af.smart_money_activity:
                    score += 15
            if of:
                if (p.get("cvd_positive", False)) == (of.cvd > 0):
                    score += 10
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [p for s, p in scored[:10] if s >= 40]
        if not top:
            return {"found": False, "count": 0}
        wins    = sum(1 for p in top if p.get("outcome") == "WIN")
        losses  = len(top) - wins
        wr      = wins / len(top) if top else 0
        avg_pips = (sum(abs(p.get("pips", 0)) for p in top
                        if p.get("outcome") == "WIN") / max(wins, 1))
        best    = max(top, key=lambda x: abs(x.get("pips", 0)))
        return {
            "found":        True,
            "count":        len(top),
            "wins":         wins,
            "losses":       losses,
            "win_rate":     wr,
            "avg_pips":     avg_pips,
            "best_outcome": best.get("outcome", ""),
            "best_pips":    best.get("pips", 0),
            "best_session": best.get("session", ""),
            "narrative":    self._build_narrative(
                pair, direction, top, wr, avg_pips)
        }

    def _build_narrative(self, pair: str, direction: str,
                         patterns: List[Dict],
                         wr: float, avg_pips: float) -> str:
        n         = len(patterns)
        pair_disp = pair_to_display(pair)
        if wr >= 0.70:   strength = "strongly"; emoji = "🔥"
        elif wr >= 0.55: strength = "moderately"; emoji = "✅"
        else:            strength = "weakly"; emoji = "⚠️"
        narrative = (
            f"{emoji} <b>Pattern Memory says:</b>\n"
            f"We have seen <b>{n} similar setups</b> on {pair_disp} "
            f"in the past.\n\n"
            f"When these conditions aligned, the market <b>{strength} "
            f"favoured {'bulls' if direction == 'LONG' else 'bears'}</b>"
            f" — winning <b>{wr:.0%}</b> of the time with an average of "
            f"<b>{avg_pips:.1f} pips</b> gained per winning trade.\n\n"
        )
        if wr >= 0.60:
            narrative += ("Based on historical precedent, we <b>expect "
                          "the move to follow through</b>.")
        elif wr >= 0.45:
            narrative += ("History shows mixed results. "
                          "<b>Manage risk carefully</b>.")
        else:
            narrative += ("⚠️ Historical data suggests this pattern has "
                          "been <b>less reliable</b>. Reduce position size.")
        return narrative


PATTERN_MEMORY = PatternMemoryEngine()


# ════════════════════════════════════════════════════════════════
#  HISTORICAL BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

class HistoricalBacktestEngine:

    def __init__(self):
        self.results: Dict[str, Dict] = load_json_file(HISTORICAL_BT_FILE, {})
        self.is_running  = False
        self._last_run:  Optional[datetime] = None
        log.info(f"Historical backtest loaded: {len(self.results)} results")

    def _save(self):
        save_json_file(HISTORICAL_BT_FILE, self.results)

    def stale(self) -> bool:
        """Returns True if backtest has never run or ran > 24 h ago."""
        if not self.results or self._last_run is None:
            return True
        return (datetime.now(timezone.utc) - self._last_run
                ).total_seconds() > 86400

    async def run(self, pairs: Optional[List[str]] = None,
                  tf: str = "H1") -> Dict:
        return await self.run_full_backtest(pairs=pairs, timeframe=tf)

    async def run_full_backtest(self,
                                pairs: Optional[List[str]] = None,
                                timeframe: str = "H1") -> Dict:
        if self.is_running:
            return {"status": "already_running"}
        self.is_running    = True
        pairs_to_test      = pairs or ASSET_CATEGORIES["forex_major"][:5]
        all_results: Dict  = {}
        log.info(f"Historical backtest starting for "
                 f"{len(pairs_to_test)} pairs...")
        async with aiohttp.ClientSession() as session:
            for pair in pairs_to_test:
                try:
                    result = await self._backtest_pair(session, pair, timeframe)
                    if result:
                        key = f"{pair}_{timeframe}"
                        all_results[key]  = result
                        self.results[key] = result
                        log.info(f"BT {pair}: {result['win_rate']:.1%} WR "
                                 f"({result['wins']}W/{result['losses']}L)")
                    await asyncio.sleep(1.0)
                except Exception as e:
                    log.error(f"BT error {pair}: {e}")
        self._save()
        await self._feed_ml(all_results)
        self.is_running = False
        self._last_run  = datetime.now(timezone.utc)
        log.info("Historical backtest complete")
        return all_results

    async def _backtest_pair(self, session: aiohttp.ClientSession,
                             pair: str,
                             timeframe: str) -> Optional[Dict]:
        candles = await fetch_candles(
            session, pair, timeframe, HISTORICAL_BT_CANDLES)
        if len(candles) < 60:
            return None
        signals = []
        window  = 50
        step    = 10
        for i in range(window, len(candles) - 20, step):
            analysis_candles = candles[max(0, i - window):i]
            if len(analysis_candles) < 30:
                continue
            try:
                of  = OF_ENGINE.analyze(analysis_candles)
                vp  = calculate_volume_profile(analysis_candles)
                af  = ADVANCED_OF.analyze_all(analysis_candles, None)
                lz  = LIQUIDITY_DETECTOR.detect_all(analysis_candles, pair)
                result = OF_ENGINE.evaluate_confluence(
                    of, vp, None, None, None, af, pair, of.price, lz)
                if (not result.has_setup or
                        result.confidence < MIN_CONFIDENCE or
                        result.quality == SetupQuality.NO_TRADE):
                    continue
                entry_price = analysis_candles[-1].close
                atr         = calculate_atr(analysis_candles, 14)
                mult        = 1 if result.direction == "LONG" else -1
                tgt_mult    = (2.5 if result.quality == SetupQuality.A_PLUS
                               else 2.0 if result.quality == SetupQuality.A
                               else 1.5 if result.quality == SetupQuality.B
                               else 1.0)
                target       = entry_price + atr * tgt_mult * mult
                invalidation = entry_price - atr * 1.0 * mult
                future_candles = candles[i:i + 20]
                outcome, p_pips = self._simulate_outcome(
                    future_candles, entry_price, target,
                    invalidation, result.direction, pair)
                sn = get_session_name(
                    datetime.fromisoformat(
                        analysis_candles[-1].time.replace("Z", "")).hour
                    if analysis_candles[-1].time else 12)
                signals.append({
                    "pair":       pair,
                    "timeframe":  timeframe,
                    "direction":  result.direction,
                    "quality":    result.quality.value,
                    "confidence": result.confidence,
                    "outcome":    outcome,
                    "pips":       p_pips,
                    "session":    sn,
                    "vpin":       af.vpin,
                    "inst_score": af.institutional_flow_score,
                    "features":   ML_ENGINE.extract_features(
                        of, af, vp, None, None, result)
                })
            except Exception:
                continue
        if not signals:
            return None
        wins   = sum(1 for s in signals if s["outcome"] == "WIN")
        losses = sum(1 for s in signals if s["outcome"] == "LOSS")
        total  = wins + losses
        wr     = wins / total if total > 0 else 0.0
        session_wr: Dict = {}
        for sn in ["TOKYO","LONDON","NEW_YORK","OVERLAP","OFF_HOURS"]:
            ss = [s for s in signals
                  if s["session"] == sn
                  and s["outcome"] in ("WIN","LOSS")]
            if ss:
                sw = sum(1 for s in ss if s["outcome"] == "WIN")
                session_wr[sn] = {
                    "wins": sw, "total": len(ss),
                    "win_rate": sw / len(ss)}
        quality_wr: Dict = {}
        for q in [SetupQuality.A_PLUS.value, SetupQuality.A.value,
                  SetupQuality.B.value, SetupQuality.C.value]:
            qs = [s for s in signals
                  if s["quality"] == q
                  and s["outcome"] in ("WIN","LOSS")]
            if qs:
                qw = sum(1 for s in qs if s["outcome"] == "WIN")
                quality_wr[q] = {
                    "wins": qw, "total": len(qs),
                    "win_rate": qw / len(qs)}
        best_sess  = (max(session_wr.items(),
                          key=lambda x: x[1]["win_rate"])[0]
                      if session_wr else "N/A")
        worst_sess = (min(session_wr.items(),
                          key=lambda x: x[1]["win_rate"])[0]
                      if session_wr else "N/A")
        best_qual  = (max(quality_wr.items(),
                          key=lambda x: x[1]["win_rate"])[0]
                      if quality_wr else "N/A")
        avg_pips   = (sum(abs(s["pips"]) for s in signals
                          if s["outcome"] == "WIN") / max(wins, 1))
        return {
            "pair":              pair,
            "timeframe":         timeframe,
            "period_days":       30,
            "total_signals":     len(signals),
            "wins":              wins,
            "losses":            losses,
            "win_rate":          wr,
            "avg_pips":          avg_pips,
            "best_session":      best_sess,
            "worst_session":     worst_sess,
            "best_quality":      best_qual,
            "quality_win_rates": quality_wr,
            "session_win_rates": session_wr,
            "signals":           signals,
            "last_run":          datetime.now(timezone.utc).isoformat(),
            "patterns_learned":  len(signals)
        }

    def _simulate_outcome(self, future_candles: List[Candle],
                          entry: float, target: float,
                          invalidation: float,
                          direction: str,
                          instrument: str) -> Tuple[str, float]:
        for c in future_candles:
            if direction == "LONG":
                if c.high >= target:
                    return "WIN",  pips_diff(instrument, target - entry)
                if c.low  <= invalidation:
                    return "LOSS", -pips_diff(instrument, entry - invalidation)
            else:
                if c.low  <= target:
                    return "WIN",  pips_diff(instrument, entry - target)
                if c.high >= invalidation:
                    return "LOSS", -pips_diff(instrument, invalidation - entry)
        last  = future_candles[-1].close if future_candles else entry
        p_pip = ((last - entry) if direction == "LONG"
                 else (entry - last)) / pip_value(instrument)
        return ("WIN" if p_pip > 0 else "LOSS"), p_pip

    async def _feed_ml(self, results: Dict):
        all_signals: List[Dict] = []
        for key, result in results.items():
            if isinstance(result, dict):
                all_signals.extend(result.get("signals", []))
        if len(all_signals) >= ML_MIN_SAMPLES:
            ML_ENGINE.train(all_signals)
            log.info(f"ML trained on {len(all_signals)} backtest signals")

    def get_pair_result(self, pair: str,
                        timeframe: str = "H1") -> Optional[Dict]:
        return self.results.get(f"{pair}_{timeframe}")

    def report(self, predictions: List[Dict]) -> str:
        return self.format_backtest_report(predictions)

    def format_backtest_report(self, predictions: List[Dict]) -> str:
        live_completed = [p for p in predictions
                         if p.get("outcome") in ("WIN","LOSS","EXPIRED")]
        live_wins   = sum(1 for p in live_completed
                         if p.get("outcome") == "WIN")
        live_losses = sum(1 for p in live_completed
                         if p.get("outcome") == "LOSS")
        live_total  = live_wins + live_losses
        live_wr     = live_wins / live_total if live_total > 0 else 0.0
        decided     = [p for p in live_completed
                      if p.get("outcome") in ("WIN","LOSS")]
        recent20    = decided[-20:] if len(decided) >= 20 else decided
        recent_wr   = (sum(1 for p in recent20
                           if p.get("outcome") == "WIN") /
                       len(recent20) if recent20 else 0.0)
        if live_wr >= 0.70:   perf = "🔥 ELITE"
        elif live_wr >= 0.60: perf = "✅ STRONG"
        elif live_wr >= 0.50: perf = "⚡ MODERATE"
        elif live_wr >= 0.40: perf = "⚠️ WEAK"
        else:                  perf = "🔴 BUILDING DATA"
        ml_stats = ML_ENGINE.get_stats()
        msg = (
            f"{'='*35}\n📊 <b>QUANT PERFORMANCE REPORT</b>\n{'='*35}\n\n"
            f"<b>Live Signal Performance:</b>\n"
            f"├ Total Signals:    <b>{len(live_completed)}</b>\n"
            f"├ Wins:             <b>✅ {live_wins}</b>\n"
            f"├ Losses:           <b>❌ {live_losses}</b>\n"
            f"├ Win Rate:         <b>{live_wr:.1%}</b> "
            f"{progress_bar(live_wr * 100)}\n"
            f"├ Recent (last 20): <b>{recent_wr:.1%}</b>\n"
            f"└ Rating:           <b>{perf}</b>\n\n"
        )
        if self.results:
            msg += (f"{'─'*35}\n<b>📈 Historical Backtest (30 Days):"
                    f"</b>\n{'─'*35}\n")
            for key, result in list(self.results.items())[:6]:
                if isinstance(result, dict) and "win_rate" in result:
                    pair_str = pair_to_display(result.get("pair", "?"))
                    wr       = result["win_rate"]
                    sigs     = result["total_signals"]
                    best_s   = result.get("best_session", "?")
                    msg += (f"💱 <b>{pair_str}</b>: {wr:.1%} WR "
                            f"({result['wins']}W/{result['losses']}L, "
                            f"{sigs} signals)\n"
                            f"   Best session: {best_s}\n")
            msg += "\n"
        msg += f"{'─'*35}\n<b>🧠 ML Engine Status:</b>\n{'─'*35}\n"
        if ml_stats["is_trained"]:
            msg += (f"├ Status:    ✅ Active and Learning\n"
                    f"├ Samples:   {ml_stats['n_samples']}\n"
                    f"├ Accuracy:  <b>{ml_stats['latest_accuracy']:.1%}</b>\n")
            top = ml_stats.get("top_features", [])
            if top:
                msg += f"└ Top Signal: <b>{top[0][0]}</b>\n\n"
        else:
            msg += (f"├ Status: ⏳ Accumulating data\n"
                    f"└ {ml_stats['n_samples']}/{ML_MIN_SAMPLES} samples\n\n")
        msg += (
            f"{'─'*35}\n<b>⚠️ Important Reminder:</b>\n"
            f"No system wins 100% of the time. Risk management is "
            f"everything — always use a stop-loss, never risk more "
            f"than 1-2% per trade.\n\n"
            f"<i>The Quant learns from every trade.</i>\n"
            f"{'='*35}\n"
        )
        return msg

    def chart(self, predictions: List[Dict]) -> Optional[io.BytesIO]:
        return self.format_backtest_chart(predictions)

    def format_backtest_chart(self,
                              predictions: List[Dict]) -> Optional[io.BytesIO]:
        live    = [p for p in predictions
                   if p.get("outcome") in ("WIN","LOSS")]
        ht_data = [(k, v) for k, v in self.results.items()
                   if isinstance(v, dict) and "win_rate" in v]
        if not live and not ht_data:
            return None
        try:
            fig    = plt.figure(figsize=(14,10), facecolor='#0d1117')
            gs     = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            colors = {'bg':'#0d1117','text':'#c9d1d9','grid':'#21262d',
                      'up':'#3fb950','down':'#f85149',
                      'acc':'#58a6ff','gold':'#ffd700'}

            # Panel 1 — Live pie
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor(colors['bg'])
            if live:
                lw = sum(1 for p in live if p.get("outcome") == "WIN")
                ll = len(live) - lw
                if lw + ll > 0:
                    pie_v = [v for v in [lw, ll] if v > 0]
                    pie_l = [lb for v, lb in
                             zip([lw,ll],["Wins","Losses"]) if v > 0]
                    pie_c = [colors['up'],colors['down']][:len(pie_v)]
                    wd, tx, atx = ax1.pie(
                        pie_v, labels=pie_l, colors=pie_c,
                        autopct='%1.0f%%', startangle=90,
                        textprops={'color':colors['text'],'fontsize':11})
                    for at in atx:
                        at.set_color(colors['bg'])
                        at.set_fontweight('bold')
                ax1.set_title(
                    f"Live: {lw/(lw+ll):.1%} WR" if live else "No live data",
                    color=colors['text'], fontsize=12, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, "No Live Data", ha='center', va='center',
                         color=colors['text'], transform=ax1.transAxes)
                ax1.set_title("Live Performance",
                              color=colors['text'], fontsize=12)

            # Panel 2 — Historical WR bars
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor(colors['bg'])
            if ht_data:
                labels  = [pair_to_display(v["pair"])
                           for k, v in ht_data[:6]]
                wrs     = [v["win_rate"] * 100
                           for k, v in ht_data[:6]]
                bcolors = [colors['up'] if w >= 55
                           else colors['down'] for w in wrs]
                bars    = ax2.bar(np.arange(len(labels)), wrs,
                                  color=bcolors, alpha=0.85, width=0.6)
                ax2.axhline(y=50, color=colors['grid'],
                            linestyle='--', linewidth=1.5)
                ax2.set_xticks(np.arange(len(labels)))
                ax2.set_xticklabels(labels, color=colors['text'],
                                    fontsize=9, rotation=20)
                ax2.set_ylabel("Win Rate %",
                               color=colors['text'], fontsize=10)
                ax2.set_ylim(0, 100)
                for bar, (k, v) in zip(bars, ht_data[:6]):
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 2,
                             f"{v['total_signals']}s",
                             ha='center', color=colors['text'], fontsize=8)
            ax2.set_title("Historical 30-Day Backtest",
                          color=colors['text'], fontsize=12, fontweight='bold')
            ax2.tick_params(colors=colors['text'])
            ax2.grid(True, alpha=0.15, color=colors['grid'])

            # Panel 3 — ML accuracy
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.set_facecolor(colors['bg'])
            ml_hist = ML_ENGINE.accuracy_history
            if ml_hist and len(ml_hist) >= 2:
                ax3.plot(range(len(ml_hist)),
                         [v * 100 for v in ml_hist],
                         color=colors['acc'], linewidth=2.5,
                         marker='o', markersize=6)
                ax3.fill_between(range(len(ml_hist)),
                                 [v * 100 for v in ml_hist],
                                 alpha=0.2, color=colors['acc'])
                ax3.axhline(y=50, color=colors['grid'],
                            linestyle='--', linewidth=1.5)
                ax3.set_ylim(0, 100)
                ax3.set_title(f"ML Accuracy: {ml_hist[-1]:.1%}",
                              color=colors['gold'],
                              fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5,
                         f"ML learning...\n"
                         f"({ML_ENGINE.n_samples}/{ML_MIN_SAMPLES})",
                         ha='center', va='center',
                         color=colors['text'], fontsize=12,
                         transform=ax3.transAxes)
                ax3.set_title("ML Engine",
                              color=colors['gold'],
                              fontsize=12, fontweight='bold')
            ax3.tick_params(colors=colors['text'])
            ax3.grid(True, alpha=0.15, color=colors['grid'])

            # Panel 4 — Session WR
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.set_facecolor(colors['bg'])
            session_data: Dict = {
                "TOKYO":    {"w": 0, "t": 0},
                "LONDON":   {"w": 0, "t": 0},
                "NEW_YORK": {"w": 0, "t": 0},
                "OVERLAP":  {"w": 0, "t": 0}
            }
            for k, v in ht_data:
                if isinstance(v, dict):
                    for sn, sv in v.get("session_win_rates", {}).items():
                        if sn in session_data:
                            session_data[sn]["w"] += sv.get("wins", 0)
                            session_data[sn]["t"] += sv.get("total", 0)
            slabels = list(session_data.keys())
            swrs    = [(session_data[s]["w"] / session_data[s]["t"] * 100)
                       if session_data[s]["t"] > 0 else 0
                       for s in slabels]
            sc_     = [colors['up'] if w >= 55
                       else colors['down'] for w in swrs]
            ax4.bar(np.arange(len(slabels)), swrs,
                    color=sc_, alpha=0.85, width=0.6)
            ax4.axhline(y=50, color=colors['grid'],
                        linestyle='--', linewidth=1.5)
            ax4.set_xticks(np.arange(len(slabels)))
            ax4.set_xticklabels(slabels, color=colors['text'],
                                fontsize=9, rotation=15)
            ax4.set_ylabel("Win Rate %",
                           color=colors['text'], fontsize=10)
            ax4.set_ylim(0, 100)
            ax4.set_title("Win Rate by Session",
                          color=colors['text'],
                          fontsize=12, fontweight='bold')
            ax4.tick_params(colors=colors['text'])
            ax4.grid(True, alpha=0.15, color=colors['grid'])

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            fig.text(0.5, 0.97,
                     "FOREX QUANT v7.5 — PERFORMANCE ANALYTICS",
                     ha='center', color=colors['text'],
                     fontsize=14, fontweight='bold')
            fig.text(0.98, 0.01, f"Generated: {ts}",
                     ha='right', color=colors['text'],
                     fontsize=8, alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120,
                        facecolor=colors['bg'],
                        edgecolor='none', bbox_inches='tight')
            buf.seek(0); plt.close(fig); gc.collect()
            return buf
        except Exception as e:
            log.error(f"Backtest chart error: {e}")
            return None


HISTORICAL_BT = HistoricalBacktestEngine()


# ════════════════════════════════════════════════════════════════
#  USER SUBSCRIPTION MANAGER
# ════════════════════════════════════════════════════════════════

class SubscriptionManager:

    def __init__(self):
        raw = load_json_file(USER_SUBSCRIPTIONS_FILE, {})
        self.subscriptions: Dict[int, Dict] = {}
        for k, v in raw.items():
            try:
                self.subscriptions[int(k)] = v
            except Exception:
                pass
        log.info(f"Subscriptions loaded: {len(self.subscriptions)} users")

    def _save(self):
        save_json_file(USER_SUBSCRIPTIONS_FILE,
                       {str(k): v for k, v in self.subscriptions.items()})

    def subscribe(self, chat_id: int, pairs: List[str],
                  timeframe: str = "H1"):
        self.subscriptions[chat_id] = {
            "pairs":         pairs,
            "timeframe":     timeframe,
            "active":        True,
            "subscribed_at": datetime.now(timezone.utc).isoformat(),
            "last_alert":    None,
            "alert_count":   0
        }
        self._save()

    def unsubscribe(self, chat_id: int):
        if chat_id in self.subscriptions:
            self.subscriptions[chat_id]["active"] = False
            self._save()

    def is_subscribed(self, chat_id: int) -> bool:
        sub = self.subscriptions.get(chat_id)
        return sub is not None and sub.get("active", False)

    # alias
    def is_active(self, chat_id: int) -> bool:
        return self.is_subscribed(chat_id)

    def get_subscription(self, chat_id: int) -> Optional[Dict]:
        return self.subscriptions.get(chat_id)

    # alias
    def get(self, chat_id: int) -> Optional[Dict]:
        return self.get_subscription(chat_id)

    def get_all_active(self) -> List[Tuple[int, Dict]]:
        return [(cid, sub)
                for cid, sub in self.subscriptions.items()
                if sub.get("active", False)]

    # alias
    def all_active(self) -> List[Tuple[int, Dict]]:
        return self.get_all_active()

    def record_alert(self, chat_id: int):
        if chat_id in self.subscriptions:
            self.subscriptions[chat_id]["last_alert"] = (
                datetime.now(timezone.utc).isoformat())
            self.subscriptions[chat_id]["alert_count"] = (
                self.subscriptions[chat_id].get("alert_count", 0) + 1)
            self._save()

    def add_pair(self, chat_id: int, pair: str):
        if chat_id in self.subscriptions:
            pairs = self.subscriptions[chat_id].get("pairs", [])
            if pair not in pairs:
                pairs.append(pair)
                self.subscriptions[chat_id]["pairs"] = pairs
                self._save()

    def remove_pair(self, chat_id: int, pair: str):
        if chat_id in self.subscriptions:
            pairs = self.subscriptions[chat_id].get("pairs", [])
            if pair in pairs:
                pairs.remove(pair)
                self.subscriptions[chat_id]["pairs"] = pairs
                self._save()

    def set_tf(self, chat_id: int, timeframe: str):
        """Set alert timeframe for a subscriber."""
        if chat_id in self.subscriptions:
            self.subscriptions[chat_id]["timeframe"] = timeframe
            self._save()


SUB_MANAGER = SubscriptionManager()

# ════════════════════════════════════════════════════════════════
#  SESSION ADVISOR (single definition)
# ════════════════════════════════════════════════════════════════

class SessionAdvisor:

    INFO = {
        "OVERLAP": {
            "name":  "London-NY Overlap (13:00-16:00 UTC)",
            "emoji": "⚡",
            "desc":  ("The most powerful session. Maximum liquidity, tightest "
                      "spreads, biggest moves. Institutional volume peaks here. "
                      "This is the holy grail of trading windows."),
            "best":  ["EUR_USD", "GBP_USD", "XAU_USD", "EUR_GBP", "GBP_JPY"]
        },
        "LONDON": {
            "name":  "London Session (07:00-16:00 UTC)",
            "emoji": "🏦",
            "desc":  ("Second most liquid. Major institutions set the tone. "
                      "Most trend initiations happen during London open."),
            "best":  ["EUR_USD", "GBP_USD", "EUR_GBP", "USD_CHF", "EUR_CHF"]
        },
        "NEW_YORK": {
            "name":  "New York Session (13:00-22:00 UTC)",
            "emoji": "🗽",
            "desc":  ("Strong first half — especially during overlap. "
                      "USD pairs and Gold very active. Weakens after 17:00 UTC."),
            "best":  ["EUR_USD", "USD_CAD", "USD_JPY", "XAU_USD", "NAS100_USD"]
        },
        "TOKYO": {
            "name":  "Tokyo Session (00:00-09:00 UTC)",
            "emoji": "🗼",
            "desc":  ("Lower volatility but excellent for JPY pairs. "
                      "Range-bound action is common. Trends set here "
                      "often continue into London."),
            "best":  ["USD_JPY", "AUD_JPY", "EUR_JPY", "GBP_JPY", "AUD_USD"]
        },
        "OFF_HOURS": {
            "name":  "Off Hours (22:00-00:00 UTC)",
            "emoji": "😴",
            "desc":  ("Very low liquidity. Wide spreads. Stop-hunts common. "
                      "We strongly advise AGAINST trading during this period."),
            "best":  []
        }
    }

    def advice(self, pair: str) -> str:
        sn   = session_name(datetime.now(timezone.utc).hour)
        info = self.INFO.get(sn, self.INFO["OFF_HOURS"])
        best = info.get("best", [])
        msg  = (
            f"{'─'*35}\n<b>{info['emoji']} Session Advice</b>\n{'─'*35}\n\n"
            f"<b>{info['name']}</b>\n\n"
            f"{info['desc']}\n\n"
        )
        if pair in best:
            msg += (f"✅ <b>{pair_display(pair)} is one of the BEST "
                    f"pairs for this session.</b>\n")
        elif sn == "OFF_HOURS":
            msg += f"⚠️ <b>Avoid trading {pair_display(pair)} during off hours.</b>\n"
        else:
            msg += (f"⚡ {pair_display(pair)} can be traded this session "
                    f"but is not optimal.\n")
        if best:
            msg += "\n<b>Best pairs right now:</b>\n"
            for p in best[:4]:
                msg += f"  • {pair_display(p)}\n"
        return msg

    def schedule(self) -> str:
        now  = datetime.now(timezone.utc)
        hour = now.hour
        sn   = session_name(hour)
        msg  = (
            f"{'='*35}\n🗓️ <b>TRADING SESSION GUIDE</b>\n{'='*35}\n\n"
            f"<b>Current Time:</b> {now.strftime('%H:%M UTC')}\n"
            f"<b>Current Session:</b> "
            f"{self.INFO.get(sn, {}).get('name', 'Off Hours')}\n\n"
        )
        for s_key, sd in self.INFO.items():
            active = " ◀️ YOU ARE HERE" if s_key == sn else ""
            best_s = (", ".join(pair_display(p)
                                for p in sd.get("best", [])[:3]) or "N/A")
            desc   = sd["desc"][:120] + "..." if len(sd["desc"]) > 120 else sd["desc"]
            msg += (
                f"{sd['emoji']} <b>{sd['name']}</b>{active}\n"
                f"{desc}\n"
                f"Best: {best_s}\n\n"
            )
        msg += (
            f"{'─'*35}\n"
            f"<b>💡 Golden Rules:</b>\n"
            f"• Trade during your pair's session\n"
            f"• Never trade off-hours\n"
            f"• London-NY overlap is premium time\n\n"
            f"<b>⚠️ Risk Reminder:</b>\n"
            f"No analysis is 100% correct. Use a stop-loss on every "
            f"trade. Risk max 1-2% per trade. Being wrong is completely "
            f"normal — even the best traders lose regularly."
        )
        return msg


SA = SessionAdvisor()


# ════════════════════════════════════════════════════════════════
#  QUANT BRAIN
# ════════════════════════════════════════════════════════════════

class QuantBrain:

    def __init__(self):
        self.history: List[Dict] = load_json_file(PREDICTIONS_FILE, [])

    def _save(self):
        save_json_file(PREDICTIONS_FILE, self.history[-1000:])

    def generate(self, candles: List[Candle], of: OrderFlowData,
                 vp: Optional[VolumeProfile], ob: Optional[OrderBookData],
                 pb: Optional[PositionBookData], cs: Optional[Dict],
                 af: Optional[AdvancedOrderFlowData], instrument: str,
                 lz: Optional[List[LiquidityZone]] = None) -> Dict:

        price = candles[-1].close
        atr   = calculate_atr(candles, 14)
        cr    = OF_ENGINE.evaluate_confluence(
            of, vp, ob, pb, cs, af, instrument, price, lz)

        if not cr.has_setup or cr.quality == SetupQuality.NO_TRADE:
            return {
                "has_setup":  False,
                "bull_count": cr.bullish_count,
                "bear_count": cr.bearish_count,
                "cr":         cr
            }
        if cr.confidence < MIN_CONFIDENCE:
            return {
                "has_setup":  False,
                "bull_count": cr.bullish_count,
                "bear_count": cr.bearish_count,
                "cr":         cr
            }

        m   = 1 if cr.direction == "LONG" else -1
        tm  = (2.5 if cr.quality == SetupQuality.A_PLUS else
               2.0 if cr.quality == SetupQuality.A      else
               1.5 if cr.quality == SetupQuality.B      else 1.0)
        inv = atr * 1.0
        tgt = max(atr * tm, inv)
        target = price + tgt * m
        stop   = price - inv * m
        kl     = identify_key_levels(candles, vp, price, instrument)
        feats  = ML_ENGINE.extract_features(of, af, vp, ob, pb, cr)
        ml_p, ml_l = ML_ENGINE.predict_proba(feats)

        return {
            "has_setup":  True,
            "direction":  cr.direction,
            "quality":    cr.quality.value,
            "target":     target,
            "stop":       stop,
            "confidence": cr.confidence,
            "reasons":    cr.reasons,
            "key_levels": kl,
            "bull_count": cr.bullish_count,
            "bear_count": cr.bearish_count,
            "factors":    [(f.name, f.direction, f.description)
                           for f in cr.factors],
            "features":   feats,
            "ml_prob":    ml_p,
            "ml_label":   ml_l,
            "cr":         cr
        }

    def add(self, pred: QuantPrediction):
        self.history.append({
            "prediction_id":   pred.prediction_id,
            "pair":            pred.pair,
            "timeframe":       pred.timeframe,
            "timestamp":       pred.timestamp,
            "current_price":   pred.current_price,
            "direction":       pred.direction,
            "target_price":    pred.target_price,
            "invalidation_price": pred.invalidation_price,
            "confidence":      pred.confidence,
            "quality":         pred.quality,
            "reasons":         pred.reasons,
            "key_levels":      pred.key_levels,
            "factors_aligned": pred.factors_aligned,
            "features":        pred.features,
            "ml_confidence":   pred.ml_confidence,
            "ml_used":         pred.ml_used,
            "outcome":         "PENDING",
            "pips_gained":     0.0
        })
        self._save()


QB = QuantBrain()


# ════════════════════════════════════════════════════════════════
#  PREDICTION TRACKER
# ════════════════════════════════════════════════════════════════

class PredictionTracker:

    def __init__(self):
        self.active: List[QuantPrediction] = []
        self._load()

    def _load(self):
        if not os.path.exists(ACTIVE_PREDICTIONS_FILE):
            return
        try:
            raw = load_json_file(ACTIVE_PREDICTIONS_FILE, [])
            self.active = []
            for p in raw:
                if p.get("status") == "ACTIVE":
                    try:
                        self.active.append(QuantPrediction(**p))
                    except Exception as e:
                        log.warning(f"Skip prediction: {e}")
            log.info(f"Active predictions: {len(self.active)}")
        except Exception as e:
            log.error(f"Load predictions: {e}")

    def _save(self):
        try:
            data = []
            for p in self.active:
                data.append({
                    "prediction_id":      p.prediction_id,
                    "pair":               p.pair,
                    "timeframe":          p.timeframe,
                    "timestamp":          p.timestamp,
                    "current_price":      p.current_price,
                    "direction":          p.direction,
                    "target_price":       p.target_price,
                    "invalidation_price": p.invalidation_price,
                    "confidence":         p.confidence,
                    "quality":            p.quality,
                    "reasons":            p.reasons,
                    "key_levels":         p.key_levels,
                    "factors_aligned":    p.factors_aligned,
                    "features":           p.features,
                    "status":             p.status,
                    "outcome":            p.outcome,
                    "hit_time":           p.hit_time,
                    "chat_id":            p.chat_id,
                    "ml_confidence":      p.ml_confidence,
                    "ml_used":            p.ml_used,
                    "pips_gained":        p.pips_gained
                })
            save_json_file(ACTIVE_PREDICTIONS_FILE, data)
        except Exception as e:
            log.error(f"Save predictions: {e}")

    def add(self, pred: QuantPrediction):
        self.active.append(pred)
        self._save()
        log.info(f"Tracking {pred.prediction_id} {pred.pair}")

    async def check(self, bot) -> List[Dict]:
        notifications: List[Dict] = []
        if not self.active:
            return notifications

        async with aiohttp.ClientSession() as session:
            for pred in self.active[:]:
                try:
                    cp = await fetch_current_price(session, pred.pair)
                    if cp is None:
                        continue

                    resolved = False

                    if pred.direction == "LONG":
                        if cp >= pred.target_price:
                            pred.status      = "TARGET_HIT"
                            pred.outcome     = "WIN"
                            pred.hit_time    = datetime.now(timezone.utc).isoformat()
                            pred.pips_gained = pips_diff(
                                pred.pair,
                                pred.target_price - pred.current_price)
                            notifications.append(
                                {"pred": pred, "type": "TARGET_HIT", "cp": cp})
                            resolved = True
                        elif cp <= pred.invalidation_price:
                            pred.status      = "INVALIDATED"
                            pred.outcome     = "LOSS"
                            pred.hit_time    = datetime.now(timezone.utc).isoformat()
                            pred.pips_gained = -pips_diff(
                                pred.pair,
                                pred.current_price - pred.invalidation_price)
                            notifications.append(
                                {"pred": pred, "type": "INVALIDATED", "cp": cp})
                            resolved = True
                    else:  # SHORT
                        if cp <= pred.target_price:
                            pred.status      = "TARGET_HIT"
                            pred.outcome     = "WIN"
                            pred.hit_time    = datetime.now(timezone.utc).isoformat()
                            pred.pips_gained = pips_diff(
                                pred.pair,
                                pred.current_price - pred.target_price)
                            notifications.append(
                                {"pred": pred, "type": "TARGET_HIT", "cp": cp})
                            resolved = True
                        elif cp >= pred.invalidation_price:
                            pred.status      = "INVALIDATED"
                            pred.outcome     = "LOSS"
                            pred.hit_time    = datetime.now(timezone.utc).isoformat()
                            pred.pips_gained = -pips_diff(
                                pred.pair,
                                pred.invalidation_price - pred.current_price)
                            notifications.append(
                                {"pred": pred, "type": "INVALIDATED", "cp": cp})
                            resolved = True

                    if not resolved:
                        try:
                            pt = datetime.fromisoformat(
                                pred.timestamp.replace("Z", "+00:00"))
                            if datetime.now(timezone.utc) - pt > timedelta(hours=24):
                                pred.status  = "EXPIRED"
                                pred.outcome = "EXPIRED"
                                notifications.append(
                                    {"pred": pred, "type": "EXPIRED", "cp": cp})
                                resolved = True
                        except Exception:
                            pass

                    if resolved:
                        self.active.remove(pred)

                    await asyncio.sleep(0.1)

                except Exception as e:
                    log.error(f"Check {pred.prediction_id}: {e}")

        if notifications:
            self._save()
            try:
                # Update QB history outcomes
                for n in notifications:
                    p = n["pred"]
                    for h in QB.history:
                        if h.get("prediction_id") == p.prediction_id:
                            h["outcome"]     = p.outcome or "EXPIRED"
                            h["pips_gained"] = p.pips_gained
                            break
                    QB._save()

                done = [p for p in QB.history
                        if p.get("outcome") in ("WIN", "LOSS")]
                if len(done) >= ML_MIN_SAMPLES:
                    ML_ENGINE.train(QB.history)
                for n in notifications:
                    p = n["pred"]
                    PATTERN_MEMORY.update_outcome(
                        p.prediction_id,
                        p.outcome or "EXPIRED",
                        p.pips_gained)
            except Exception as e:
                log.error(f"Post-resolve: {e}")

        return notifications


PT = PredictionTracker()


# ════════════════════════════════════════════════════════════════
#  ALERT NOTIFICATION FORMATTER
# ════════════════════════════════════════════════════════════════

def _alert_msg(pred: QuantPrediction, atype: str, cp: float) -> str:
    em   = asset_emoji(pred.pair)
    name = pair_display(pred.pair)
    if atype == "TARGET_HIT":
        return (
            f"{'='*35}\n🎯 <b>TARGET HIT! ✅</b>\n{'='*35}\n\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"├ Entry:  <code>{fmt_price(pred.current_price,  pred.pair)}</code>\n"
            f"├ Target: <code>{fmt_price(pred.target_price,   pred.pair)}</code>\n"
            f"├ Exit:   <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"└ Profit: <b>{pred.pips_gained:.1f} pips 💰</b>\n\n"
            f"Quality: {pred.quality} | Confidence: {pred.confidence:.0f}%\n"
            f"{'='*35}\n"
            f"<i>Congratulations on the winning trade!\n"
            f"Remember: not every trade will win. "
            f"Risk management is everything.</i>"
        )
    elif atype == "INVALIDATED":
        return (
            f"{'='*35}\n❌ <b>SETUP INVALIDATED</b>\n{'='*35}\n\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"├ Entry: <code>{fmt_price(pred.current_price,      pred.pair)}</code>\n"
            f"├ Stop:  <code>{fmt_price(pred.invalidation_price, pred.pair)}</code>\n"
            f"├ Exit:  <code>{fmt_price(cp,                      pred.pair)}</code>\n"
            f"└ Loss:  <b>{abs(pred.pips_gained):.1f} pips</b>\n\n"
            f"{'='*35}\n"
            f"<i>This is normal. No system wins 100% of the time.\n"
            f"The stop-loss did its job. Capital is preserved.</i>"
        )
    else:
        p_diff = ((cp - pred.current_price) if pred.direction == "LONG"
                  else (pred.current_price - cp)) / pip_value(pred.pair)
        return (
            f"{'='*35}\n⏰ <b>PREDICTION EXPIRED</b>\n{'='*35}\n\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"├ Entry:   <code>{fmt_price(pred.current_price,  pred.pair)}</code>\n"
            f"├ Target:  <code>{fmt_price(pred.target_price,   pred.pair)}</code>\n"
            f"├ Current: <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"└ Move:    {p_diff:+.1f} pips\n\n"
            f"{'='*35}\n"
            f"<i>Setup expired after 24 hours.</i>"
        )


# ════════════════════════════════════════════════════════════════
#  MONITOR LOOP
# ════════════════════════════════════════════════════════════════

async def monitor_loop(app):
    log.info("Monitor started")
    while True:
        try:
            notes = await PT.check(app.bot)
            for n in notes:
                pred = n["pred"]
                if pred.chat_id:
                    await send_msg(app.bot, pred.chat_id,
                                   _alert_msg(pred, n["type"], n["cp"]),
                                   parse_mode="HTML")
            await asyncio.sleep(MONITORING_INTERVAL)
        except Exception as e:
            log.error(f"Monitor loop: {e}")
            await asyncio.sleep(MONITORING_INTERVAL)


# ════════════════════════════════════════════════════════════════
#  MARKET SUMMARIZER
# ════════════════════════════════════════════════════════════════

class MarketSummarizer:

    def summarize(self, instrument: str, of: OrderFlowData,
                  af: Optional[AdvancedOrderFlowData],
                  vp: Optional[VolumeProfile],
                  ob: Optional[OrderBookData],
                  pb: Optional[PositionBookData],
                  lz: Optional[List[LiquidityZone]] = None,
                  pred: Optional[Dict] = None) -> str:
        name = pair_display(instrument)
        ts   = datetime.now(timezone.utc).strftime("%H:%M UTC")
        parts: List[str] = []
        parts.append(
            f"{'='*35}\n🧠 <b>MARKET INTELLIGENCE BRIEF</b>\n{'='*35}\n"
            f"📍 <b>{name}</b>  |  🕐 {ts}\n\n"
            f"<i>Every data point explained in plain English.</i>"
        )
        parts.append(self._order_flow(of))
        if af:
            parts.append(self._vpin_section(af, of))
            parts.append(self._toxicity_section(af))
            parts.append(self._depth_section(af))
            parts.append(self._absorption_section(af))
        if vp:
            parts.append(self._vp_section(vp, of.price, instrument))
        if lz:
            lz_txt = self._lz_section(lz, of.price, instrument)
            if lz_txt:
                parts.append(lz_txt)
        if pb:
            parts.append(self._pb_section(pb, instrument))
        if ob:
            parts.append(self._ob_section(ob))
        parts.append(self._sm_vs_retail(of, af, pb))
        parts.append(self._conclusion(instrument, of, af, vp, pb, ob, pred))
        return "\n\n".join(parts)

    def _order_flow(self, of: OrderFlowData) -> str:
        cvd_t = (
            f"The <b>CVD is positive ({fmt_signed(of.cvd)})</b> — buyers have "
            f"collectively bought more than sellers have sold."
            if of.cvd > 0 else
            f"The <b>CVD is negative ({fmt_signed(of.cvd)})</b> — sellers have "
            f"been more aggressive than buyers."
        )
        imb_t = (
            "Order flow is balanced. Expect choppy price action."
            if abs(of.imbalance) < 5 else
            f"Buy orders outweigh sells by <b>{of.imbalance:+.1f}%</b>. "
            f"Buyers have the edge."
            if of.imbalance > 0 else
            f"Sell orders outweigh buys by <b>{abs(of.imbalance):.1f}%</b>. "
            f"Sellers in control."
        )
        mom_t = (
            "Delta momentum is flat — no acceleration."
            if abs(of.delta_momentum) < 100 else
            f"<b>Buying momentum accelerating</b> "
            f"({fmt_signed(of.delta_momentum)}). "
            f"Each wave of buyers is larger — growing bullish conviction."
            if of.delta_momentum > 0 else
            f"<b>Selling momentum accelerating</b> "
            f"({fmt_signed(of.delta_momentum)}). "
            f"Each wave of sellers intensifies — bearish conviction building."
        )
        vol_t = (
            f"Volume rising (+{of.volume_trend:.1f}%). Confirms real participation."
            if of.volume_trend > 15 else
            f"Volume declining ({of.volume_trend:.1f}%). Moves lack conviction."
            if of.volume_trend < -15 else
            "Volume stable."
        )
        vs     = "above" if of.price > of.vwap else "below"
        vwap_t = (f"Price is <b>{vs} the VWAP</b> — the institutional fair value. "
                  f"Creates a {'bullish' if vs == 'above' else 'bearish'} bias.")
        climax_t = ""
        if of.buying_climax:
            climax_t = ("\n\n⚠️ <b>BUYING CLIMAX:</b> Volume spiked but momentum is "
                        "fading. Institutions sell into retail euphoria — reversal risk HIGH.")
        elif of.selling_climax:
            climax_t = ("\n\n⚠️ <b>SELLING CLIMAX:</b> Volume spiked on selling but fading. "
                        "Smart money absorbs panic — watch for bounce.")
        return (f"{'─'*35}\n<b>📈 ORDER FLOW</b>\n{'─'*35}\n\n"
                f"{cvd_t}\n\n{imb_t}\n\n{mom_t}\n\n{vol_t}\n\n{vwap_t}{climax_t}")

    def _vpin_section(self, af: AdvancedOrderFlowData,
                      of: OrderFlowData) -> str:
        pct = af.vpin * 100
        if af.vpin >= 0.7:
            vt = (f"VPIN is <b>EXTREMELY HIGH at {pct:.1f}%</b>. A large portion of "
                  f"volume is from informed traders. Do NOT trade against this.")
        elif af.vpin >= 0.5:
            vt = (f"VPIN is <b>HIGH at {pct:.1f}%</b>. Smart money is actively "
                  f"participating and building positions.")
        elif af.vpin >= 0.3:
            vt = f"VPIN is <b>MODERATE at {pct:.1f}%</b>. Mixed retail and institutional flow."
        else:
            vt = (f"VPIN is <b>LOW at {pct:.1f}%</b>. Mostly retail-driven. "
                  f"Smart money largely on the sidelines.")
        sig = af.informed_trader_signal
        it  = (f"Informed signal: <b>{sig}</b> — "
               + ("smart money positioning LONG." if "BULLISH" in sig else
                  "smart money positioning SHORT." if "BEARISH" in sig else
                  "no clear smart money direction."))
        kl  = af.kyle_lambda
        kt  = ("Kyle's Lambda is very high — large informed orders moving price significantly."
               if abs(kl) >= 0.0001 else
               "Kyle's Lambda shows moderate price impact."
               if abs(kl) >= 0.00001 else
               "Kyle's Lambda minimal — liquid, efficient market.")
        return (f"{'─'*35}\n<b>🧠 VPIN AND SMART MONEY</b>\n{'─'*35}\n\n"
                f"{vt}\n\n{it}\n\n{kt}")

    def _toxicity_section(self, af: AdvancedOrderFlowData) -> str:
        pct = af.toxicity * 100
        if af.toxicity >= 0.7:
            tt = (f"Toxicity <b>VERY HIGH ({pct:.1f}%)</b>. Market makers are getting "
                  f"systematically picked off. Expect sudden violent moves.")
        elif af.toxicity >= 0.5:
            tt = (f"Toxicity <b>HIGH ({pct:.1f}%)</b>. Market makers under pressure. "
                  f"Sharp moves becoming more likely.")
        elif af.toxicity >= 0.3:
            tt = f"Toxicity <b>MODERATE ({pct:.1f}%)</b>. Some adverse selection, manageable."
        else:
            tt = (f"Toxicity <b>LOW ({pct:.1f}%)</b>. Clean, healthy order flow. "
                  f"Good conditions for trading.")
        a   = af.amihud_illiquidity
        lt  = (f"Market is <b>VERY ILLIQUID</b> (Amihud: {a:.1f}). Use wider stops."
               if a >= 100 else
               f"Liquidity <b>REDUCED</b> (Amihud: {a:.1f}). Be careful with size."
               if a >= 50 else
               f"Liquidity <b>GOOD</b> (Amihud: {a:.1f}). Normal conditions.")
        return (f"{'─'*35}\n<b>⚗️ TOXICITY AND LIQUIDITY</b>\n{'─'*35}\n\n"
                f"{tt}\n\n{lt}")

    def _depth_section(self, af: AdvancedOrderFlowData) -> str:
        d = af.market_depth_imbalance * 100
        if d >= 30:
            dt = (f"Order book <b>HEAVILY SKEWED TO BIDS (+{d:.1f}%)</b>. "
                  f"Strong institutional buying below price.")
        elif d >= 10:
            dt = f"<b>Moderate bid pressure (+{d:.1f}%)</b>. Buyers building a base."
        elif d <= -30:
            dt = (f"Order book <b>HEAVILY SKEWED TO ASKS ({d:.1f}%)</b>. "
                  f"Wall of sell orders above — strong overhead resistance.")
        elif d <= -10:
            dt = f"<b>Moderate ask pressure ({d:.1f}%)</b>. Sellers capping rallies."
        else:
            dt = "Order book depth balanced. No structural advantage for either side."
        it = ""
        if af.iceberg_count > 0:
            it = (f"\n\n🧊 <b>ICEBERG ORDERS: {af.iceberg_count} levels detected</b>\n"
                  f"Large institutional orders are deliberately hidden — a powerful signal.")
        else:
            it = "\n\nNo significant iceberg orders detected."
        return (f"{'─'*35}\n<b>📖 ORDER BOOK DEPTH AND HIDDEN ORDERS</b>\n"
                f"{'─'*35}\n\n{dt}{it}")

    def _absorption_section(self, af: AdvancedOrderFlowData) -> str:
        r  = af.absorption_ratio
        at = (f"Buying absorption <b>STRONG (ratio: {r:.2f})</b>. Every dip is bought."
              if r > 1.5 else
              f"Moderate buying absorption (ratio: {r:.2f})."
              if r > 1.2 else
              f"Selling absorption <b>STRONG (ratio: {r:.2f})</b>. Every rally is sold."
              if r < 0.67 else
              f"Moderate selling absorption (ratio: {r:.2f})."
              if r < 0.83 else
              f"Absorption balanced (ratio: {r:.2f}).")
        ag  = af.aggressor_side
        agt = ("<b>BUYERS are the aggressors</b> — actively lifting the ask with urgency."
               if ag == "BUYERS" else
               "<b>SELLERS are the aggressors</b> — hitting the bid with urgency."
               if ag == "SELLERS" else
               "Neither side clearly aggressing. Wait for a clear aggressor.")
        sc  = af.institutional_flow_score
        ist = (f"\n\n🏦 <b>Institutional Score: {sc:.0f}% — VERY HIGH</b>\n"
               f"Multiple institutional fingerprints present."
               if sc >= 70 else
               f"\n\n🏦 Institutional Score: {sc:.0f}% — Moderate activity."
               if sc >= 50 else
               f"\n\n🏦 Institutional Score: {sc:.0f}% — Mostly retail-driven.")
        return (f"{'─'*35}\n<b>💥 ABSORPTION AND AGGRESSION</b>\n{'─'*35}\n\n"
                f"{at}\n\n{agt}{ist}")

    def _vp_section(self, vp: VolumeProfile, price: float,
                    instrument: str) -> str:
        pt  = (f"The <b>POC is at {fmt_price(vp.poc, instrument)}</b>. "
               f"This is where the most volume has traded.")
        in_va = vp.val <= price <= vp.vah
        vat = (f"Value Area {fmt_price(vp.val, instrument)} to "
               f"{fmt_price(vp.vah, instrument)} — where 70% of volume traded. "
               f"Price is "
               + ("<b>INSIDE value area</b> — equilibrium."
                  if in_va else
                  "<b>OUTSIDE value area</b> — price discovery mode."))
        por = ("Price <b>ABOVE POC</b> — buyers control value."
               if price > vp.poc else
               "Price <b>BELOW POC</b> — sellers pushed below fair value."
               if price < vp.poc else
               "Price AT the POC — major decision expected.")
        return (f"{'─'*35}\n<b>📐 VOLUME PROFILE</b>\n{'─'*35}\n\n"
                f"{pt}\n\n{vat}\n\n{por}")

    def _lz_section(self, lz: List[LiquidityZone],
                    price: float, instrument: str) -> str:
        if not lz:
            return ""
        above = sorted([z for z in lz if z.price > price],
                       key=lambda x: x.price)[:3]
        below = sorted([z for z in lz if z.price <= price],
                       key=lambda x: x.price, reverse=True)[:3]
        if not above and not below:
            return ""
        msg = (f"{'─'*35}\n<b>🌊 LIQUIDITY ZONES</b>\n{'─'*35}\n\n"
               f"Liquidity zones are where stop-losses and pending orders "
               f"cluster. Institutions drive price to these levels.\n\n")
        if above:
            msg += "<b>Resistance / Supply Above:</b>\n"
            for z in above:
                dist = pips_diff(instrument, z.price - price)
                ic   = ("🧊" if "EQUAL" in z.zone_type else
                        "🔴" if "SUPPLY" in z.zone_type else "🌀")
                msg += (f"{ic} <b>{fmt_price(z.price, instrument)}</b> "
                        f"[{z.zone_type.replace('_',' ')}] +{dist:.0f} pips\n"
                        f"   <i>{z.description}</i>\n\n")
        if below:
            msg += "<b>Support / Demand Below:</b>\n"
            for z in below:
                dist = pips_diff(instrument, price - z.price)
                ic   = ("🧊" if "EQUAL" in z.zone_type else
                        "🟢" if "DEMAND" in z.zone_type else "🌀")
                msg += (f"{ic} <b>{fmt_price(z.price, instrument)}</b> "
                        f"[{z.zone_type.replace('_',' ')}] -{dist:.0f} pips\n"
                        f"   <i>{z.description}</i>\n\n")
        return msg.rstrip()

    def _pb_section(self, pb: PositionBookData, instrument: str) -> str:
        if pb.long_pct > 60:
            rt = (f"<b>{pb.long_pct:.1f}% of retail traders are LONG</b>. "
                  f"Smart money pushes price lower to trigger those stop-losses.")
        elif pb.short_pct > 60:
            rt = (f"<b>{pb.short_pct:.1f}% of retail traders are SHORT</b>. "
                  f"Classic short squeeze setup.")
        else:
            rt = (f"Retail positioning balanced — {pb.long_pct:.1f}% long "
                  f"vs {pb.short_pct:.1f}% short.")
        tt  = (f"<b>{pb.total_underwater:.1f}% of positions are underwater</b>. "
               f"Pain threshold: {fmt_price(pb.pain_threshold, instrument)}."
               if pb.total_underwater > 30 else
               f"Some trapped positions ({pb.total_underwater:.1f}% underwater)."
               if pb.total_underwater > 10 else
               "Few trapped traders. Limited stop-loss momentum.")
        sqt = ("⚡ <b>SQUEEZE POTENTIAL HIGH</b>"
               if pb.squeeze_potential == "HIGH" else
               "Moderate squeeze potential."
               if pb.squeeze_potential == "MODERATE" else
               "Low squeeze potential.")
        return (f"{'─'*35}\n<b>👥 POSITION BOOK</b>\n{'─'*35}\n\n"
                f"{rt}\n\n{tt}\n\n{sqt}")

    def _ob_section(self, ob: OrderBookData) -> str:
        if ob.breakout_bias == "UPWARD_PRESSURE":
            ot = (f"Pending orders show <b>UPWARD PRESSURE</b> "
                  f"({ob.long_pct:.1f}% long).")
        elif ob.breakout_bias == "DOWNWARD_PRESSURE":
            ot = (f"Pending orders show <b>DOWNWARD PRESSURE</b> "
                  f"({ob.short_pct:.1f}% short).")
        else:
            ot = "Pending orders balanced. Market likely to range."
        return (f"{'─'*35}\n<b>📋 PENDING ORDER LANDSCAPE</b>\n"
                f"{'─'*35}\n\n{ot}")

    def _sm_vs_retail(self, of: OrderFlowData,
                      af: Optional[AdvancedOrderFlowData],
                      pb: Optional[PositionBookData]) -> str:
        inst_l: List[str] = []
        ret_l:  List[str] = []
        if af:
            if af.vpin >= 0.5:
                inst_l.append(f"• Elevated informed trading (VPIN: {af.vpin:.0%})")
            if af.aggressor_side == "BUYERS":
                inst_l.append("• Institutions aggressively buying")
            elif af.aggressor_side == "SELLERS":
                inst_l.append("• Institutions aggressively selling")
            if af.iceberg_count > 0:
                inst_l.append(f"• {af.iceberg_count} hidden iceberg levels")
            if af.absorption_ratio > 1.5:
                inst_l.append("• Institutions absorbing all selling pressure")
            elif af.absorption_ratio < 0.67:
                inst_l.append("• Institutions distributing — absorbing every rally")
            if "BULLISH" in af.informed_trader_signal:
                inst_l.append("• Informed signal: BULLISH")
            elif "BEARISH" in af.informed_trader_signal:
                inst_l.append("• Informed signal: BEARISH")
        if pb:
            if pb.long_pct > 60:
                ret_l.append(f"• Retail {pb.long_pct:.0f}% LONG — dangerously crowded")
            elif pb.short_pct > 60:
                ret_l.append(f"• Retail {pb.short_pct:.0f}% SHORT — dangerously crowded")
            if pb.total_underwater > 20:
                ret_l.append(f"• {pb.total_underwater:.0f}% of retail positions losing")
            if pb.squeeze_potential == "HIGH":
                ret_l.append("• Retail in vulnerable squeeze position")
        if of.buying_climax:
            ret_l.append("• Retail chasing the top (buying climax)")
        elif of.selling_climax:
            ret_l.append("• Retail panic selling (selling climax)")
        ib = "\n".join(inst_l) if inst_l else "• No strong institutional signal"
        rb = "\n".join(ret_l)  if ret_l  else "• No extreme retail signal"
        return (f"{'─'*35}\n<b>🏦 INSTITUTIONS vs 👥 RETAIL</b>\n"
                f"{'─'*35}\n\n"
                f"<b>What Institutions Are Doing:</b>\n{ib}\n\n"
                f"<b>What Retail Traders Are Doing:</b>\n{rb}")

    def _conclusion(self, instrument: str, of: OrderFlowData,
                    af: Optional[AdvancedOrderFlowData],
                    vp: Optional[VolumeProfile],
                    pb: Optional[PositionBookData],
                    ob: Optional[OrderBookData],
                    pred: Optional[Dict]) -> str:
        name = pair_display(instrument)
        bull = bear = 0
        if of.cvd > 0:                  bull += 2
        else:                            bear += 2
        if of.imbalance > 5:            bull += 1
        elif of.imbalance < -5:          bear += 1
        if of.delta_momentum > 0:       bull += 1
        else:                            bear += 1
        if of.price > of.vwap:          bull += 1
        else:                            bear += 1
        if af:
            if "BULLISH" in af.informed_trader_signal:   bull += 3
            elif "BEARISH" in af.informed_trader_signal: bear += 3
            if af.aggressor_side == "BUYERS":            bull += 2
            elif af.aggressor_side == "SELLERS":         bear += 2
            if af.market_depth_imbalance > 0.2:          bull += 1
            elif af.market_depth_imbalance < -0.2:       bear += 1
        if pb:
            if pb.contrarian_signal == "BULLISH":        bull += 2
            elif pb.contrarian_signal == "BEARISH":      bear += 2
        if ob:
            if ob.pressure_direction == "BULLISH":       bull += 1
            elif ob.pressure_direction == "BEARISH":     bear += 1
        tot   = bull + bear
        bpct  = bull / tot * 100 if tot > 0 else 50.0
        if bpct >= 65:
            bias  = "📈 <b>BULLISH BIAS</b>"
            bt    = f"Weight of evidence points UP. {bull}/{tot} data points favour buyers."
            tgt_t = "bulls targeting recent resistance and higher."
        elif bpct <= 35:
            bias  = "📉 <b>BEARISH BIAS</b>"
            bt    = f"Weight of evidence points DOWN. {bear}/{tot} data points favour sellers."
            tgt_t = "bears targeting recent support and lower."
        else:
            bias  = "↔️ <b>NEUTRAL / MIXED</b>"
            bt    = "Evidence is mixed — no clear directional edge."
            tgt_t = "price likely to range until a catalyst emerges."
        tgt_sec = ""
        if pred and pred.get("has_setup"):
            d, tgt, stp, cf = (pred["direction"], pred["target"],
                                pred["stop"], pred["confidence"])
            tgt_sec = (
                f"\n\n<b>🎯 Algorithm Target:</b>\n"
                f"System expects {name} to move "
                f"{'UP' if d == 'LONG' else 'DOWN'} to "
                f"<code>{fmt_price(tgt, instrument)}</code> "
                f"({cf:.0f}% confidence).\n"
                f"Invalidated if price closes "
                f"{'below' if d == 'LONG' else 'above'} "
                f"<code>{fmt_price(stp, instrument)}</code>.\n\n"
                f"<b>⚠️ High-confidence setups still fail.</b> "
                f"Always use your stop-loss."
            )
        elif vp:
            tgt_sec = (
                f"\n\n<b>🎯 Key Levels to Watch:</b>\n"
                f"• POC: {fmt_price(vp.poc, instrument)}\n"
                f"• VAH: {fmt_price(vp.vah, instrument)}\n"
                f"• VAL: {fmt_price(vp.val, instrument)}"
            )
        risk_t = ""
        if af and af.toxicity >= 0.5:
            risk_t = ("\n\n⚠️ <b>ELEVATED RISK:</b> High toxicity detected. "
                      "Reduce size and use wider stops.")
        return (
            f"{'='*35}\n<b>📊 CONCLUSION</b>\n{'='*35}\n\n"
            f"{bias}\n\n{bt}\n\nBased on all data, {tgt_t}"
            f"{tgt_sec}{risk_t}\n\n"
            f"<i>📚 No analysis is 100% correct. Profitable trading is "
            f"about disciplined risk management.</i>"
        )


MS = MarketSummarizer()


# ════════════════════════════════════════════════════════════════
#  AUTO ALERT ENGINE
# ════════════════════════════════════════════════════════════════

class AlertEngine:

    def __init__(self):
        self.last_alert: Dict[str, datetime] = {}
        self.cooldown = timedelta(hours=4)

    def _can_alert(self, pair: str, tf: str) -> bool:
        key  = f"{pair}_{tf}"
        last = self.last_alert.get(key)
        return last is None or datetime.now(timezone.utc) - last > self.cooldown

    def _mark(self, pair: str, tf: str):
        self.last_alert[f"{pair}_{tf}"] = datetime.now(timezone.utc)

    async def scan(self, app) -> int:
        active = SUB_MANAGER.get_all_active()
        if not active:
            return 0
        pairs_map: Dict[str, str] = {}
        for _, sub in active:
            for p in sub.get("pairs", []):
                pairs_map[p] = sub.get("timeframe", "H1")

        sent = 0
        async with aiohttp.ClientSession() as session:
            for pair, tf in pairs_map.items():
                if not self._can_alert(pair, tf):
                    continue
                try:
                    dd  = await full_analysis(session, pair, tf)
                    of  = dd["of"]; af = dd["af"]; lz = dd["lz"]
                    pred = QB.generate(
                        dd["candles"], of, dd["vp"], dd["ob"],
                        dd["pb"], dd["cs"], af, pair, lz)
                    if not pred.get("has_setup"):
                        continue
                    aligned = (pred["bull_count"] if pred["direction"] == "LONG"
                               else pred["bear_count"])
                    if pred["confidence"] < MIN_ALERT_CONFIDENCE:
                        continue
                    if aligned < MIN_ALERT_FACTORS:
                        continue
                    if af and af.institutional_flow_score < MIN_ALERT_INST_SCORE:
                        continue
                    self._mark(pair, tf)
                    sn   = session_name(datetime.now(timezone.utc).hour)
                    sim  = PATTERN_MEMORY.find_similar(
                        pair, pred["direction"], af, of, sn)
                    msg  = self._format(pair, tf, pred, of, af, lz, sim, sn)
                    for chat_id, sub in active:
                        if pair in sub.get("pairs", []):
                            ok = await send_msg(app.bot, chat_id, msg,
                                                parse_mode="HTML")
                            if ok:
                                SUB_MANAGER.record_alert(chat_id)
                                sent += 1
                    # Track prediction
                    pid  = (f"AUTO_{pair}_{tf}_"
                            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
                    pobj = QuantPrediction(
                        prediction_id=pid, pair=pair, timeframe=tf,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        current_price=of.price, direction=pred["direction"],
                        target_price=pred["target"],
                        invalidation_price=pred["stop"],
                        confidence=pred["confidence"],
                        quality=pred["quality"],
                        reasons=pred["reasons"],
                        key_levels=pred["key_levels"],
                        factors_aligned=aligned,
                        features=pred.get("features", []),
                        status="ACTIVE", chat_id=None,
                        ml_confidence=pred.get("ml_prob", 0.0),
                        ml_used=ML_ENGINE.is_trained
                    )
                    QB.add(pobj)
                    PT.add(pobj)
                    PATTERN_MEMORY.record_pattern(pobj, of, af, sn)
                    await asyncio.sleep(2.0)
                except Exception as e:
                    log.error(f"Alert scan {pair}: {e}")
        return sent

    def _format(self, pair: str, tf: str, pred: Dict,
                of: OrderFlowData,
                af: Optional[AdvancedOrderFlowData],
                lz: List[LiquidityZone], sim: Dict,
                sn: str) -> str:
        name  = pair_display(pair)
        em    = asset_emoji(pair)
        ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        d     = pred["direction"]
        de    = "🟢" if d == "LONG" else "🔴"
        dw    = "BUY" if d == "LONG" else "SELL"
        cp    = of.price
        q     = pred["quality"]
        qe    = ("🔥" if "A+" in q else "✅" if "A " in q or q == "A (Good)"
                 else "⚡" if "B" in q else "⚠️")
        rew   = abs(pred["target"] - cp)
        risk  = abs(cp - pred["stop"])
        rr    = rew / risk if risk > 0 else 0.0
        rp    = pips_diff(pair, rew)
        riskp = pips_diff(pair, risk)
        ml_p  = pred.get("ml_prob",  0.0)
        ml_l  = pred.get("ml_label", "")
        msg   = (
            f"{'='*35}\n🚨 <b>INSTITUTIONAL ALERT</b> 🚨\n{'='*35}\n\n"
            f"{em} <b>{name}</b> | "
            f"{TIMEFRAMES.get(tf, {}).get('emoji', '')} {tf}\n"
            f"🕐 {ts} | Session: {sn}\n\n"
            f"{'─'*35}\n"
            f"{de} <b>{dw}</b> | {qe} <b>{q}</b>\n"
            f"📊 Confidence: <b>{pred['confidence']:.0f}%</b>\n"
        )
        if ml_l and "No ML" not in ml_l:
            msg += f"🧠 ML: <b>{ml_p:.0%}</b> — {ml_l}\n"
        msg += (
            f"{'─'*35}\n\n"
            f"<b>📍 Trade Levels:</b>\n"
            f"├ Entry:    <code>{fmt_price(cp,              pair)}</code>\n"
            f"├ 🎯 Target: <code>{fmt_price(pred['target'], pair)}</code>\n"
            f"└ ⛔ Stop:   <code>{fmt_price(pred['stop'],   pair)}</code>\n\n"
            f"<b>⚖️ Risk/Reward:</b>\n"
            f"├ Potential: +{rp:.1f} pips\n"
            f"├ Risk:      -{riskp:.1f} pips\n"
            f"└ R/R:       1:{rr:.1f}\n\n"
            f"<b>Why This Alert Fired:</b>\n"
        )
        for i, r in enumerate(pred["reasons"][:5], 1):
            msg += f"  {i}. {r}\n"
        msg += "\n"
        if af:
            sm_e = ("🔥" if "VERY HIGH" in af.smart_money_activity else
                    "⚠️" if "HIGH" in af.smart_money_activity else "📊")
            msg += (
                f"<b>{sm_e} Institutional Activity:</b>\n"
                f"├ Smart Money: <b>{af.smart_money_activity}</b>\n"
                f"├ VPIN:        <code>{af.vpin:.2%}</code> ({af.vpin_level})\n"
                f"├ Informed:    <b>{af.informed_trader_signal}</b>\n"
                f"├ Aggressor:   <b>{af.aggressor_side}</b>\n"
                f"└ Inst. Score: <b>{af.institutional_flow_score:.0f}%</b>\n\n"
            )
            if af.iceberg_count > 0:
                msg += f"🧊 <b>{af.iceberg_count} hidden iceberg orders detected</b>\n\n"
        if lz:
            nearby = [z for z in lz
                      if pips_diff(pair, abs(z.price - cp)) < 100][:3]
            if nearby:
                msg += "<b>🌊 Nearby Liquidity Zones:</b>\n"
                for z in nearby:
                    side = "above" if z.price > cp else "below"
                    dist = pips_diff(pair, abs(z.price - cp))
                    msg += (f"  • {z.zone_type.replace('_', ' ')} {side} "
                            f"at {fmt_price(z.price, pair)} ({dist:.0f} pips)\n")
                msg += "\n"
        if sim.get("found"):
            msg += f"{'─'*35}\n{sim['narrative']}\n\n"
        si   = SA.INFO.get(sn, {})
        best = si.get("best", [])
        msg += f"<b>{si.get('emoji', '🕐')} Session:</b> {si.get('name', '')}\n"
        if pair in best:
            msg += f"✅ This is one of the BEST sessions for {name}.\n\n"
        else:
            msg += "\n"
        msg += (
            f"{'='*35}\n"
            f"<b>⚠️ Risk Reminder:</b>\n"
            f"This is a high-confidence institutional setup but no signal "
            f"is 100% accurate. Stop-loss at "
            f"<code>{fmt_price(pred['stop'], pair)}</code>. "
            f"Risk max 1-2% of your account.\n"
            f"{'='*35}"
        )
        return msg


AE = AlertEngine()


async def alert_loop(app):
    log.info("Alert scanner started")
    while True:
        try:
            await asyncio.sleep(ALERT_SCAN_INTERVAL)
            n = await AE.scan(app)
            if n > 0:
                log.info(f"Alerts sent: {n}")
        except Exception as e:
            log.error(f"Alert loop: {e}")
            await asyncio.sleep(60)


# ════════════════════════════════════════════════════════════════
#  FULL ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════

async def full_analysis(session: aiohttp.ClientSession,
                        pair: str, tf: str) -> Dict:
    tf_cfg = TIMEFRAMES[tf]
    r      = await asyncio.gather(
        fetch_candles(session, pair, tf, tf_cfg["candles"]),
        fetch_order_book(session, pair),
        fetch_position_book(session, pair),
        return_exceptions=True
    )
    candles = r[0] if not isinstance(r[0], Exception) else []
    ob_raw  = r[1] if not isinstance(r[1], Exception) else None
    pb_raw  = r[2] if not isinstance(r[2], Exception) else None
    if not candles or len(candles) < 30:
        raise ValueError("Insufficient candle data")
    of = OF_ENGINE.analyze(candles)
    vp = calculate_volume_profile(candles)
    ob = analyze_order_book(ob_raw, of.price)
    pb = analyze_position_book(pb_raw, of.price)
    af = ADVANCED_OF.analyze_all(candles, ob_raw)
    lz = LIQUIDITY_DETECTOR.detect_all(candles, pair)
    cs = None
    if pair in FOREX_PAIRS:
        cs = await STRENGTH_CACHE.get(session)
    return {"candles": candles, "of": of, "vp": vp, "ob": ob,
            "pb": pb, "af": af, "lz": lz, "cs": cs, "ob_raw": ob_raw}


# ════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════

def chart_prediction(candles: List[Candle], instrument: str,
                     pred: Dict, vp: Optional[VolumeProfile],
                     of: OrderFlowData,
                     af: Optional[AdvancedOrderFlowData],
                     lz: Optional[List[LiquidityZone]] = None) -> io.BytesIO:
    fig  = plt.figure(figsize=(14, 12), facecolor='#0d1117')
    gs   = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.8], hspace=0.12)
    C    = {'bg':'#0d1117','text':'#c9d1d9','grid':'#21262d',
            'up':'#3fb950','dn':'#f85149','tgt':'#3fb950',
            'stp':'#f85149','ent':'#f0883e',
            'poc':'#a371f7','vwap':'#58a6ff','sm':'#ffd700',
            'dem':'#3fb950','sup':'#f85149','fvg':'#58a6ff'}
    disp = candles[-70:] if len(candles) > 70 else candles
    n    = len(disp)

    ax_p = fig.add_subplot(gs[0])
    ax_p.set_facecolor(C['bg'])
    for i, c in enumerate(disp):
        col = C['up'] if c.is_bullish else C['dn']
        ax_p.plot([i, i], [c.low, c.high],   color=col, linewidth=0.8)
        ax_p.plot([i, i], [c.open, c.close], color=col, linewidth=3.5)

    cp = candles[-1].close
    for yv, lbl, col, ls in [
        (cp,             f"ENTRY\n{fmt_price(cp,             instrument)}", C['ent'], '-'),
        (pred['target'], f"TARGET\n{fmt_price(pred['target'], instrument)}", C['tgt'], '--'),
        (pred['stop'],   f"STOP\n{fmt_price(pred['stop'],     instrument)}", C['stp'], '--'),
    ]:
        ax_p.axhline(y=yv, color=col, linestyle=ls, linewidth=2, alpha=0.9)
        ax_p.text(n + 1, yv, lbl, color=col, fontsize=8,
                  fontweight='bold', va='center')

    ax_p.axhline(y=of.vwap, color=C['vwap'], linestyle=':', linewidth=1.5, alpha=0.7)
    if vp:
        ax_p.axhline(y=vp.poc, color=C['poc'], linestyle=':', linewidth=1.5, alpha=0.7)

    if lz:
        lo_ = min(c.low  for c in disp)
        hi_ = max(c.high for c in disp)
        for z in lz[:6]:
            if lo_ <= z.price <= hi_:
                zc = (C['dem'] if z.zone_type == "DEMAND" else
                      C['sup'] if z.zone_type == "SUPPLY" else C['fvg'])
                if z.upper > 0 and z.lower > 0:
                    ax_p.axhspan(z.lower, z.upper, alpha=0.07, color=zc)
                ax_p.axhline(y=z.price, color=zc, linestyle=':', linewidth=0.8, alpha=0.4)

    ac = C['up'] if pred['direction'] == 'LONG' else C['dn']
    ax_p.annotate('', xy=(n - 5, pred['target']), xytext=(n - 5, cp),
                  arrowprops=dict(arrowstyle='->', color=ac, lw=3))

    ml_l = pred.get('ml_label', '')
    de   = "🟢" if pred['direction'] == 'LONG' else "🔴"
    title = (f"{asset_emoji(instrument)} {pair_display(instrument)} | "
             f"{de} {pred['direction']} | {pred['quality']}")
    if ml_l and "No ML" not in ml_l:
        title += f" | {ml_l}"
    ax_p.set_title(title, color=C['text'], fontsize=12, fontweight='bold', pad=12)
    ax_p.tick_params(colors=C['text'], labelsize=9)
    ax_p.grid(True, alpha=0.15, color=C['grid'])
    ax_p.set_xlim(-2, n + 15)

    ax_d = fig.add_subplot(gs[1])
    ax_d.set_facecolor(C['bg'])
    deltas = [c.delta for c in disp]
    ax_d.bar(range(n), deltas,
             color=[C['up'] if d >= 0 else C['dn'] for d in deltas],
             alpha=0.7, width=0.8)
    ax_d.axhline(y=0, color=C['grid'], linewidth=1)
    ax_d.set_title('Order Flow Delta', color=C['text'], fontsize=10, pad=5)
    ax_d.tick_params(colors=C['text'], labelsize=8)
    ax_d.grid(True, alpha=0.15, color=C['grid'])
    ax_d.set_xlim(-2, n + 15)

    ax_c = fig.add_subplot(gs[2])
    ax_c.set_facecolor(C['bg'])
    cvd     = np.cumsum(deltas)
    cvd_col = C['up'] if len(cvd) > 0 and cvd[-1] > cvd[0] else C['dn']
    ax_c.fill_between(range(n), cvd, alpha=0.3, color=cvd_col)
    ax_c.plot(range(n), cvd, color=cvd_col, linewidth=1.5)
    ax_c.set_title('CVD', color=C['text'], fontsize=10, pad=5)
    ax_c.tick_params(colors=C['text'], labelsize=8)
    ax_c.grid(True, alpha=0.15, color=C['grid'])
    ax_c.set_xlim(-2, n + 15)

    ax_s = fig.add_subplot(gs[3])
    ax_s.set_facecolor(C['bg'])
    if af:
        ss = []
        for i in range(n):
            wc = disp[max(0, i - 10):i + 1]
            bv = sum(c.buy_volume  for c in wc)
            sv = sum(c.sell_volume for c in wc)
            tv = bv + sv
            ss.append(abs(bv - sv) / tv * 100 if tv > 0 else 0.0)
        ax_s.fill_between(range(n), ss, alpha=0.5, color=C['sm'])
        ax_s.plot(range(n), ss, color=C['sm'], linewidth=1.5)
        ax_s.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_s.set_title(
            f"Smart Money | VPIN: {af.vpin_level} | {af.smart_money_activity}",
            color=C['sm'], fontsize=10, pad=5)
    else:
        ax_s.text(0.5, 0.5, 'Smart Money Analysis', ha='center', va='center',
                  color=C['text'], transform=ax_s.transAxes)
    ax_s.tick_params(colors=C['text'], labelsize=8)
    ax_s.grid(True, alpha=0.15, color=C['grid'])
    ax_s.set_xlim(-2, n + 15)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.98, 0.01, f'Generated: {ts}', ha='right',
             color=C['text'], fontsize=8, alpha=0.7)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor=C['bg'], edgecolor='none', bbox_inches='tight')
    buf.seek(0); plt.close(fig); gc.collect()
    return buf


def chart_strength(strengths: Dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    C  = {'up':'#3fb950','dn':'#f85149','text':'#c9d1d9','grid':'#21262d'}
    sc = sorted(strengths.values(), key=lambda x: x.strength, reverse=True)
    cu = [cs.currency for cs in sc]
    va = [cs.strength for cs in sc]
    bc = [C['up'] if v > 0 else C['dn'] for v in va]
    yp = np.arange(len(cu))
    ax.barh(yp, va, color=bc, alpha=0.85, height=0.6)
    ax.set_yticks(yp)
    ax.set_yticklabels(cu, color=C['text'], fontsize=13, fontweight='bold')
    ax.axvline(x=0, color=C['grid'], linestyle='-', linewidth=2)
    for i, (c, v) in enumerate(zip(cu, va)):
        em = "💪" if v > 0.2 else ("💀" if v < -0.2 else "➖")
        ax.text(v + (0.08 if v >= 0 else -0.08), i,
                f'{em} {v:+.2f}%', va='center',
                ha='left' if v >= 0 else 'right',
                color=C['text'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Strength (%)', color=C['text'], fontsize=12)
    ax.set_title('💱 Currency Strength (24H)',
                 color=C['text'], fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(colors=C['text'], labelsize=10)
    ax.grid(True, alpha=0.2, color=C['grid'], axis='x')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110,
                facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    buf.seek(0); plt.close(fig); gc.collect()
    return buf


def chart_no_setup(candles: List[Candle], instrument: str) -> io.BytesIO:
    fig  = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    gs   = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.12)
    C    = {'bg':'#0d1117','text':'#c9d1d9','grid':'#21262d',
            'up':'#3fb950','dn':'#f85149','neu':'#f0883e'}
    disp = candles[-70:] if len(candles) > 70 else candles
    n    = len(disp)
    ax_p = fig.add_subplot(gs[0])
    ax_p.set_facecolor(C['bg'])
    for i, c in enumerate(disp):
        col = C['up'] if c.is_bullish else C['dn']
        ax_p.plot([i, i], [c.low, c.high],   color=col, linewidth=0.8)
        ax_p.plot([i, i], [c.open, c.close], color=col, linewidth=3.5)
    ax_p.set_title(
        f"{asset_emoji(instrument)} {pair_display(instrument)} | ⚪ NO CLEAR SETUP",
        color=C['neu'], fontsize=14, fontweight='bold', pad=15)
    ax_p.tick_params(colors=C['text'], labelsize=9)
    ax_p.grid(True, alpha=0.15, color=C['grid'])
    ax_p.set_xlim(-2, n + 5)
    ax_d = fig.add_subplot(gs[1])
    ax_d.set_facecolor(C['bg'])
    deltas = [c.delta for c in disp]
    ax_d.bar(range(n), deltas,
             color=[C['up'] if d >= 0 else C['dn'] for d in deltas],
             alpha=0.7, width=0.8)
    ax_d.axhline(y=0, color=C['grid'], linewidth=1)
    ax_d.set_title('Order Flow Delta', color=C['text'], fontsize=10, pad=5)
    ax_d.tick_params(colors=C['text'], labelsize=8)
    ax_d.grid(True, alpha=0.15, color=C['grid'])
    ax_d.set_xlim(-2, n + 5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor=C['bg'], edgecolor='none', bbox_inches='tight')
    buf.seek(0); plt.close(fig); gc.collect()
    return buf


# ════════════════════════════════════════════════════════════════
#  MESSAGE FORMATTERS
# ════════════════════════════════════════════════════════════════

def msg_prediction(instrument: str, tf: str, pred: Dict,
                   of: OrderFlowData, ob: Optional[OrderBookData],
                   pb: Optional[PositionBookData],
                   af: Optional[AdvancedOrderFlowData],
                   lz: Optional[List[LiquidityZone]] = None,
                   sim: Optional[Dict] = None) -> str:
    name  = pair_display(instrument)
    em    = asset_emoji(instrument)
    ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    ti    = TIMEFRAMES.get(tf, {})
    d     = pred['direction']
    de    = "🟢" if d == "LONG" else "🔴"
    dw    = "BUY"  if d == "LONG" else "SELL"
    cp    = of.price
    q     = pred['quality']
    qe    = ("🔥" if "A+" in q else "✅" if "A " in q or q == "A (Good)"
             else "⚡" if "B" in q else "⚠️")
    rew   = abs(pred['target'] - cp)
    risk  = abs(cp - pred['stop'])
    rr    = rew / risk if risk > 0 else 0.0
    ml_p  = pred.get('ml_prob',  0.0)
    ml_l  = pred.get('ml_label', '')
    msg   = (
        f"{'='*35}\n🎯 <b>PRICE PREDICTION</b>\n{'='*35}\n\n"
        f"{em} <b>{name}</b> | {ti.get('emoji','')} {ti.get('label',tf)}\n"
        f"🕐 {ts}\n\n"
        f"{'─'*35}\n"
        f"{de} <b>{dw}</b> | {qe} <b>{q}</b>\n"
        f"📊 Confidence: <b>{pred['confidence']:.0f}%</b>\n"
    )
    if ml_l and "No ML" not in ml_l:
        msg += f"🧠 ML: <b>{ml_p:.0%}</b> — {ml_l}\n"
    msg += (
        f"{'─'*35}\n\n"
        f"💰 Entry:       <code>{fmt_price(cp,              instrument)}</code>\n"
        f"🎯 Target:      <code>{fmt_price(pred['target'],  instrument)}</code>\n"
        f"⛔ Stop Loss:   <code>{fmt_price(pred['stop'],    instrument)}</code>\n"
        f"⚖️ R/R:         1:{rr:.1f}\n\n"
        f"<b>Factors ({pred['bull_count']}🟢 vs {pred['bear_count']}🔴):</b>\n"
    )
    for i, r in enumerate(pred['reasons'][:6], 1):
        msg += f"  {i}. {r}\n"
    msg += "\n"
    if pred.get('key_levels'):
        msg += "<b>Key Levels:</b>\n"
        for lv in pred['key_levels'][:4]:
            msg += f"  • {lv['type']}: <code>{fmt_price(lv['price'], instrument)}</code>\n"
        msg += "\n"
    msg += (
        f"{'─'*35}\n<b>📊 Order Flow:</b>\n"
        f"├ CVD:      <code>{fmt_signed(of.cvd)}</code>\n"
        f"├ Imbalance:<code>{of.imbalance:+.1f}%</code>\n"
        f"├ Momentum: <code>{fmt_signed(of.delta_momentum)}</code>\n"
        f"└ Vol Trend:<code>{of.volume_trend:+.1f}%</code>\n\n"
    )
    if af:
        msg += (
            f"{'─'*35}\n<b>🧠 Smart Money:</b>\n"
            f"├ VPIN:     <code>{af.vpin:.2%}</code> ({af.vpin_level})\n"
            f"├ Informed: <b>{af.informed_trader_signal}</b>\n"
            f"├ Aggressor:<b>{af.aggressor_side}</b>\n"
            f"├ SM:       <b>{af.smart_money_activity}</b>\n"
            f"└ Inst:     <b>{af.institutional_flow_score:.0f}%</b>\n\n"
        )
        if af.iceberg_count > 0:
            msg += f"🧊 <b>{af.iceberg_count} iceberg orders detected</b>\n\n"
    if lz:
        nearby = [z for z in lz
                  if pips_diff(instrument, abs(z.price - cp)) < 80][:3]
        if nearby:
            msg += "<b>🌊 Nearby Liquidity Zones:</b>\n"
            for z in nearby:
                side = "above" if z.price > cp else "below"
                dist = pips_diff(instrument, abs(z.price - cp))
                msg += (f"  • {z.zone_type.replace('_',' ')} {side} "
                        f"— {fmt_price(z.price, instrument)} ({dist:.0f}p)\n")
            msg += "\n"
    if ob:
        msg += (
            f"<b>📖 Order Book:</b>\n"
            f"├ Longs:    {ob.long_pct:.1f}%\n"
            f"├ Shorts:   {ob.short_pct:.1f}%\n"
            f"└ Pressure: {ob.pressure_direction}\n\n"
        )
    if pb:
        msg += (
            f"<b>👥 Position Book:</b>\n"
            f"├ Longs:  {pbar(pb.long_pct)} {pb.long_pct:.0f}%\n"
            f"├ Shorts: {pbar(pb.short_pct)} {pb.short_pct:.0f}%\n"
            f"├ Trapped Longs:  {pb.trapped_longs_pct:.1f}%\n"
            f"├ Trapped Shorts: {pb.trapped_shorts_pct:.1f}%\n"
            f"└ Squeeze: {pb.squeeze_potential}\n\n"
        )
    if sim and sim.get("found"):
        msg += f"{'─'*35}\n{sim['narrative']}\n\n"
    msg += (
        f"{'='*35}\n"
        f"<i>✅ Live tracking ON — alerts when target is hit or stop triggered.\n\n"
        f"⚠️ Being wrong is okay. Use your stop-loss every time.</i>\n"
    )
    return msg


def msg_no_setup(instrument: str, tf: str, of: OrderFlowData,
                 bc: int, brc: int,
                 af: Optional[AdvancedOrderFlowData] = None) -> str:
    name  = pair_display(instrument)
    em    = asset_emoji(instrument)
    ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    ti    = TIMEFRAMES.get(tf, {})
    msg   = (
        f"{'='*35}\n⚪ <b>NO CLEAR SETUP</b>\n{'='*35}\n\n"
        f"{em} <b>{name}</b> | {ti.get('emoji','')} {ti.get('label',tf)}\n"
        f"🕐 {ts}\n\n"
        f"{'─'*35}\n<b>Why no prediction?</b>\n{'─'*35}\n\n"
        f"Confluence factors are mixed:\n"
        f"├ 🟢 Bullish: {bc}\n"
        f"├ 🔴 Bearish: {brc}\n"
        f"└ Minimum:   {MIN_CONFLUENCE_FACTORS}\n\n"
        f"{'─'*35}\n<b>📊 Current Flow:</b>\n"
        f"├ CVD:       <code>{fmt_signed(of.cvd)}</code>\n"
        f"├ Imbalance: <code>{of.imbalance:+.1f}%</code>\n"
        f"└ Momentum:  <code>{fmt_signed(of.delta_momentum)}</code>\n\n"
    )
    if af:
        msg += (
            f"<b>🧠 Smart Money:</b>\n"
            f"├ VPIN:    <code>{af.vpin:.2%}</code> ({af.vpin_level})\n"
            f"├ SM:      {af.smart_money_activity}\n"
            f"└ Inst:    {af.institutional_flow_score:.0f}%\n\n"
        )
    msg += (
        f"{'='*35}\n"
        f"<i>💡 Patience is an edge. Wait for all factors to align "
        f"before committing capital.</i>\n"
    )
    return msg


def msgs_analysis(instrument: str, tf: str, of: OrderFlowData,
                  vp: Optional[VolumeProfile], ob: Optional[OrderBookData],
                  pb: Optional[PositionBookData],
                  af: Optional[AdvancedOrderFlowData] = None,
                  lz: Optional[List[LiquidityZone]] = None) -> List[str]:
    msgs: List[str] = []
    name = pair_display(instrument)
    ts   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    ti   = TIMEFRAMES.get(tf, {})
    vs   = "above" if of.price > of.vwap else "below"
    m1   = (
        f"{'='*32}\n📊 <b>{name} ANALYSIS</b>\n{'='*32}\n"
        f"⏰ {ts} | {ti.get('emoji','')} {ti.get('label',tf)}\n\n"
        f"💰 <b>Price:</b> <code>{fmt_price(of.price, instrument)}</code>\n"
        f"📐 <b>VWAP:</b>  <code>{fmt_price(of.vwap,  instrument)}</code>\n\n"
        f"<i>Price {vs} VWAP = {'bullish' if vs=='above' else 'bearish'} bias</i>\n\n"
        f"{'─'*32}\n<b>📈 ORDER FLOW</b>\n{'─'*32}\n\n"
        f"{'🟢' if of.cvd>0 else '🔴'} "
        f"<b>{'BUYERS' if of.cvd>0 else 'SELLERS'} dominant</b>\n\n"
        f"├ Buy Vol:   <code>{fmt_num(of.buy_volume)}</code> ({of.buy_pct:.1f}%)\n"
        f"├ Sell Vol:  <code>{fmt_num(of.sell_volume)}</code> ({of.sell_pct:.1f}%)\n"
        f"├ Imbalance: <code>{fmt_pct(of.imbalance)}</code>\n"
        f"├ CVD:       <code>{fmt_signed(of.cvd)}</code>\n"
        f"├ Momentum:  <code>{fmt_signed(of.delta_momentum)}</code>\n"
        f"└ Vol Trend: <code>{fmt_pct(of.volume_trend)}</code>\n"
    )
    msgs.append(m1)
    if af:
        m2 = (
            f"{'─'*32}\n<b>🧠 SMART MONEY INTELLIGENCE</b>\n{'─'*32}\n\n"
            f"<b>VPIN (Informed Trading):</b>\n"
            f"├ VPIN:  <code>{af.vpin:.2%}</code>\n"
            f"├ Level: <b>{af.vpin_level}</b>\n"
            f"└ High VPIN = Smart money active\n\n"
            f"<b>Kyle's Lambda:</b>\n"
            f"├ Value: <code>{fmt_sci(af.kyle_lambda)}</code>\n"
            f"└ {af.kyle_lambda_interpretation}\n\n"
            f"<b>Toxicity:</b>\n"
            f"├ Value: <code>{af.toxicity:.2%}</code>\n"
            f"├ Level: <b>{af.toxicity_level}</b>\n"
            f"└ High = Reversal risk\n\n"
            f"<b>Liquidity:</b>\n"
            f"├ Amihud: <code>{af.amihud_illiquidity:.2f}</code>\n"
            f"└ Status: <b>{af.liquidity_level}</b>\n\n"
            f"<b>Order Book Depth:</b>\n"
            f"├ Imbalance: <code>{af.market_depth_imbalance:+.2%}</code>\n"
            f"└ Bias: <b>{af.depth_bias}</b>\n\n"
        )
        if af.iceberg_count > 0:
            m2 += f"🧊 <b>{af.iceberg_count} iceberg orders detected</b>\n\n"
        m2 += (
            f"<b>Absorption:</b>\n"
            f"├ Ratio: <code>{af.absorption_ratio:.2f}</code>\n"
            f"└ {'Buying' if af.absorption_ratio > 1 else 'Selling'} absorption\n\n"
            f"<b>Activity:</b>\n"
            f"├ Arrival Rate:  <code>{af.trade_arrival_rate:+.2f}σ</code>\n"
            f"├ Clustering:    <code>{af.volume_clustering:.2%}</code>\n"
            f"└ Aggressor: <b>{af.aggressor_side}</b>\n\n"
            f"{'─'*32}\n<b>🎯 SUMMARY:</b>\n"
            f"├ Smart Money:  <b>{af.smart_money_activity}</b>\n"
            f"├ Informed:     <b>{af.informed_trader_signal}</b>\n"
            f"└ Inst. Score:  <b>{af.institutional_flow_score:.0f}%</b>\n"
        )
        msgs.append(m2)
    m3 = f"{'─'*32}\n<b>📐 VOLUME PROFILE</b>\n{'─'*32}\n\n"
    if vp:
        in_va = vp.val <= of.price <= vp.vah
        m3 += (
            f"├ POC: <code>{fmt_price(vp.poc, instrument)}</code>\n"
            f"├ VAH: <code>{fmt_price(vp.vah, instrument)}</code>\n"
            f"├ VAL: <code>{fmt_price(vp.val, instrument)}</code>\n"
            f"└ In Value Area: {'✅' if in_va else '❌'}\n"
        )
    else:
        m3 += "Not available.\n"
    if lz:
        m3 += f"\n<b>🌊 Liquidity Zones:</b>\n"
        above = sorted([z for z in lz if z.price > of.price],
                       key=lambda x: x.price)[:3]
        below = sorted([z for z in lz if z.price <= of.price],
                       key=lambda x: x.price, reverse=True)[:3]
        for z in above:
            dist = pips_diff(instrument, z.price - of.price)
            m3 += (f"🔴 {fmt_price(z.price, instrument)} "
                   f"[{z.zone_type.replace('_',' ')}] +{dist:.0f}p\n")
        for z in below:
            dist = pips_diff(instrument, of.price - z.price)
            m3 += (f"🟢 {fmt_price(z.price, instrument)} "
                   f"[{z.zone_type.replace('_',' ')}] -{dist:.0f}p\n")
    msgs.append(m3)
    if pb:
        m4 = (
            f"{'─'*32}\n<b>👥 POSITION BOOK</b>\n{'─'*32}\n\n"
            f"├ Longs:    {pbar(pb.long_pct)} {pb.long_pct:.1f}%\n"
            f"├ Shorts:   {pbar(pb.short_pct)} {pb.short_pct:.1f}%\n"
            f"├ Skew:     <code>{fmt_pct(pb.skew)}</code>\n"
            f"├ Contrarian: <b>{pb.contrarian_signal}</b>\n"
            f"├ Trapped L:  {pb.trapped_longs_pct:.1f}%\n"
            f"├ Trapped S:  {pb.trapped_shorts_pct:.1f}%\n"
            f"├ Underwater: {pb.total_underwater:.1f}%\n"
            f"├ Pain:       <code>{fmt_price(pb.pain_threshold, instrument)}</code>\n"
            f"└ Squeeze:    {pb.squeeze_potential}\n"
        )
        msgs.append(m4)
    if ob:
        m5 = (
            f"{'─'*32}\n<b>📖 ORDER BOOK</b>\n{'─'*32}\n\n"
            f"├ Longs:    {pbar(ob.long_pct)} {ob.long_pct:.1f}%\n"
            f"├ Shorts:   {pbar(ob.short_pct)} {ob.short_pct:.1f}%\n"
            f"└ Pressure: <b>{ob.breakout_bias.replace('_',' ')}</b>\n"
        )
        msgs.append(m5)
    msgs.append(
        f"\n{'='*32}\n"
        f"<i>Use /predict for prediction\n"
        f"Use /summary for full brief\n"
        f"Use /sessions for session guide</i>\n"
    )
    return msgs


# ════════════════════════════════════════════════════════════════
#  KEYBOARDS
# ════════════════════════════════════════════════════════════════

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎯 Get Prediction",    callback_data="predict_menu")],
        [InlineKeyboardButton("📊 Analyze Pair",      callback_data="analyze_menu")],
        [InlineKeyboardButton("🧠 Market Summary",    callback_data="summary_menu")],
        [InlineKeyboardButton("🔔 Auto Alerts",       callback_data="alerts_menu")],
        [InlineKeyboardButton("💪 Currency Strength", callback_data="strength")],
        [InlineKeyboardButton("🌍 Market Overview",   callback_data="overview")],
        [InlineKeyboardButton("🕐 Session Guide",     callback_data="sessions")],
        [
            InlineKeyboardButton("🥇 Gold",   callback_data="predict_pair_XAU_USD"),
            InlineKeyboardButton("🥈 Silver", callback_data="predict_pair_XAG_USD")
        ],
        [InlineKeyboardButton("📈 NASDAQ",            callback_data="predict_pair_NAS100_USD")],
        [InlineKeyboardButton("📚 Guide",             callback_data="guide")]
    ])


def kb_pairs(mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []
    kb.append([InlineKeyboardButton("─── Major ───", callback_data="none")])
    row: List = []
    for p in ASSET_CATEGORIES["forex_major"]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3: kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("─── Cross ───", callback_data="none")])
    row = []
    for p in ASSET_CATEGORIES["forex_cross"][:12]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3: kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("More Crosses ▼",
                                    callback_data=f"{mode}_more")])
    kb.append([InlineKeyboardButton("─── Metals/Index ───", callback_data="none")])
    kb.append([
        InlineKeyboardButton("🥇 Gold",   callback_data=f"{mode}_pair_XAU_USD"),
        InlineKeyboardButton("🥈 Silver", callback_data=f"{mode}_pair_XAG_USD"),
        InlineKeyboardButton("📈 NAS",    callback_data=f"{mode}_pair_NAS100_USD")
    ])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(kb)


def kb_more_crosses(mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []; row: List = []
    for p in ASSET_CATEGORIES["forex_cross"][12:]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3: kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("◀️ Back", callback_data=f"{mode}_menu")])
    return InlineKeyboardMarkup(kb)


def kb_tf(pair: str, mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []
    for tf, info in TIMEFRAMES.items():
        kb.append([InlineKeyboardButton(
            f"{info['emoji']} {info['label']}",
            callback_data=f"{mode}_pair_{pair}_{tf}"
        )])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data=f"{mode}_menu")])
    return InlineKeyboardMarkup(kb)


def kb_alerts(chat_id: int) -> InlineKeyboardMarkup:
    is_sub = SUB_MANAGER.is_subscribed(chat_id)
    kb: List[List] = []
    if is_sub:
        kb.append([InlineKeyboardButton("✅ Alerts ON",       callback_data="none")])
        kb.append([InlineKeyboardButton("➕ Add Pair",         callback_data="al_add")])
        kb.append([InlineKeyboardButton("➖ Remove Pair",      callback_data="al_rem")])
        kb.append([InlineKeyboardButton("⚡ Change Timeframe", callback_data="al_tf")])
        kb.append([InlineKeyboardButton("📊 My Stats",        callback_data="al_stats")])
        kb.append([InlineKeyboardButton("🔕 Unsubscribe",     callback_data="al_unsub")])
    else:
        kb.append([InlineKeyboardButton("🔔 Subscribe to Alerts",
                                        callback_data="al_sub")])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(kb)


def kb_alert_pairs(action: str) -> InlineKeyboardMarkup:
    kb: List[List] = []
    kb.append([InlineKeyboardButton("─── Major ───", callback_data="none")])
    row: List = []
    for p in ASSET_CATEGORIES["forex_major"]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{action}_{p}"))
        if len(row) == 3: kb.append(row); row = []
    if row: kb.append(row)
    kb.append([
        InlineKeyboardButton("🥇 Gold", callback_data=f"{action}_XAU_USD"),
        InlineKeyboardButton("📈 NAS",  callback_data=f"{action}_NAS100_USD")
    ])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="alerts_menu")])
    return InlineKeyboardMarkup(kb)


def kb_alert_tf() -> InlineKeyboardMarkup:
    kb: List[List] = []
    for tf, info in TIMEFRAMES.items():
        kb.append([InlineKeyboardButton(
            f"{info['emoji']} {info['label']}",
            callback_data=f"al_settf_{tf}"
        )])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="alerts_menu")])
    return InlineKeyboardMarkup(kb)


# ════════════════════════════════════════════════════════════════
#  TELEGRAM COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user  = update.effective_user
    name  = esc(user.first_name) if user else "Trader"
    ml_s  = ML_ENGINE.get_stats()
    ml_st = (f"✅ Active ({ml_s['n_samples']} samples, "
             f"{ml_s['latest_acc']:.0%} acc)"
             if ml_s["is_trained"] else
             f"⏳ Learning ({ml_s['n_samples']}/{ML_MIN_SAMPLES})")
    bt_n  = len(HISTORICAL_BT.results)
    text  = (
        f"🎯 <b>Welcome, {name}!</b>\n\n"
        f"<i>FOREX QUANT v7.5 — Ultimate Trading Intelligence</i>\n\n"
        f"{'─'*30}\n\n"
        f"<b>🧠 Institutional Metrics:</b>\n"
        f"• Academic VPIN (Easley et al. 2012)\n"
        f"• Kyle's Lambda + Flow Toxicity\n"
        f"• Iceberg Detection + Absorption\n"
        f"• Liquidity Zones (FVG, OB, EQH/EQL)\n\n"
        f"<b>🤖 ML Engine:</b> {ml_st}\n"
        f"<b>📈 Backtest:</b> {bt_n} pairs stored\n\n"
        f"<b>🔔 Auto Alerts:</b>\n"
        f"• Institutional setups only ({MIN_ALERT_CONFIDENCE}%+ confidence)\n"
        f"• Pattern memory context included\n"
        f"• Opt-in with /alerts\n\n"
        f"<b>Assets:</b> 28 Forex + Gold + Silver + NASDAQ\n\n"
        f"<i>Select an option below:</i>"
    )
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=kb_main())


async def cmd_predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎯 <b>SELECT PAIR</b>", parse_mode="HTML",
        reply_markup=kb_pairs("predict"))


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 <b>SELECT PAIR</b>", parse_mode="HTML",
        reply_markup=kb_pairs("analyze"))


async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 <b>SELECT PAIR</b>", parse_mode="HTML",
        reply_markup=kb_pairs("summary"))


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "⏳ Generating report...", parse_mode="HTML")
    report = HISTORICAL_BT.format_backtest_report(QB.history)
    ch     = HISTORICAL_BT.format_backtest_chart(QB.history)
    await del_msg(msg)
    if ch:
        await send_photo(ctx.bot, update.effective_chat.id, ch,
                         caption="📊 <b>Performance Analytics</b>",
                         parse_mode="HTML")
    await update.message.reply_text(
        report, parse_mode="HTML", reply_markup=kb_main())


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    is_sub  = SUB_MANAGER.is_subscribed(chat_id)
    if is_sub:
        sub   = SUB_MANAGER.get_subscription(chat_id) or {}
        pairs = sub.get("pairs", [])
        tf    = sub.get("timeframe", "H1")
        cnt   = sub.get("alert_count", 0)
        text  = (
            f"🔔 <b>AUTO ALERTS — ACTIVE</b>\n\n"
            f"Watching: {', '.join(pair_display(p) for p in pairs) or 'None'}\n"
            f"Timeframe: {tf}\n"
            f"Alerts received: {cnt}\n\n"
            f"<b>Alert Criteria:</b>\n"
            f"• Confidence ≥ {MIN_ALERT_CONFIDENCE}%\n"
            f"• Inst. Score ≥ {MIN_ALERT_INST_SCORE}%\n"
            f"• Factors ≥ {MIN_ALERT_FACTORS}\n"
            f"• Cooldown: 4 hours per pair"
        )
    else:
        text = (
            f"🔔 <b>AUTO ALERTS</b>\n\n"
            f"Get notified when institutional setups appear.\n\n"
            f"<b>Alert fires when ALL of these are true:</b>\n"
            f"• Confidence ≥ {MIN_ALERT_CONFIDENCE}%\n"
            f"• Institutional score ≥ {MIN_ALERT_INST_SCORE}%\n"
            f"• {MIN_ALERT_FACTORS}+ confluence factors\n"
            f"• Smart money activity detected\n\n"
            f"These are NOT casual signals. Pattern memory context included.\n\n"
            f"<i>Tap Subscribe to get started.</i>"
        )
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=kb_alerts(chat_id))


async def cmd_sessions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        SA.schedule(), parse_mode="HTML", reply_markup=kb_main())


async def cmd_overview(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Loading...", parse_mode="HTML")
    try:
        async with aiohttp.ClientSession() as session:
            st  = await STRENGTH_CACHE.get(session)
            ch  = chart_strength(st)
            sc  = sorted(st.values(), key=lambda x: x.strength, reverse=True)
            sn, hrs = current_session()
            cap = (f"🌍 <b>MARKET OVERVIEW</b>\n\n🕐 {sn} ({hrs:.1f}h)\n\n"
                   f"🟢 Strongest: {sc[0].currency} ({sc[0].strength:+.2f}%)\n"
                   f"🔴 Weakest:   {sc[-1].currency} ({sc[-1].strength:+.2f}%)\n\n"
                   f"💡 Best pair: {sc[0].currency}/{sc[-1].currency}")
            await del_msg(msg)
            await send_photo(ctx.bot, update.effective_chat.id, ch,
                             caption=cap, parse_mode="HTML",
                             reply_markup=kb_main())
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:100]}")


async def cmd_strength(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Calculating...", parse_mode="HTML")
    try:
        async with aiohttp.ClientSession() as session:
            st  = await STRENGTH_CACHE.get(session)
            ch  = chart_strength(st)
            sc  = sorted(st.values(), key=lambda x: x.strength, reverse=True)
            cap = (f"💪 <b>Currency Strength (24H)</b>\n\n"
                   f"🟢 {sc[0].currency} ({sc[0].strength:+.2f}%)\n"
                   f"🔴 {sc[-1].currency} ({sc[-1].strength:+.2f}%)")
            await del_msg(msg)
            await send_photo(ctx.bot, update.effective_chat.id, ch,
                             caption=cap, parse_mode="HTML",
                             reply_markup=kb_main())
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:100]}")


async def cmd_guide(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ml_s = ML_ENGINE.get_stats()
    text = (
        f"📚 <b>FOREX QUANT v7.5 GUIDE</b>\n\n"
        f"<b>🧠 Smart Money Metrics:</b>\n"
        f"• VPIN — Probability of Informed Trading\n"
        f"• Kyle's Lambda — Price impact per volume\n"
        f"• Toxicity — Adverse selection risk\n"
        f"• Roll Spread — True implied spread\n"
        f"• Iceberg Detection — Hidden institutional orders\n"
        f"• Absorption — Buy vs sell pressure\n\n"
        f"<b>🌊 Liquidity Zones:</b>\n"
        f"• Demand/Supply zones (order blocks)\n"
        f"• Equal highs/lows (stop-hunt levels)\n"
        f"• Fair Value Gaps (FVG)\n\n"
        f"<b>🤖 ML Engine:</b>\n"
        f"• Status: {'✅ Trained' if ml_s['is_trained'] else '⏳ Learning'}\n"
        f"• Samples: {ml_s['n_samples']}\n"
        f"• Accuracy: {ml_s['latest_acc']:.0%}\n"
        f"• RF + Gradient Boosting + Logistic Regression\n"
        f"• Global model — all users benefit equally\n\n"
        f"<b>📊 Setup Quality:</b>\n"
        f"🔥 A+ = 8+ factors — 1:2.5 R/R\n"
        f"✅ A  = 6-7 factors — 1:2.0 R/R\n"
        f"⚡ B  = 4-5 factors — 1:1.5 R/R\n"
        f"⚠️ C  = 3   factors — 1:1.0 R/R\n\n"
        f"<b>🔔 Auto Alerts:</b>\n"
        f"• Fires at {MIN_ALERT_CONFIDENCE}%+ confidence only\n"
        f"• Includes pattern memory context\n"
        f"• Session-aware timing\n"
        f"• 4-hour cooldown per pair\n\n"
        f"<b>⚠️ Risk Reminder:</b>\n"
        f"No system wins 100% of the time. Use a stop-loss "
        f"on every single trade. Risk max 1-2% per trade."
    )
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=kb_main())


# ════════════════════════════════════════════════════════════════
#  CALLBACK HANDLER
# ════════════════════════════════════════════════════════════════

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await cb_answer(query)
    data    = query.data
    chat_id = query.message.chat_id

    if data == "none":
        return

    try:
        # ── Navigation ─────────────────────────────────────────
        if data == "main_menu":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "🎯 <b>FOREX QUANT v7.5</b>\n\n<i>Select option:</i>",
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data in ("predict_menu", "analyze_menu", "summary_menu"):
            mode = data.replace("_menu", "")
            ic   = ("🎯" if mode == "predict" else
                    "📊" if mode == "analyze" else "🧠")
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           f"{ic} <b>SELECT PAIR</b>",
                           parse_mode="HTML", reply_markup=kb_pairs(mode))
            return

        if data in ("predict_more", "analyze_more", "summary_more"):
            mode = data.replace("_more", "")
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id, "<b>More Crosses:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_more_crosses(mode))
            return

        if data == "sessions":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id, SA.schedule(),
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data == "overview":
            await del_msg(query.message)
            loading = await send_msg(ctx.bot, chat_id, "⏳ Loading...",
                                     parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as session:
                    st  = await STRENGTH_CACHE.get(session)
                    ch  = chart_strength(st)
                    sc  = sorted(st.values(),
                                 key=lambda x: x.strength, reverse=True)
                    sn, hrs = current_session()
                    cap = (f"🌍 <b>OVERVIEW</b>\n\n🕐 {sn}\n\n"
                           f"🟢 {sc[0].currency} | 🔴 {sc[-1].currency}")
                    if loading: await del_msg(loading)
                    await send_photo(ctx.bot, chat_id, ch,
                                     caption=cap, parse_mode="HTML",
                                     reply_markup=kb_main())
            except Exception as e:
                if loading:
                    await loading.edit_text(f"❌ {str(e)[:50]}")
            return

        if data == "strength":
            await del_msg(query.message)
            loading = await send_msg(ctx.bot, chat_id, "⏳ Calculating...",
                                     parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as session:
                    st  = await STRENGTH_CACHE.get(session)
                    ch  = chart_strength(st)
                    sc  = sorted(st.values(),
                                 key=lambda x: x.strength, reverse=True)
                    cap = (f"💪 <b>Currency Strength</b>\n\n"
                           f"🟢 {sc[0].currency} | 🔴 {sc[-1].currency}")
                    if loading: await del_msg(loading)
                    await send_photo(ctx.bot, chat_id, ch,
                                     caption=cap, parse_mode="HTML",
                                     reply_markup=kb_main())
            except Exception as e:
                if loading:
                    await loading.edit_text(f"❌ {str(e)[:50]}")
            return

        if data == "guide":
            await del_msg(query.message)
            ml_s = ML_ENGINE.get_stats()
            await send_msg(
                ctx.bot, chat_id,
                f"📚 <b>QUANT v7.5</b>\n\n"
                f"🧠 VPIN · Kyle's λ · Toxicity · Liquidity Zones\n"
                f"🤖 ML: {'Active' if ml_s['is_trained'] else 'Learning'} "
                f"({ml_s['n_samples']} samples)\n"
                f"🔔 Auto Alerts: {MIN_ALERT_CONFIDENCE}%+ confidence only\n"
                f"Use /guide for full details.",
                parse_mode="HTML", reply_markup=kb_main())
            return

        # ── Alerts ─────────────────────────────────────────────
        if data == "alerts_menu":
            await del_msg(query.message)
            is_sub = SUB_MANAGER.is_subscribed(chat_id)
            sub    = SUB_MANAGER.get_subscription(chat_id) or {}
            if is_sub:
                pairs = sub.get("pairs", [])
                tf    = sub.get("timeframe", "H1")
                cnt   = sub.get("alert_count", 0)
                text  = (f"🔔 <b>ALERTS ACTIVE</b>\n\n"
                         f"Watching: "
                         f"{', '.join(pair_display(p) for p in pairs) or 'None'}\n"
                         f"Timeframe: {tf}\nAlerts sent: {cnt}")
            else:
                text = ("🔔 <b>AUTO ALERTS</b>\n\n"
                        "<i>Subscribe to get institutional alerts.</i>")
            await send_msg(ctx.bot, chat_id, text,
                           parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data == "al_sub":
            SUB_MANAGER.subscribe(chat_id, [], "H1")
            await send_msg(ctx.bot, chat_id,
                           "✅ <b>Subscribed!</b>\n\nNow add pairs to watch.",
                           parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data == "al_unsub":
            SUB_MANAGER.unsubscribe(chat_id)
            await send_msg(ctx.bot, chat_id,
                           "🔕 <b>Unsubscribed.</b> No more auto alerts.",
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data == "al_add":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "➕ <b>SELECT PAIR TO ADD:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_alert_pairs("aladd"))
            return

        if data == "al_rem":
            sub = SUB_MANAGER.get_subscription(chat_id)
            if sub and sub.get("pairs"):
                await del_msg(query.message)
                await send_msg(ctx.bot, chat_id,
                               "➖ <b>SELECT PAIR TO REMOVE:</b>",
                               parse_mode="HTML",
                               reply_markup=kb_alert_pairs("alrem"))
            else:
                await send_msg(ctx.bot, chat_id,
                               "No pairs to remove.",
                               reply_markup=kb_alerts(chat_id))
            return

        if data == "al_tf":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "⚡ <b>SELECT TIMEFRAME FOR ALERTS:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_alert_tf())
            return

        if data == "al_stats":
            sub = SUB_MANAGER.get_subscription(chat_id) or {}
            await send_msg(
                ctx.bot, chat_id,
                f"📊 <b>Your Alert Stats</b>\n\n"
                f"Total alerts: {sub.get('alert_count', 0)}\n"
                f"Last alert:   {sub.get('last_alert', 'Never')}\n\n"
                f"<i>Alerts fire only on genuine institutional setups.</i>",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("aladd_"):
            pair = data[6:]
            SUB_MANAGER.add_pair(chat_id, pair)
            sub   = SUB_MANAGER.get_subscription(chat_id) or {}
            pairs = sub.get("pairs", [])
            await send_msg(
                ctx.bot, chat_id,
                f"✅ Added {pair_display(pair)}\n\n"
                f"Watching: {', '.join(pair_display(p) for p in pairs)}",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("alrem_"):
            pair = data[6:]
            SUB_MANAGER.remove_pair(chat_id, pair)
            sub   = SUB_MANAGER.get_subscription(chat_id) or {}
            pairs = sub.get("pairs", [])
            await send_msg(
                ctx.bot, chat_id,
                f"✅ Removed {pair_display(pair)}\n\n"
                f"Watching: "
                f"{', '.join(pair_display(p) for p in pairs) or 'None'}",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("al_settf_"):
            tf = data[9:]
            SUB_MANAGER.set_tf(chat_id, tf)
            await send_msg(ctx.bot, chat_id,
                           f"✅ Alert timeframe set to <b>{tf}</b>",
                           parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        # ── Pair → timeframe selection ─────────────────────────
        for mode in ("predict", "analyze", "summary"):
            prefix = f"{mode}_pair_"
            if data.startswith(prefix):
                pair, tf = parse_pair_tf(data, prefix)
                if pair and tf:
                    break
                # No timeframe yet — show TF selector
                candidate = data[len(prefix):]
                if candidate in ALL_PAIRS:
                    await del_msg(query.message)
                    await send_msg(
                        ctx.bot, chat_id,
                        f"{asset_emoji(candidate)} "
                        f"<b>{pair_display(candidate)}</b>"
                        f"\n\n<i>Select timeframe:</i>",
                        parse_mode="HTML",
                        reply_markup=kb_tf(candidate, mode))
                    return

        # ── PREDICT ────────────────────────────────────────────
        if data.startswith("predict_pair_"):
            pair, tf = parse_pair_tf(data, "predict_pair_")
            if not pair or not tf:
                await send_msg(ctx.bot, chat_id, "❌ Invalid selection")
                return
            await del_msg(query.message)
            loading = await send_msg(
                ctx.bot, chat_id,
                f"⏳ <b>Analyzing {pair_display(pair)}...</b>\n\n"
                f"<i>Running institutional + ML analysis...</i>",
                parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as session:
                    dd = await full_analysis(session, pair, tf)
                of  = dd["of"]; af = dd["af"]; lz = dd["lz"]
                pred = QB.generate(dd["candles"], of, dd["vp"], dd["ob"],
                                   dd["pb"], dd["cs"], af, pair, lz)
                if loading: await del_msg(loading)
                sn_now = session_name(datetime.now(timezone.utc).hour)
                sim    = PATTERN_MEMORY.find_similar(
                    pair, pred.get("direction", "LONG"),
                    af, of, sn_now)
                if pred.get("has_setup"):
                    pid  = (f"{pair}_{tf}_"
                            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
                    aligned = (pred["bull_count"]
                               if pred["direction"] == "LONG"
                               else pred["bear_count"])
                    pobj = QuantPrediction(
                        prediction_id=pid, pair=pair, timeframe=tf,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        current_price=of.price,
                        direction=pred["direction"],
                        target_price=pred["target"],
                        invalidation_price=pred["stop"],
                        confidence=pred["confidence"],
                        quality=pred["quality"],
                        reasons=pred["reasons"],
                        key_levels=pred["key_levels"],
                        factors_aligned=aligned,
                        features=pred.get("features", []),
                        status="ACTIVE", chat_id=chat_id,
                        ml_confidence=pred.get("ml_prob", 0.0),
                        ml_used=ML_ENGINE.is_trained
                    )
                    QB.add(pobj)
                    PT.add(pobj)
                    PATTERN_MEMORY.record_pattern(pobj, of, af, sn_now)
                    ch  = chart_prediction(dd["candles"], pair, pred,
                                           dd["vp"], of, af, lz)
                    msg = msg_prediction(pair, tf, pred, of, dd["ob"],
                                         dd["pb"], af, lz, sim)
                    sm  = (af.smart_money_activity if af else "N/A")
                    await send_photo(
                        ctx.bot, chat_id, ch,
                        caption=(f"{asset_emoji(pair)} "
                                 f"<b>{pair_display(pair)}</b> | "
                                 f"{'🟢' if pred['direction']=='LONG' else '🔴'} "
                                 f"{pred['direction']} | 🧠 {sm}"),
                        parse_mode="HTML")
                    await send_msg(ctx.bot, chat_id, msg,
                                   parse_mode="HTML", reply_markup=kb_main())
                    await send_msg(ctx.bot, chat_id,
                                   SA.advice(pair), parse_mode="HTML")
                else:
                    ch  = chart_no_setup(dd["candles"], pair)
                    msg = msg_no_setup(pair, tf, of,
                                       pred.get("bull_count", 0),
                                       pred.get("bear_count", 0), af)
                    await send_photo(
                        ctx.bot, chat_id, ch,
                        caption=(f"{asset_emoji(pair)} "
                                 f"<b>{pair_display(pair)}</b>"
                                 f" | ⚪ NO SETUP"),
                        parse_mode="HTML")
                    await send_msg(ctx.bot, chat_id, msg,
                                   parse_mode="HTML", reply_markup=kb_main())
            except Exception as e:
                log.error(f"Predict {pair}: {e}", exc_info=True)
                if loading: await del_msg(loading)
                await send_msg(ctx.bot, chat_id,
                               f"❌ Error: {str(e)[:100]}",
                               parse_mode="HTML", reply_markup=kb_main())
            return

        # ── ANALYZE ────────────────────────────────────────────
        if data.startswith("analyze_pair_"):
            pair, tf = parse_pair_tf(data, "analyze_pair_")
            if not pair or not tf:
                await send_msg(ctx.bot, chat_id, "❌ Invalid selection")
                return
            await del_msg(query.message)
            loading = await send_msg(
                ctx.bot, chat_id,
                f"⏳ <b>Analyzing {pair_display(pair)}...</b>",
                parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as session:
                    dd = await full_analysis(session, pair, tf)
                ms = msgs_analysis(pair, tf, dd["of"], dd["vp"],
                                   dd["ob"], dd["pb"], dd["af"], dd["lz"])
                if loading: await del_msg(loading)
                for m in ms:
                    await send_msg(ctx.bot, chat_id, m, parse_mode="HTML")
                    await asyncio.sleep(0.3)
                await send_msg(ctx.bot, chat_id,
                               SA.advice(pair), parse_mode="HTML")
                await send_msg(ctx.bot, chat_id,
                               "✅ <b>Analysis complete!</b>",
                               parse_mode="HTML", reply_markup=kb_main())
            except Exception as e:
                log.error(f"Analyze {pair}: {e}", exc_info=True)
                if loading: await del_msg(loading)
                await send_msg(ctx.bot, chat_id,
                               f"❌ Error: {str(e)[:100]}",
                               parse_mode="HTML", reply_markup=kb_main())
            return

        # ── SUMMARY ────────────────────────────────────────────
        if data.startswith("summary_pair_"):
            pair, tf = parse_pair_tf(data, "summary_pair_")
            if not pair or not tf:
                await send_msg(ctx.bot, chat_id, "❌ Invalid selection")
                return
            await del_msg(query.message)
            loading = await send_msg(
                ctx.bot, chat_id,
                f"⏳ <b>Generating Market Intelligence Brief...</b>",
                parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as session:
                    dd = await full_analysis(session, pair, tf)
                pred = QB.generate(dd["candles"], dd["of"], dd["vp"],
                                   dd["ob"], dd["pb"], dd["cs"],
                                   dd["af"], pair, dd["lz"])
                summ = MS.summarize(
                    pair, dd["of"], dd["af"], dd["vp"],
                    dd["ob"], dd["pb"], dd["lz"],
                    pred if pred.get("has_setup") else None)
                if loading: await del_msg(loading)
                sections = summ.split("\n\n")
                chunk    = ""
                for sec in sections:
                    if len(chunk) + len(sec) + 2 > 3800:
                        if chunk.strip():
                            await send_msg(ctx.bot, chat_id,
                                           chunk.strip(), parse_mode="HTML")
                            await asyncio.sleep(0.4)
                        chunk = sec + "\n\n"
                    else:
                        chunk += sec + "\n\n"
                if chunk.strip():
                    await send_msg(ctx.bot, chat_id,
                                   chunk.strip(), parse_mode="HTML")
                await send_msg(ctx.bot, chat_id,
                               SA.advice(pair), parse_mode="HTML")
                await send_msg(ctx.bot, chat_id,
                               "✅ <b>Brief complete!</b>",
                               parse_mode="HTML", reply_markup=kb_main())
            except Exception as e:
                log.error(f"Summary {pair}: {e}", exc_info=True)
                if loading: await del_msg(loading)
                await send_msg(ctx.bot, chat_id,
                               f"❌ Error: {str(e)[:100]}",
                               parse_mode="HTML", reply_markup=kb_main())
            return

    except Exception as e:
        log.error(f"Callback error: {e}", exc_info=True)
        await send_msg(ctx.bot, chat_id,
                       "❌ An error occurred. Please try again.",
                       reply_markup=kb_main())


async def on_error(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error(f"Error: {ctx.error}", exc_info=True)
    try:
        if update and update.effective_chat:
            await send_msg(ctx.bot, update.effective_chat.id,
                           "❌ An error occurred. Please try again.",
                           reply_markup=kb_main())
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  BACKGROUND TASKS
# ════════════════════════════════════════════════════════════════

async def bg_tasks(app):
    asyncio.create_task(monitor_loop(app))
    asyncio.create_task(alert_loop(app))
    asyncio.create_task(ml_retrain_loop())
    asyncio.create_task(initial_backtest())
    log.info("All background tasks started")


async def ml_retrain_loop():
    while True:
        try:
            await asyncio.sleep(6 * 3600)
            done = [p for p in QB.history
                    if p.get("outcome") in ("WIN", "LOSS")]
            if len(done) >= ML_MIN_SAMPLES:
                result = ML_ENGINE.train(QB.history)
                log.info(f"ML retrain: {result}")
        except Exception as e:
            log.error(f"ML retrain loop: {e}")


async def initial_backtest():
    await asyncio.sleep(30)
    try:
        if HISTORICAL_BT.stale():
            log.info("Running initial backtest...")
            await HISTORICAL_BT.run_full_backtest(
                pairs=ASSET_CATEGORIES["forex_major"][:5],
                timeframe="H1")
    except Exception as e:
        log.error(f"Initial backtest: {e}")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    Thread(target=start_health_server, daemon=True).start()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    log.info("=" * 65)
    log.info("  FOREX QUANT v7.5 — ULTIMATE TRADING INTELLIGENCE SYSTEM")
    log.info("=" * 65)
    log.info(f"  VPIN · Kyle's Lambda · Toxicity · Roll Spread")
    log.info(f"  Liquidity Zones: FVG · OB · Equal H/L · Demand/Supply")
    log.info(f"  ML Ensemble: RF + Gradient Boosting + Logistic Regression")
    log.info(f"  Historical Backtest: Autonomous 30-day engine")
    log.info(f"  Pattern Memory: Learns from every resolved trade")
    log.info(f"  Auto Alerts: {MIN_ALERT_CONFIDENCE}%+ conf | "
             f"{MIN_ALERT_INST_SCORE}%+ inst | {MIN_ALERT_FACTORS}+ factors")
    log.info(f"  Session Advisor: Best times per pair")
    log.info(f"  Assets: {len(ALL_PAIRS)} instruments")
    ml_s = ML_ENGINE.get_stats()
    log.info(f"  ML: {'Trained' if ml_s['is_trained'] else 'Learning'} "
             f"({ml_s['n_samples']} samples)")
    log.info("=" * 65)

    app = (Application.builder()
           .token(TELEGRAM_TOKEN)
           .read_timeout(30)
           .write_timeout(30)
           .build())

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("predict",  cmd_predict))
    app.add_handler(CommandHandler("analyze",  cmd_analyze))
    app.add_handler(CommandHandler("summary",  cmd_summary))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("alerts",   cmd_alerts))
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(CommandHandler("overview", cmd_overview))
    app.add_handler(CommandHandler("strength", cmd_strength))
    app.add_handler(CommandHandler("guide",    cmd_guide))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_error_handler(on_error)

    app.post_init = bg_tasks

    log.info("🚀 FOREX QUANT v7.5 is live!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
