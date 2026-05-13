
# ════════════════════════════════════════════════════════════════

import asyncio
import aiohttp
import logging
import sys
import platform
import io
import os
import json
import gc
import math
import uuid
import numpy as np
from scipy import stats
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import warnings
warnings.filterwarnings('ignore')

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Application, CommandHandler,
                           CallbackQueryHandler, ContextTypes)
from telegram.error import TimedOut, NetworkError

# ════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("ForexQuant")

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

OANDA_API_KEY    = os.environ.get(
    "OANDA_API_KEY",
    "67eae51bea4d1ddbc5899613f6660977-ac693290039d6b19fe0634712ee248a4")
OANDA_ACCOUNT_ID = os.environ.get(
    "OANDA_ACCOUNT_ID", "101-001-27452070-001")
TELEGRAM_TOKEN   = os.environ.get(
    "TELEGRAM_TOKEN",
    "7123896226:AAFLeyCPnfJjJgakBH8twdSDPyznu69ZQa4")
ENVIRONMENT      = os.environ.get("OANDA_ENV", "practice")

# ── File paths ────────────────────────────────────────────────────
PREDICTIONS_FILE        = "predictions_v81.json"
ACTIVE_PREDICTIONS_FILE = "active_predictions_v81.json"
ML_MODEL_FILE           = "ml_model_v81.pkl"
HISTORICAL_BT_FILE      = "historical_backtest_v81.json"
PATTERN_MEMORY_FILE     = "pattern_memory_v81.json"
USER_SUBSCRIPTIONS_FILE = "user_subscriptions_v81.json"
PERFORMANCE_FILE        = "performance_v81.json"
GLOBAL_STATS_FILE       = "global_stats_v81.json"
TRADE_LOG_FILE          = "trade_log_v81.json"
ML_TRAINING_DATA_FILE   = "ml_training_data_v81.json"

# ── API ───────────────────────────────────────────────────────────
API_BASE = ("https://api-fxpractice.oanda.com"
            if ENVIRONMENT == "practice"
            else "https://api-fxtrade.oanda.com")
HEADERS  = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Accept-Datetime-Format": "RFC3339"
}

# ── Instruments ───────────────────────────────────────────────────
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
    "forex_major": [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
        "USD_CAD", "NZD_USD", "USD_CHF"
    ],
    "forex_cross": [
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_AUD", "GBP_AUD",
        "AUD_JPY", "EUR_CAD", "GBP_CAD", "AUD_CAD", "EUR_NZD",
        "GBP_NZD", "AUD_NZD", "NZD_JPY", "CAD_JPY", "EUR_CHF",
        "GBP_CHF", "AUD_CHF", "NZD_CHF", "CAD_CHF", "CHF_JPY",
        "NZD_CAD"
    ],
    "metals":  ["XAU_USD", "XAG_USD"],
    "indices": ["NAS100_USD"]
}

CORRELATION_GROUPS = {
    "EUR_USD": ["GBP_USD", "EUR_GBP", "EUR_JPY"],
    "GBP_USD": ["EUR_USD", "GBP_JPY", "EUR_GBP"],
    "USD_JPY": ["EUR_JPY", "GBP_JPY", "AUD_JPY"],
    "AUD_USD": ["NZD_USD", "AUD_JPY", "AUD_NZD"],
    "XAU_USD": ["EUR_USD", "AUD_USD", "USD_CHF"],
    "NAS100_USD": ["USD_JPY"]
}

# ── Timeframes ────────────────────────────────────────────────────
TIMEFRAMES = {
    "M5":  {"label": "5 Minutes",  "candles": 288, "emoji": "⚡", "minutes": 5},
    "M15": {"label": "15 Minutes", "candles": 200, "emoji": "🕐", "minutes": 15},
    "H1":  {"label": "1 Hour",     "candles": 168, "emoji": "📊", "minutes": 60},
    "H4":  {"label": "4 Hours",    "candles": 120, "emoji": "📈", "minutes": 240},
    "D":   {"label": "Daily",      "candles": 60,  "emoji": "📅", "minutes": 1440},
}

# ── Signal thresholds ─────────────────────────────────────────────
MIN_CONFIDENCE          = 55.0
MIN_ALERT_CONFIDENCE    = 62.0
MIN_RR_RATIO            = 1.5     # minimum reward/risk
MIN_TARGET_PIPS_FOREX   = 12.0    # minimum target distance forex
MIN_TARGET_PIPS_GOLD    = 80.0    # minimum target pips for XAU
MIN_TARGET_PIPS_SILVER  = 30.0
MIN_TARGET_PIPS_NAS     = 15.0
MIN_STOP_PIPS_FOREX     = 8.0
MIN_STOP_PIPS_GOLD      = 50.0
MIN_STOP_PIPS_SILVER    = 20.0
MIN_STOP_PIPS_NAS       = 10.0
ML_MIN_SAMPLES          = 50      # reduced from 300 so ML activates sooner
                                   # but still meaningful
STRENGTH_CACHE_TTL      = 300

# ── Risk & performance ────────────────────────────────────────────
DEFAULT_RISK_PCT          = 1.0
REDUCED_RISK_PCT          = 0.5
MAX_CONSECUTIVE_LOSSES    = 5
MONITORING_INTERVAL       = 25    # seconds between prediction checks
BREAKEVEN_TRIGGER_PCT     = 50.0  # trigger breakeven when 50% of target reached
TRAILING_TRIGGER_PCT      = 75.0  # suggest trailing stop at 75%

# ── Scanning ──────────────────────────────────────────────────────
SCAN_INTERVAL_KILL_ZONE = 90      # seconds during kill zones
SCAN_INTERVAL_NORMAL    = 240     # normal session
SCAN_INTERVAL_OFF_HOURS = 600     # off hours (still scans!)

# ════════════════════════════════════════════════════════════════
#  ENUMS
# ════════════════════════════════════════════════════════════════

class SetupQuality(Enum):
    A_PLUS   = "A+ (Institutional)"
    A        = "A (Strong)"
    B        = "B (Moderate)"
    C        = "C (Weak)"
    NO_TRADE = "No Clear Setup"

class SessionType(Enum):
    SYDNEY    = "Sydney"
    TOKYO     = "Tokyo"
    LONDON    = "London"
    NEW_YORK  = "New York"
    OVERLAP   = "London-NY Overlap"
    OFF_HOURS = "Off Hours"

class RegimeType(Enum):
    STRONG_TREND  = "Strong Trend"
    WEAK_TREND    = "Weak Trend"
    RANGING       = "Ranging"
    COMPRESSION   = "Compression"
    EXPANSION     = "Expansion"
    CHAOTIC       = "Chaotic"

class StrategyType(Enum):
    ICT_CONTINUATION  = "ICT Trend Continuation"
    ICT_REVERSAL      = "ICT Liquidity Reversal"
    AMD_SESSION       = "AMD Session Model"
    RETAIL_CONTRARIAN = "Retail Contrarian Squeeze"
    BREAKOUT          = "Breakout Compression"
    MEAN_REVERSION    = "Statistical Mean Reversion"
    CORRELATION_DIV   = "Correlation Divergence"

class StructureType(Enum):
    BULLISH_BOS   = "Bullish BOS"
    BEARISH_BOS   = "Bearish BOS"
    BULLISH_CHOCH = "Bullish CHOCH"
    BEARISH_CHOCH = "Bearish CHOCH"
    NONE          = "None"

class PredictionStatus(Enum):
    ACTIVE     = "ACTIVE"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT   = "STOP_HIT"
    BREAKEVEN  = "BREAKEVEN"
    EXPIRED    = "EXPIRED"
    CANCELLED  = "CANCELLED"

# ════════════════════════════════════════════════════════════════
#  TRADE ID GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_trade_id(pair: str, direction: str) -> str:
    """
    Generate unique human-readable trade ID.
    Format: FQ-EURUSD-BUY-20240115-A3K9
    """
    now    = datetime.now(timezone.utc)
    date_s = now.strftime("%Y%m%d")
    time_s = now.strftime("%H%M")
    pair_s = pair.replace("_", "")[:6]
    dir_s  = "BUY" if direction == "LONG" else "SEL"
    unique = uuid.uuid4().hex[:4].upper()
    return f"FQ-{pair_s}-{dir_s}-{date_s}{time_s}-{unique}"

# ════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    time:     str
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   float
    complete: bool = True

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_abs(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def close_location_value(self) -> float:
        if self.range == 0:
            return 0.5
        return (self.close - self.low) / self.range

    @property
    def buy_volume(self) -> float:
        return self.volume * self.close_location_value

    @property
    def sell_volume(self) -> float:
        return self.volume - self.buy_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def body_ratio(self) -> float:
        if self.range == 0:
            return 0.0
        return self.body_abs / self.range

    @property
    def is_displacement(self) -> bool:
        return self.body_ratio >= 0.7

    def is_displacement_vs(self, avg_body: float) -> bool:
        return (self.body_abs >= avg_body * 1.5 and
                self.body_ratio >= 0.65)


@dataclass
class SwingPoint:
    price:    float
    index:    int
    time:     str
    is_high:  bool
    strength: int = 1


@dataclass
class MarketStructure:
    swing_highs:     List[SwingPoint] = field(default_factory=list)
    swing_lows:      List[SwingPoint] = field(default_factory=list)
    trend:           str = "NEUTRAL"
    last_bos:        Optional[StructureType] = None
    last_choch:      Optional[StructureType] = None
    internal_trend:  str = "NEUTRAL"
    htf_bias:        str = "NEUTRAL"


@dataclass
class LiquidityLevel:
    price:      float
    level_type: str
    strength:   int
    swept:      bool  = False
    sweep_time: Optional[str] = None
    zone_upper: float = 0.0
    zone_lower: float = 0.0


@dataclass
class FairValueGap:
    upper:     float
    lower:     float
    mid:       float
    direction: str
    time:      str
    filled:    bool  = False
    inverted:  bool  = False
    strength:  float = 0.0


@dataclass
class OrderBlock:
    high:            float
    low:             float
    mid:             float
    direction:       str
    time:            str
    valid:           bool  = True
    swept_before:    bool  = False
    left_fvg:        bool  = False
    broke_structure: bool  = False
    strength:        float = 0.0


@dataclass
class SignalValidation:
    """Result of pre-flight signal geometry validation."""
    is_valid:      bool
    reason:        str
    rr_ratio:      float
    reward_pips:   float
    risk_pips:     float
    entry:         float
    target:        float
    stop:          float


@dataclass
class OrderFlowData:
    buy_volume:        float = 0.0
    sell_volume:       float = 0.0
    total_volume:      float = 0.0
    cvd:               float = 0.0
    delta:             float = 0.0
    buy_pct:           float = 50.0
    sell_pct:          float = 50.0
    imbalance:         float = 0.0
    imbalance_zscore:  float = 0.0
    price:             float = 0.0
    vwap:              float = 0.0
    vwap_zscore:       float = 0.0
    delta_momentum:    float = 0.0
    volume_trend:      float = 0.0
    recent_delta_5:    float = 0.0
    recent_delta_10:   float = 0.0
    recent_delta_20:   float = 0.0
    cvd_slope:         float = 0.0
    volume_accel:      float = 0.0
    buying_climax:     bool  = False
    selling_climax:    bool  = False
    vol_ratio:         float = 1.0
    efficiency_ratio:  float = 0.0


@dataclass
class RegimeData:
    regime:               RegimeType = RegimeType.RANGING
    hurst_exponent:       float = 0.5
    fractal_dimension:    float = 1.5
    entropy:              float = 1.0
    autocorr_lag1:        float = 0.0
    variance_ratio:       float = 1.0
    efficiency_ratio:     float = 0.0
    volatility_ratio:     float = 1.0
    adr_consumed_pct:     float = 0.0
    adr_today:            float = 0.0
    adr_average:          float = 0.0
    trend_slope:          float = 0.0
    r_squared:            float = 0.0
    recommended_strategy: Optional[StrategyType] = None
    contradictory:        bool  = False   # Hurst vs ER conflict flag


@dataclass
class SessionData:
    session_type:    SessionType
    session_name:    str
    is_kill_zone:    bool
    kill_zone_name:  str
    hours_remaining: float
    asian_high:      Optional[float] = None
    asian_low:       Optional[float] = None
    asian_range:     float = 0.0
    amd_phase:       str = "UNKNOWN"
    is_dst_london:   bool = False
    is_dst_ny:       bool = False
    wat_time:        str = ""


@dataclass
class PivotLevels:
    daily_pp:     float = 0.0
    daily_r1:     float = 0.0
    daily_r2:     float = 0.0
    daily_r3:     float = 0.0
    daily_s1:     float = 0.0
    daily_s2:     float = 0.0
    daily_s3:     float = 0.0
    weekly_pp:    float = 0.0
    weekly_r1:    float = 0.0
    weekly_s1:    float = 0.0
    monthly_pp:   float = 0.0
    camarilla_r4: float = 0.0
    camarilla_s4: float = 0.0


@dataclass
class VolumeProfile:
    poc:          float = 0.0
    vah:          float = 0.0
    val:          float = 0.0
    hvn:          List[float] = field(default_factory=list)
    lvn:          List[float] = field(default_factory=list)
    total_volume: float = 0.0


@dataclass
class OrderBookData:
    price:               float = 0.0
    total_longs:         float = 0.0
    total_shorts:        float = 0.0
    net_imbalance:       float = 0.0
    pending_delta:       float = 0.0
    breakout_bias:       str   = "BALANCED"
    long_pct:            float = 50.0
    short_pct:           float = 50.0
    pressure_dir:        str   = "NEUTRAL"
    stop_cluster_above:  float = 0.0
    stop_cluster_below:  float = 0.0


@dataclass
class PositionBookData:
    price:               float = 0.0
    long_pct:            float = 50.0
    short_pct:           float = 50.0
    skew:                float = 0.0
    skew_change:         float = 0.0
    contrarian_signal:   str   = "NEUTRAL"
    trapped_longs_pct:   float = 0.0
    trapped_shorts_pct:  float = 0.0
    total_underwater:    float = 0.0
    crowded_trade_index: float = 0.0
    squeeze_potential:   str   = "LOW"
    pain_threshold:      float = 0.0


@dataclass
class AdvancedFlowData:
    vpin:                   float = 0.0
    vpin_level:             str   = "LOW"
    price_impact_ratio:     float = 0.0
    toxicity:               float = 0.0
    toxicity_level:         str   = "LOW"
    amihud_illiquidity:     float = 0.0
    liquidity_level:        str   = "NORMAL"
    market_depth_imbalance: float = 0.0
    depth_bias:             str   = "NEUTRAL"
    iceberg_count:          int   = 0
    smart_money_activity:   str   = "LOW"
    informed_signal:        str   = "NEUTRAL"
    absorption_ratio:       float = 1.0
    aggressor_side:         str   = "NEUTRAL"
    institutional_score:    float = 0.0
    roll_spread:            float = 0.0


@dataclass
class CurrencyStrength:
    currency:         str   = ""
    strength:         float = 0.0
    strength_zscore:  float = 0.0
    trend:            str   = "NEUTRAL"
    rank:             int   = 0
    momentum:         float = 0.0


@dataclass
class ConfluenceFactor:
    name:        str
    direction:   str
    strength:    float
    description: str
    category:    str   = "OTHER"
    weight:      float = 1.0


@dataclass
class StrategySignal:
    strategy:         StrategyType
    direction:        str
    confidence:       float
    entry:            float
    target:           float
    stop:             float
    quality:          SetupQuality
    reasons:          List[str]
    factors:          List[ConfluenceFactor]
    features:         List[float]
    rr_ratio:         float = 0.0
    reward_pips:      float = 0.0
    risk_pips:        float = 0.0
    regime_aligned:   bool  = True
    session_aligned:  bool  = True
    htf_aligned:      bool  = True
    ml_probability:   float = 0.5
    calibrated_prob:  float = 0.5
    adr_ok:           bool  = True
    breakeven_price:  float = 0.0   # price at which to move SL to BE
    trailing_price:   float = 0.0   # price at which to trail stop


@dataclass
class QuantPrediction:
    trade_id:           str          # FQ-EURUSD-BUY-20240115-A3K9
    prediction_id:      str          # internal UUID
    pair:               str
    timeframe:          str
    timestamp:          str
    current_price:      float
    direction:          str
    target_price:       float
    invalidation_price: float
    breakeven_price:    float
    trailing_price:     float
    confidence:         float
    calibrated_prob:    float
    quality:            str
    strategy:           str
    reasons:            List[str]
    key_levels:         List[Dict]
    factors_aligned:    int
    features:           List[float]
    rr_ratio:           float
    reward_pips:        float
    risk_pips:          float
    status:             str   = "ACTIVE"
    outcome:            Optional[str] = None
    hit_time:           Optional[str] = None
    chat_ids:           List[int] = field(default_factory=list)  # ALL recipients
    ml_confidence:      float = 0.0
    ml_used:            bool  = False
    pips_gained:        float = 0.0
    mae_pips:           float = 0.0
    mfe_pips:           float = 0.0
    regime_at_signal:   str   = ""
    session_at_signal:  str   = ""
    breakeven_notified: bool  = False
    trailing_notified:  bool  = False
    was_sent_to_users:  bool  = False   # KEY: only notify if this is True
    sent_quality:       str   = ""      # quality when signal was sent


# ════════════════════════════════════════════════════════════════
#  SIGNAL GEOMETRY VALIDATOR  (CRITICAL FIX)
# ════════════════════════════════════════════════════════════════

def get_min_pips(instrument: str) -> Tuple[float, float]:
    """Returns (min_target_pips, min_stop_pips) for instrument."""
    if "XAU" in instrument:
        return MIN_TARGET_PIPS_GOLD,   MIN_STOP_PIPS_GOLD
    if "XAG" in instrument:
        return MIN_TARGET_PIPS_SILVER, MIN_STOP_PIPS_SILVER
    if "NAS" in instrument:
        return MIN_TARGET_PIPS_NAS,    MIN_STOP_PIPS_NAS
    return MIN_TARGET_PIPS_FOREX,      MIN_STOP_PIPS_FOREX


def validate_signal_geometry(direction:  str,
                              entry:      float,
                              target:     float,
                              stop:       float,
                              instrument: str,
                              min_rr:     float = MIN_RR_RATIO
                              ) -> SignalValidation:
    """
    CRITICAL: Validates all signal geometry before any signal is sent.
    Catches inverted R/R, targets too close, stops wrong side, etc.
    """
    pip            = pip_value(instrument)
    min_tgt, min_stp = get_min_pips(instrument)

    # ── Direction geometry ────────────────────────────────────────
    if direction == "LONG":
        if target <= entry:
            return SignalValidation(
                False,
                f"LONG target ({fmt_price(target, instrument)}) must be "
                f"ABOVE entry ({fmt_price(entry, instrument)})",
                0.0, 0.0, 0.0, entry, target, stop)
        if stop >= entry:
            return SignalValidation(
                False,
                f"LONG stop ({fmt_price(stop, instrument)}) must be "
                f"BELOW entry ({fmt_price(entry, instrument)})",
                0.0, 0.0, 0.0, entry, target, stop)
        reward = target - entry
        risk   = entry  - stop

    elif direction == "SHORT":
        if target >= entry:
            return SignalValidation(
                False,
                f"SHORT target ({fmt_price(target, instrument)}) must be "
                f"BELOW entry ({fmt_price(entry, instrument)})",
                0.0, 0.0, 0.0, entry, target, stop)
        if stop <= entry:
            return SignalValidation(
                False,
                f"SHORT stop ({fmt_price(stop, instrument)}) must be "
                f"ABOVE entry ({fmt_price(entry, instrument)})",
                0.0, 0.0, 0.0, entry, target, stop)
        reward = entry  - target
        risk   = stop   - entry
    else:
        return SignalValidation(
            False, f"Invalid direction: {direction}",
            0.0, 0.0, 0.0, entry, target, stop)

    # ── Pip distances ─────────────────────────────────────────────
    reward_pips = reward / pip
    risk_pips   = risk   / pip

    if reward_pips < min_tgt:
        return SignalValidation(
            False,
            f"Target too close: {reward_pips:.1f} pips "
            f"(minimum {min_tgt:.0f} pips for {instrument})",
            0.0, reward_pips, risk_pips, entry, target, stop)

    if risk_pips < min_stp:
        return SignalValidation(
            False,
            f"Stop too tight: {risk_pips:.1f} pips "
            f"(minimum {min_stp:.0f} pips for {instrument})",
            0.0, reward_pips, risk_pips, entry, target, stop)

    # ── R/R ──────────────────────────────────────────────────────
    if risk == 0:
        return SignalValidation(
            False, "Zero risk distance",
            0.0, reward_pips, risk_pips, entry, target, stop)

    rr = reward / risk   # CORRECT: reward divided by risk

    if rr < min_rr:
        return SignalValidation(
            False,
            f"R/R too low: 1:{rr:.2f} (minimum 1:{min_rr}). "
            f"Reward {reward_pips:.1f}p vs Risk {risk_pips:.1f}p",
            rr, reward_pips, risk_pips, entry, target, stop)

    return SignalValidation(
        True, "Valid", rr, reward_pips, risk_pips, entry, target, stop)


def calculate_breakeven_price(direction: str,
                               entry:     float,
                               target:    float,
                               pct:       float = BREAKEVEN_TRIGGER_PCT
                               ) -> float:
    """Price at which to suggest moving stop to breakeven."""
    if direction == "LONG":
        return entry + (target - entry) * (pct / 100)
    else:
        return entry - (entry - target) * (pct / 100)


def calculate_trailing_price(direction: str,
                              entry:     float,
                              target:    float,
                              pct:       float = TRAILING_TRIGGER_PCT
                              ) -> float:
    """Price at which to suggest trailing the stop."""
    if direction == "LONG":
        return entry + (target - entry) * (pct / 100)
    else:
        return entry - (entry - target) * (pct / 100)


# ════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_pip_info(instrument: str) -> Tuple[float, int]:
    if "JPY" in instrument:   return 0.01,   3
    if "XAU" in instrument:   return 0.10,   2
    if "XAG" in instrument:   return 0.01,   3
    if "NAS" in instrument:   return 1.00,   1
    return 0.0001, 5

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

def format_sci(value: float, precision: int = 4) -> str:
    if value == 0: return "0.0000"
    if abs(value) < 0.0001 or abs(value) > 100000:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"

def progress_bar(pct: float, length: int = 10) -> str:
    filled = int(max(0, min(pct, 100)) / 100 * length)
    return "█" * filled + "░" * (length - filled)

def escape_html(text: str) -> str:
    return (str(text).replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))

def pair_display(instrument: str) -> str:
    return instrument.replace("_", "/")

def asset_emoji(instrument: str) -> str:
    if "XAU" in instrument: return "🥇"
    if "XAG" in instrument: return "🥈"
    if "NAS" in instrument: return "📈"
    return "💱"

def quality_emoji(quality_str: str) -> str:
    if "A+" in quality_str: return "🔥"
    if quality_str.startswith("A"):  return "✅"
    if "B" in quality_str:  return "⚡"
    return "⚠️"

def load_json(filepath: str, default: Any) -> Any:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Load {filepath}: {e}")
    return default

def save_json(filepath: str, data: Any):
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Save {filepath}: {e}")

# Short aliases
fmt_price  = format_price
fmt_num    = format_number
fmt_signed = format_signed
fmt_pct    = format_pct
fmt_sci    = format_sci
pbar       = progress_bar
esc        = escape_html
pips       = pips_diff
pip_val    = pip_value

# ════════════════════════════════════════════════════════════════
#  ASYNC SAFE HELPERS
# ════════════════════════════════════════════════════════════════

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

async def safe_send_message(bot, chat_id: int,
                             text: str, **kwargs) -> Optional[object]:
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                bot.send_message(
                    chat_id=chat_id, text=text, **kwargs),
                timeout=30.0)
        except (TimedOut, NetworkError):
            if attempt < 2:
                await asyncio.sleep(1 + attempt)
        except Exception as e:
            log.error(f"Send message: {e}")
            return None
    return None

async def safe_send_photo(bot, chat_id: int,
                           photo, **kwargs) -> Optional[object]:
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                bot.send_photo(
                    chat_id=chat_id, photo=photo, **kwargs),
                timeout=60.0)
        except (TimedOut, NetworkError):
            if attempt < 2:
                await asyncio.sleep(1 + attempt)
        except Exception as e:
            log.error(f"Send photo: {e}")
            return None
    return None

cb_answer  = safe_answer_callback
del_msg    = safe_delete_message
send_msg   = safe_send_message
send_photo = safe_send_photo

# ════════════════════════════════════════════════════════════════
#  OANDA API LAYER
# ════════════════════════════════════════════════════════════════

async def fetch_candles(session:     aiohttp.ClientSession,
                        instrument:  str,
                        granularity: str,
                        count:       int) -> List[Candle]:
    url    = f"{API_BASE}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        async with session.get(
                url, headers=HEADERS, params=params,
                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                log.warning(
                    f"Candles {instrument} {granularity}: "
                    f"HTTP {resp.status}")
                return []
            data    = await resp.json()
            candles = []
            for c in data.get("candles", []):
                if "mid" not in c:
                    continue
                m = c["mid"]
                o, h, l, cl = (float(m["o"]), float(m["h"]),
                               float(m["l"]), float(m["c"]))
                vol = max(float(c.get("volume", 1)), 1)
                if h < l or h < o or h < cl or l > o or l > cl:
                    continue
                candles.append(Candle(
                    time=c.get("time", ""),
                    open=o, high=h, low=l, close=cl,
                    volume=vol,
                    complete=c.get("complete", True)
                ))
            return candles
    except Exception as e:
        log.error(f"Fetch candles {instrument}: {e}")
        return []


async def fetch_current_price(session:    aiohttp.ClientSession,
                               instrument: str) -> Optional[float]:
    url    = f"{API_BASE}/v3/instruments/{instrument}/candles"
    params = {"granularity": "S5", "count": 1, "price": "M"}
    try:
        async with session.get(
                url, headers=HEADERS, params=params,
                timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            cc   = data.get("candles", [])
            if cc and "mid" in cc[0]:
                return float(cc[0]["mid"]["c"])
    except Exception as e:
        log.error(f"Fetch price {instrument}: {e}")
    return None


async def fetch_order_book(session:    aiohttp.ClientSession,
                            instrument: str) -> Optional[Dict]:
    url = f"{API_BASE}/v3/instruments/{instrument}/orderBook"
    try:
        async with session.get(
                url, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            return (await resp.json()).get("orderBook")
    except Exception:
        return None


async def fetch_position_book(session:    aiohttp.ClientSession,
                               instrument: str) -> Optional[Dict]:
    url = f"{API_BASE}/v3/instruments/{instrument}/positionBook"
    try:
        async with session.get(
                url, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            return (await resp.json()).get("positionBook")
    except Exception:
        return None

# ════════════════════════════════════════════════════════════════
#  SESSION ENGINE  (DST-AWARE, WAT-CORRECT)
# ════════════════════════════════════════════════════════════════

class SessionEngine:
    """
    All session times in UTC.
    Nigerian time WAT = UTC+1 (no DST).
    London: UTC+0 winter, UTC+1 summer.
    New York: UTC-5 winter, UTC-4 summer.
    """

    @staticmethod
    def _london_dst(dt: datetime) -> bool:
        year = dt.year
        mar31 = date(year, 3, 31)
        dst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
        oct31 = date(year, 10, 31)
        dst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
        return dst_start <= dt.date() < dst_end

    @staticmethod
    def _ny_dst(dt: datetime) -> bool:
        year = dt.year
        mar1 = date(year, 3, 1)
        dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
        nov1 = date(year, 11, 1)
        dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
        return dst_start <= dt.date() < dst_end

    def get_session(self,
                    dt: Optional[datetime] = None) -> SessionData:
        if dt is None:
            dt = datetime.now(timezone.utc)

        h  = dt.hour
        m  = dt.minute
        hm = h + m / 60.0

        lon_dst = self._london_dst(dt)
        ny_dst  = self._ny_dst(dt)

        lon_open  = 6  if lon_dst else 7
        lon_close = 15 if lon_dst else 16
        ny_open   = 11 if ny_dst  else 12
        ny_close  = 20 if ny_dst  else 21

        lon_kz_close = lon_open + 3
        ny_kz_close  = ny_open  + 3

        overlap_open  = ny_open
        overlap_close = lon_close

        in_london  = lon_open <= hm < lon_close
        in_ny      = ny_open  <= hm < ny_close
        in_overlap = (overlap_open <= hm < overlap_close
                      and in_london and in_ny)
        in_tokyo   = 0 <= hm < 9
        in_sydney  = (21 <= hm < 24) or (0 <= hm < 1)

        in_lon_kz = lon_open <= hm < lon_kz_close
        in_ny_kz  = ny_open  <= hm < ny_kz_close

        amd_phase = self._amd_phase(hm, lon_open, ny_open)

        if in_overlap:
            stype     = SessionType.OVERLAP
            sname     = "London-NY Overlap"
            remaining = overlap_close - hm
        elif in_london:
            stype     = SessionType.LONDON
            sname     = "London"
            remaining = lon_close - hm
        elif in_ny:
            stype     = SessionType.NEW_YORK
            sname     = "New York"
            remaining = ny_close - hm
        elif in_tokyo:
            stype     = SessionType.TOKYO
            sname     = "Tokyo / Asian"
            remaining = 9.0 - hm
        elif in_sydney:
            stype     = SessionType.SYDNEY
            sname     = "Sydney"
            remaining = (24.0 - hm) if hm >= 21 else (1.0 - hm)
        else:
            stype     = SessionType.OFF_HOURS
            sname     = "Off Hours"
            remaining = 0.0

        kz_name = ""
        is_kz   = False
        if in_lon_kz:
            is_kz   = True
            kz_name = "London Kill Zone (08:00–11:00 WAT)"
        elif in_ny_kz:
            is_kz   = True
            kz_name = "NY Kill Zone (13:00–16:00 WAT)"

        # WAT display
        wat = (dt + timedelta(hours=1)).strftime("%H:%M WAT")

        return SessionData(
            session_type    = stype,
            session_name    = sname,
            is_kill_zone    = is_kz,
            kill_zone_name  = kz_name,
            hours_remaining = max(0.0, remaining),
            amd_phase       = amd_phase,
            is_dst_london   = lon_dst,
            is_dst_ny       = ny_dst,
            wat_time        = wat,
        )

    @staticmethod
    def _amd_phase(hm: float, lon_open: float, ny_open: float) -> str:
        if 0 <= hm < lon_open:
            return "ACCUMULATION"
        if lon_open <= hm < lon_open + 1.5:
            return "MANIPULATION"
        if lon_open + 1.5 <= hm < ny_open:
            return "DISTRIBUTION"
        if ny_open <= hm < ny_open + 1.5:
            return "MANIPULATION"
        if hm >= ny_open + 1.5:
            return "DISTRIBUTION"
        return "UNKNOWN"

    def get_scan_interval(self, session: SessionData) -> int:
        if session.session_type == SessionType.OFF_HOURS:
            return SCAN_INTERVAL_OFF_HOURS
        if session.is_kill_zone:
            return SCAN_INTERVAL_KILL_ZONE
        return SCAN_INTERVAL_NORMAL

    def get_min_confidence(self, session: SessionData) -> float:
        if session.session_type == SessionType.OFF_HOURS:
            return MIN_ALERT_CONFIDENCE + 8.0
        if session.is_kill_zone:
            return MIN_ALERT_CONFIDENCE - 4.0
        return MIN_ALERT_CONFIDENCE

    def wat_time(self, dt: Optional[datetime] = None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        return (dt + timedelta(hours=1)).strftime("%H:%M WAT")

    def utc_time(self, dt: Optional[datetime] = None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%H:%M UTC")


SESSION_ENGINE = SessionEngine()

# ════════════════════════════════════════════════════════════════
#  MARKET MATHEMATICS ENGINE
# ════════════════════════════════════════════════════════════════

class MarketMathEngine:

    def hurst_exponent(self, prices: List[float],
                       min_window: int = 10) -> float:
        n = len(prices)
        if n < 20:
            return 0.5
        try:
            ts      = np.array(prices, dtype=float)
            lags    = range(min_window, n // 2)
            rs_arr  = []
            lag_arr = []
            for lag in lags:
                subseries = [ts[i:i+lag]
                             for i in range(0, n - lag, lag)]
                rs_vals = []
                for sub in subseries:
                    if len(sub) < 4:
                        continue
                    mean = np.mean(sub)
                    dev  = np.cumsum(sub - mean)
                    r    = np.max(dev) - np.min(dev)
                    s    = np.std(sub, ddof=1)
                    if s > 0:
                        rs_vals.append(r / s)
                if rs_vals:
                    rs_arr.append(np.log(np.mean(rs_vals)))
                    lag_arr.append(np.log(lag))
            if len(lag_arr) < 5:
                return 0.5
            slope, _, _, _, _ = linregress(lag_arr, rs_arr)
            return float(np.clip(slope, 0.05, 0.95))
        except Exception:
            return 0.5

    def fractal_dimension(self, prices: List[float]) -> float:
        n = len(prices)
        if n < 20:
            return 1.5
        try:
            arr  = np.array(prices, dtype=float)
            lo   = np.min(arr)
            hi   = np.max(arr)
            rng  = hi - lo
            if rng == 0:
                return 1.0
            norm   = (arr - lo) / rng
            path   = np.sum(np.abs(np.diff(norm)))
            direct = abs(norm[-1] - norm[0])
            if direct == 0:
                return 2.0
            fd = 1.0 + math.log(path) / math.log(n)
            return float(np.clip(fd, 1.0, 2.0))
        except Exception:
            return 1.5

    def price_entropy(self, returns: List[float],
                      bins: int = 10) -> float:
        if len(returns) < 10:
            return 1.0
        try:
            arr       = np.array(returns, dtype=float)
            counts, _ = np.histogram(arr, bins=bins)
            counts    = counts[counts > 0]
            probs     = counts / counts.sum()
            entropy   = -np.sum(probs * np.log2(probs))
            return float(entropy / math.log2(bins))
        except Exception:
            return 1.0

    def autocorrelation(self, returns: List[float],
                        lag: int = 1) -> float:
        if len(returns) < lag + 5:
            return 0.0
        try:
            arr  = np.array(returns, dtype=float)
            if np.std(arr) == 0:
                return 0.0
            corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def variance_ratio(self, prices: List[float],
                       q: int = 5) -> float:
        if len(prices) < q + 5:
            return 1.0
        try:
            arr  = np.array(prices, dtype=float)
            r1   = np.diff(arr)
            rq   = arr[q:] - arr[:-q]
            var1 = np.var(r1, ddof=1)
            varq = np.var(rq, ddof=1) / q
            if var1 == 0:
                return 1.0
            return float(varq / var1)
        except Exception:
            return 1.0

    def efficiency_ratio(self, prices: List[float],
                         period: int = 10) -> float:
        if len(prices) < period + 1:
            return 0.0
        try:
            arr    = np.array(prices[-period-1:], dtype=float)
            direct = abs(arr[-1] - arr[0])
            path   = np.sum(np.abs(np.diff(arr)))
            if path == 0:
                return 0.0
            return float(np.clip(direct / path, 0.0, 1.0))
        except Exception:
            return 0.0

    def std_bands(self, prices: List[float],
                  period: int = 20) -> Dict[str, float]:
        if len(prices) < period:
            return {}
        try:
            arr  = np.array(prices[-period:], dtype=float)
            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=1))
            return {
                "mean":    mean,   "std":     std,
                "upper_1": mean + std,       "lower_1": mean - std,
                "upper_2": mean + 2 * std,   "lower_2": mean - 2 * std,
                "upper_3": mean + 3 * std,   "lower_3": mean - 3 * std,
            }
        except Exception:
            return {}

    def vwap_zscore(self, price: float, vwap: float,
                    prices: List[float],
                    period: int = 20) -> float:
        if len(prices) < period or vwap == 0:
            return 0.0
        try:
            std = float(np.std(prices[-period:], ddof=1))
            if std == 0:
                return 0.0
            return float((price - vwap) / std)
        except Exception:
            return 0.0

    def imbalance_zscore(self, current: float,
                         history: List[float]) -> float:
        if len(history) < 10:
            return 0.0
        try:
            arr  = np.array(history, dtype=float)
            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=1))
            if std == 0:
                return 0.0
            return float((current - mean) / std)
        except Exception:
            return 0.0

    def average_daily_range(self,
                            daily_candles: List[Candle]) -> float:
        if len(daily_candles) < 5:
            return 0.0
        return float(np.mean(
            [c.range for c in daily_candles[-20:]]))

    def adr_consumed(self,
                     daily_candles: List[Candle],
                     current_price: float
                     ) -> Tuple[float, float]:
        adr = self.average_daily_range(daily_candles)
        if adr == 0 or not daily_candles:
            return 0.0, 0.0
        today_rng = daily_candles[-1].high - daily_candles[-1].low
        pct       = min(100.0, today_rng / adr * 100)
        return float(pct), float(adr)

    def fibonacci_levels(self, swing_high: float,
                          swing_low: float,
                          direction: str = "LONG") -> Dict[str, float]:
        rng = swing_high - swing_low
        if rng == 0:
            return {}
        if direction == "LONG":
            return {
                "0.0":    swing_high,
                "0.236":  swing_high - 0.236 * rng,
                "0.382":  swing_high - 0.382 * rng,
                "0.500":  swing_high - 0.500 * rng,
                "0.618":  swing_high - 0.618 * rng,
                "0.705":  swing_high - 0.705 * rng,
                "0.786":  swing_high - 0.786 * rng,
                "1.0":    swing_low,
                "1.272":  swing_low  - 0.272 * rng,
                "1.618":  swing_low  - 0.618 * rng,
                "-0.272": swing_high + 0.272 * rng,
                "-0.618": swing_high + 0.618 * rng,
            }
        else:
            return {
                "0.0":    swing_low,
                "0.236":  swing_low  + 0.236 * rng,
                "0.382":  swing_low  + 0.382 * rng,
                "0.500":  swing_low  + 0.500 * rng,
                "0.618":  swing_low  + 0.618 * rng,
                "0.705":  swing_low  + 0.705 * rng,
                "0.786":  swing_low  + 0.786 * rng,
                "1.0":    swing_high,
                "1.272":  swing_high + 0.272 * rng,
                "1.618":  swing_high + 0.618 * rng,
                "-0.272": swing_low  - 0.272 * rng,
                "-0.618": swing_low  - 0.618 * rng,
            }

    def is_in_ote(self, price: float,
                  swing_high: float,
                  swing_low: float,
                  direction: str) -> bool:
        fibs = self.fibonacci_levels(swing_high, swing_low, direction)
        if not fibs:
            return False
        if direction == "LONG":
            return fibs["0.786"] <= price <= fibs["0.618"]
        else:
            return fibs["0.618"] <= price <= fibs["0.786"]

    def calculate_pivots(self, prev_high: float,
                          prev_low: float,
                          prev_close: float) -> PivotLevels:
        pp  = (prev_high + prev_low + prev_close) / 3
        rng = prev_high - prev_low
        return PivotLevels(
            daily_pp = pp,
            daily_r1 = 2 * pp - prev_low,
            daily_r2 = pp + rng,
            daily_r3 = prev_high + 2 * (pp - prev_low),
            daily_s1 = 2 * pp - prev_high,
            daily_s2 = pp - rng,
            daily_s3 = prev_low - 2 * (prev_high - pp),
            camarilla_r4 = prev_close + rng * 1.1 / 2,
            camarilla_s4 = prev_close - rng * 1.1 / 2,
        )

    def linear_regression(self, prices: List[float],
                           period: int = 50) -> Dict[str, float]:
        if len(prices) < period:
            period = len(prices)
        if period < 5:
            return {}
        try:
            arr    = np.array(prices[-period:], dtype=float)
            x      = np.arange(len(arr))
            slope, intercept, r, _, _ = linregress(x, arr)
            fitted = slope * x + intercept
            resid  = arr - fitted
            std_r  = float(np.std(resid, ddof=1))
            cur_ft = float(slope * (len(arr) - 1) + intercept)
            return {
                "slope":     float(slope),
                "r_squared": float(r ** 2),
                "upper_1":   cur_ft + std_r,
                "upper_2":   cur_ft + 2 * std_r,
                "lower_1":   cur_ft - std_r,
                "lower_2":   cur_ft - 2 * std_r,
                "mid":       cur_ft,
                "std_resid": std_r,
            }
        except Exception:
            return {}

    def expected_move(self, price: float,
                      atr: float,
                      candles_remaining: int = 12) -> Dict[str, float]:
        if atr == 0:
            return {}
        sigma = atr * math.sqrt(candles_remaining / 14)
        return {
            "1sigma_up":   price + sigma,
            "1sigma_down": price - sigma,
            "2sigma_up":   price + 2 * sigma,
            "2sigma_down": price - 2 * sigma,
            "sigma":       sigma,
        }

    def kelly_fraction(self, win_rate: float,
                       avg_win: float,
                       avg_loss: float) -> float:
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        b  = avg_win / avg_loss
        q  = 1.0 - win_rate
        k  = (b * win_rate - q) / b
        return float(max(0.0, min(k * 0.5, 0.05)))

    def calculate_atr(self, candles: List[Candle],
                       period: int = 14) -> float:
        if len(candles) < period + 1:
            return candles[-1].range if candles else 0.0001
        tr = [
            max(candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i-1].close),
                abs(candles[i].low  - candles[i-1].close))
            for i in range(1, len(candles))
        ]
        return float(np.mean(tr[-period:])) if tr else 0.0001

    def volatility_ratio(self, candles: List[Candle],
                          short: int = 14,
                          long:  int = 50) -> float:
        if len(candles) < long + 1:
            return 1.0
        satr = self.calculate_atr(candles, short)
        latr = self.calculate_atr(candles[-long:], min(14, long - 1))
        if latr == 0:
            return 1.0
        return float(satr / latr)

    def statistical_levels(self,
                            candles:    List[Candle],
                            instrument: str,
                            n_levels:   int = 6) -> List[Dict]:
        if len(candles) < 20:
            return []
        pip  = pip_value(instrument)
        tol  = pip * 5
        points: List[Tuple[float, int, bool]] = []
        for i in range(2, len(candles) - 2):
            c = candles[i]
            if all(c.high >= candles[i+d].high and
                   c.high >= candles[i-d].high
                   for d in range(1, 3)):
                points.append((c.high, i, True))
            if all(c.low <= candles[i+d].low and
                   c.low <= candles[i-d].low
                   for d in range(1, 3)):
                points.append((c.low, i, False))
        if not points:
            return []
        clusters = []
        used     = set()
        for idx, (price, ci, is_high) in enumerate(points):
            if idx in used:
                continue
            group = [(price, ci, is_high)]
            for jdx, (p2, ci2, ih2) in enumerate(
                    points[idx+1:], idx+1):
                if jdx not in used and abs(p2 - price) <= tol:
                    group.append((p2, ci2, ih2))
                    used.add(jdx)
            used.add(idx)
            touches   = len(group)
            avg_price = sum(g[0] for g in group) / touches
            recency   = sum(g[1] for g in group) / (
                len(candles) * touches)
            strength  = touches * (0.5 + recency)
            clusters.append({
                "price":   avg_price,
                "type":    "RESISTANCE" if is_high else "SUPPORT",
                "touches": touches,
                "strength": strength,
            })
        clusters.sort(key=lambda x: x["strength"], reverse=True)
        return clusters[:n_levels]

    def calculate_vwap(self, candles: List[Candle]) -> float:
        vp = sum((c.high + c.low + c.close) / 3 * c.volume
                 for c in candles)
        tv = sum(c.volume for c in candles)
        return vp / tv if tv > 0 else (
            candles[-1].close if candles else 0.0)

    def check_regime_contradiction(self,
                                   hurst:    float,
                                   er:       float,
                                   autocorr: float) -> bool:
        """
        Returns True if regime signals contradict each other.
        Example: Hurst says trending but ER says choppy.
        """
        hurst_trending = hurst > 0.58
        er_trending    = er    > 0.40
        ac_trending    = autocorr > 0.10
        trending_votes = sum([hurst_trending, er_trending, ac_trending])
        return trending_votes == 1   # exactly one says trending = contradiction


MATH = MarketMathEngine()

# ════════════════════════════════════════════════════════════════
#  REGIME DETECTION ENGINE
# ════════════════════════════════════════════════════════════════

class RegimeEngine:

    def detect(self,
               candles:       List[Candle],
               daily_candles: Optional[List[Candle]] = None,
               instrument:    str = "") -> RegimeData:
        if len(candles) < 30:
            return RegimeData()

        closes  = [c.close for c in candles]
        returns = [closes[i] / closes[i-1] - 1.0
                   for i in range(1, len(closes))]

        hurst   = MATH.hurst_exponent(closes[-100:])
        frac    = MATH.fractal_dimension(closes[-50:])
        entropy = MATH.price_entropy(returns[-50:])
        ac1     = MATH.autocorrelation(returns[-30:], lag=1)
        vr      = MATH.variance_ratio(closes[-50:], q=5)
        er      = MATH.efficiency_ratio(closes, period=20)
        vol_r   = MATH.volatility_ratio(candles, 14, 50)
        lr      = MATH.linear_regression(closes, period=50)
        slope   = lr.get("slope", 0.0)
        r2      = lr.get("r_squared", 0.0)

        adr_pct = 0.0
        adr_val = 0.0
        if daily_candles and len(daily_candles) >= 5:
            adr_pct, adr_val = MATH.adr_consumed(
                daily_candles, candles[-1].close)

        contradictory = MATH.check_regime_contradiction(hurst, er, ac1)
        regime        = self._classify(
            hurst, frac, entropy, ac1, vr, er, vol_r, r2)
        rec_strat     = self._recommend(regime, vol_r, er)

        return RegimeData(
            regime               = regime,
            hurst_exponent       = hurst,
            fractal_dimension    = frac,
            entropy              = entropy,
            autocorr_lag1        = ac1,
            variance_ratio       = vr,
            efficiency_ratio     = er,
            volatility_ratio     = vol_r,
            adr_consumed_pct     = adr_pct,
            adr_today            = adr_val,
            adr_average          = adr_val,
            trend_slope          = slope,
            r_squared            = r2,
            recommended_strategy = rec_strat,
            contradictory        = contradictory,
        )

    def _classify(self, hurst, frac, entropy, ac1,
                  vr, er, vol_r, r2) -> RegimeType:
        if vol_r < 0.5:
            return RegimeType.COMPRESSION
        if vol_r > 2.5:
            return RegimeType.EXPANSION
        if entropy > 0.9 and vol_r > 1.5:
            return RegimeType.CHAOTIC

        tv = bv = 0
        if hurst > 0.58:   tv += 2
        elif hurst < 0.42: bv += 2
        elif hurst > 0.52: tv += 1
        elif hurst < 0.48: bv += 1

        if frac < 1.35:    tv += 1
        elif frac > 1.65:  bv += 1

        if ac1 > 0.15:     tv += 1
        elif ac1 < -0.15:  bv += 1

        if vr > 1.3:       tv += 1
        elif vr < 0.7:     bv += 1

        if er > 0.45:      tv += 1
        elif er < 0.25:    bv += 1

        if r2 > 0.65:      tv += 1
        elif r2 < 0.25:    bv += 1

        if entropy < 0.5:  tv += 1
        elif entropy > 0.8: bv += 1

        if tv >= bv * 1.5:
            return (RegimeType.STRONG_TREND
                    if tv >= 5 else RegimeType.WEAK_TREND)
        if bv >= tv * 1.5:
            return RegimeType.RANGING
        return RegimeType.RANGING

    def _recommend(self, regime: RegimeType,
                   vol_r: float,
                   er: float) -> Optional[StrategyType]:
        if regime in (RegimeType.STRONG_TREND, RegimeType.WEAK_TREND):
            return StrategyType.ICT_CONTINUATION
        if regime == RegimeType.RANGING:
            return StrategyType.ICT_REVERSAL
        if regime == RegimeType.COMPRESSION:
            return StrategyType.BREAKOUT
        return None

    def is_strategy_valid(self, strategy: StrategyType,
                           regime: RegimeData,
                           session: SessionData) -> bool:
        if regime.regime == RegimeType.CHAOTIC:
            return False
        if strategy == StrategyType.ICT_CONTINUATION:
            return regime.regime in (
                RegimeType.STRONG_TREND, RegimeType.WEAK_TREND,
                RegimeType.EXPANSION)
        if strategy == StrategyType.ICT_REVERSAL:
            return regime.regime in (
                RegimeType.RANGING, RegimeType.COMPRESSION,
                RegimeType.WEAK_TREND)
        if strategy == StrategyType.AMD_SESSION:
            return (session.is_kill_zone and
                    session.amd_phase in ("MANIPULATION", "DISTRIBUTION"))
        if strategy == StrategyType.RETAIL_CONTRARIAN:
            return True
        if strategy == StrategyType.BREAKOUT:
            return regime.regime in (
                RegimeType.COMPRESSION, RegimeType.RANGING)
        if strategy == StrategyType.MEAN_REVERSION:
            return regime.regime == RegimeType.RANGING
        if strategy == StrategyType.CORRELATION_DIV:
            return regime.regime != RegimeType.CHAOTIC
        return True


REGIME_ENGINE = RegimeEngine()

# ════════════════════════════════════════════════════════════════
#  ORDER FLOW ENGINE
# ════════════════════════════════════════════════════════════════

class OrderFlowEngine:

    _imbalance_history: Dict[str, deque] = {}

    def analyze(self, candles: List[Candle],
                instrument: str = "") -> OrderFlowData:
        if not candles or len(candles) < 20:
            return OrderFlowData()

        bv    = sum(c.buy_volume  for c in candles)
        sv    = sum(c.sell_volume for c in candles)
        tv    = bv + sv
        cvd   = sum(c.delta for c in candles)
        vwap  = MATH.calculate_vwap(candles)
        price = candles[-1].close

        rd5   = sum(c.delta for c in candles[-5:])
        rd10  = sum(c.delta for c in candles[-10:])
        rd20  = sum(c.delta for c in candles[-20:])
        d5f   = (sum(c.delta for c in candles[-10:-5])
                 if len(candles) >= 10 else 0.0)
        mom   = rd5 - d5f

        cv = []
        r  = 0.0
        for c in candles[-20:]:
            r += c.delta
            cv.append(r)
        cvd_slope = ((sum(cv[10:]) / max(1, len(cv)-10)) -
                     sum(cv[:10]) / 10) if len(cv) >= 10 else 0.0

        if len(candles) >= 20:
            v1 = sum(c.volume for c in candles[-20:-10]) / 10
            v2 = sum(c.volume for c in candles[-10:])    / 10
            vt = (v2 - v1) / v1 * 100 if v1 > 0 else 0.0
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
        bc    = rec_v > avg_v * 2.0 and rd5 > 0 and mom < 0
        sc    = rec_v > avg_v * 2.0 and rd5 < 0 and mom > 0

        imb   = (bv - sv) / tv * 100 if tv > 0 else 0.0

        key = instrument
        if key not in self._imbalance_history:
            self._imbalance_history[key] = deque(maxlen=200)
        self._imbalance_history[key].append(imb)
        imb_z = MATH.imbalance_zscore(
            imb, list(self._imbalance_history[key]))

        closes = [c.close for c in candles]
        vwap_z = MATH.vwap_zscore(price, vwap, closes)
        er     = MATH.efficiency_ratio(closes, period=20)
        vol_r  = MATH.volatility_ratio(candles)

        return OrderFlowData(
            buy_volume       = bv,
            sell_volume      = sv,
            total_volume     = tv,
            cvd              = cvd,
            delta            = rd20,
            buy_pct          = (bv/tv*100) if tv > 0 else 50.0,
            sell_pct         = (sv/tv*100) if tv > 0 else 50.0,
            imbalance        = imb,
            imbalance_zscore = imb_z,
            price            = price,
            vwap             = vwap,
            vwap_zscore      = vwap_z,
            delta_momentum   = mom,
            volume_trend     = vt,
            recent_delta_5   = rd5,
            recent_delta_10  = rd10,
            recent_delta_20  = rd20,
            cvd_slope        = cvd_slope,
            volume_accel     = va,
            buying_climax    = bc,
            selling_climax   = sc,
            vol_ratio        = vol_r,
            efficiency_ratio = er,
        )


OF_ENGINE = OrderFlowEngine()

# ════════════════════════════════════════════════════════════════
#  ADVANCED FLOW ENGINE
# ════════════════════════════════════════════════════════════════

class AdvancedFlowEngine:

    def analyze(self, candles: List[Candle],
                order_book: Optional[Dict] = None) -> AdvancedFlowData:
        if not candles or len(candles) < 20:
            return AdvancedFlowData()

        vpin     = self._vpin(candles)
        pir      = self._price_impact_ratio(candles)
        toxicity = self._toxicity(candles)
        amihud   = self._amihud(candles)
        roll     = self._roll_spread(candles)
        abs_r    = self._absorption_ratio(candles)
        aggr     = self._aggressor_side(candles)

        depth_imb   = 0.0
        iceberg_cnt = 0
        if order_book:
            depth_imb   = self._depth_imbalance(order_book)
            iceberg_cnt = len(self._detect_icebergs(order_book))

        vpin_lv  = self._lvl_vpin(vpin)
        tox_lv   = self._lvl_toxicity(toxicity)
        liq_lv   = self._lvl_liquidity(amihud)
        depth_b  = self._bias_depth(depth_imb)
        sm_act   = self._smart_money_score(vpin, toxicity, pir, depth_imb)
        informed = self._informed_signal(candles, vpin, toxicity, aggr)
        inst_sc  = self._institutional_score(
            vpin, pir, toxicity, abs_r, iceberg_cnt)

        return AdvancedFlowData(
            vpin                   = vpin,
            vpin_level             = vpin_lv,
            price_impact_ratio     = pir,
            toxicity               = toxicity,
            toxicity_level         = tox_lv,
            amihud_illiquidity     = amihud,
            liquidity_level        = liq_lv,
            market_depth_imbalance = depth_imb,
            depth_bias             = depth_b,
            iceberg_count          = iceberg_cnt,
            smart_money_activity   = sm_act,
            informed_signal        = informed,
            absorption_ratio       = abs_r,
            aggressor_side         = aggr,
            institutional_score    = inst_sc,
            roll_spread            = roll,
        )

    def _vpin(self, candles: List[Candle],
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
        return float(min(1.0,
            sum(abs(b - s) for b, s in zip(bb, sb)) / (len(bb) * bs)))

    def _price_impact_ratio(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        try:
            pc  = [c.close - c.open for c in candles]
            sv  = [c.delta for c in candles]
            vv  = float(np.var(sv))
            if vv == 0: return 0.0
            cov = np.cov(pc, sv)
            return float(cov[0][1] / vv) if cov.shape == (2,2) else 0.0
        except Exception: return 0.0

    def _toxicity(self, candles: List[Candle]) -> float:
        if len(candles) < 10: return 0.0
        scores = []
        for i in range(1, len(candles)):
            p = candles[i-1]; c = candles[i]
            exp = (p.delta / p.volume * p.range) if p.volume > 0 else 0
            if p.range > 0:
                scores.append(
                    min(1.0, abs((c.close - p.close) - exp) / p.range))
        return float(np.mean(scores)) if scores else 0.0

    def _amihud(self, candles: List[Candle]) -> float:
        if len(candles) < 5: return 0.0
        illiq = []
        for i in range(1, len(candles)):
            pc = candles[i-1].close; cc = candles[i].close
            vol = candles[i].volume
            if pc > 0 and vol > 0:
                illiq.append(abs((cc - pc) / pc) / vol)
        return float(np.mean(illiq) * 1e9) if illiq else 0.0

    def _roll_spread(self, candles: List[Candle]) -> float:
        if len(candles) < 20: return 0.0
        ch  = [candles[i].close - candles[i-1].close
               for i in range(1, len(candles))]
        if len(ch) < 2: return 0.0
        m   = float(np.mean(ch))
        cov = float(np.mean([(ch[i]-m)*(ch[i-1]-m)
                              for i in range(1, len(ch))]))
        return float(2 * math.sqrt(-cov)) if cov < 0 else 0.0

    def _absorption_ratio(self, candles: List[Candle]) -> float:
        if not candles: return 1.0
        ba = sa = 0.0
        for c in candles:
            d   = c.range + 0.0001
            ba += c.lower_wick * c.volume / d
            sa += c.upper_wick * c.volume / d
        if sa == 0: return 2.0 if ba > 0 else 1.0
        return float(ba / sa)

    def _aggressor_side(self, candles: List[Candle]) -> str:
        if len(candles) < 5: return "NEUTRAL"
        bv = bev = 0.0
        for c in candles[-10:]:
            if c.range > 0 and c.body_ratio > 0.6:
                if c.is_bullish: bv  += c.volume
                else:            bev += c.volume
        t = bv + bev
        if t == 0: return "NEUTRAL"
        bp = bv / t
        if bp > 0.60: return "BUYERS"
        if bp < 0.40: return "SELLERS"
        return "NEUTRAL"

    def _detect_icebergs(self, ob: Optional[Dict]) -> Dict:
        if not ob or "buckets" not in ob: return {}
        fm = {}
        for b in ob["buckets"]:
            p = float(b["price"])
            t = (float(b.get("longCountPercent", 0)) +
                 float(b.get("shortCountPercent", 0)))
            fm[p] = fm.get(p, 0) + t
        if not fm: return {}
        vals = list(fm.values())
        avg  = float(np.mean(vals))
        std  = float(np.std(vals)) if len(vals) > 1 else 0.0
        return {p: f for p, f in fm.items() if f > avg + 2 * std}

    def _depth_imbalance(self, ob: Optional[Dict]) -> float:
        if not ob or "buckets" not in ob: return 0.0
        bp = float(ob.get("price", 0))
        bid = ask = 0.0
        for b in ob["buckets"]:
            p = float(b["price"])
            t = (float(b.get("longCountPercent", 0)) +
                 float(b.get("shortCountPercent", 0)))
            if p < bp: bid += t
            else:      ask += t
        td = bid + ask
        return float(max(-1.0, min(1.0, (bid-ask)/td))) if td > 0 else 0.0

    def _lvl_vpin(self, v: float) -> str:
        if v >= 0.70: return "VERY HIGH 🔥"
        if v >= 0.50: return "HIGH ⚠️"
        if v >= 0.30: return "MODERATE"
        if v >= 0.15: return "LOW"
        return "MINIMAL"

    def _lvl_toxicity(self, t: float) -> str:
        if t >= 0.70: return "VERY HIGH 🔥"
        if t >= 0.50: return "HIGH ⚠️"
        if t >= 0.30: return "MODERATE"
        return "LOW"

    def _lvl_liquidity(self, a: float) -> str:
        if a >= 100: return "VERY ILLIQUID ⚠️"
        if a >= 50:  return "ILLIQUID"
        if a >= 20:  return "MODERATE"
        return "LIQUID ✅"

    def _bias_depth(self, d: float) -> str:
        if d >= 0.30:  return "STRONG BID 🟢"
        if d >= 0.10:  return "BID PRESSURE"
        if d <= -0.30: return "STRONG ASK 🔴"
        if d <= -0.10: return "ASK PRESSURE"
        return "BALANCED"

    def _smart_money_score(self, vpin, toxicity, pir, depth) -> str:
        s = 0
        if vpin >= 0.50:       s += 30
        elif vpin >= 0.30:     s += 15
        if toxicity >= 0.50:   s += 25
        elif toxicity >= 0.30: s += 12
        if abs(pir) >= 0.00005:  s += 25
        elif abs(pir) >= 0.00001: s += 12
        if abs(depth) >= 0.30:   s += 20
        elif abs(depth) >= 0.15: s += 10
        if s >= 70: return "VERY HIGH 🔥"
        if s >= 50: return "HIGH ⚠️"
        if s >= 30: return "MODERATE"
        if s >= 15: return "LOW"
        return "MINIMAL"

    def _informed_signal(self, candles, vpin, toxicity, aggressor) -> str:
        if vpin < 0.30 and toxicity < 0.30: return "NEUTRAL"
        cvd = sum(c.delta for c in candles[-10:])
        if aggressor == "BUYERS"  and cvd > 0: return "BULLISH 🟢"
        if aggressor == "SELLERS" and cvd < 0: return "BEARISH 🔴"
        if cvd > 0: return "LEANING BULLISH"
        if cvd < 0: return "LEANING BEARISH"
        return "NEUTRAL"

    def _institutional_score(self, vpin, pir, toxicity,
                              absorption, icebergs) -> float:
        s  = min(25.0, vpin * 35)
        s += min(20.0, abs(pir) * 100000 * 20)
        s += min(20.0, toxicity * 30)
        s += min(10.0, icebergs * 5)
        if absorption > 1.5 or absorption < 0.67:   s += 20
        elif absorption > 1.2 or absorption < 0.83: s += 10
        return min(100.0, s)


ADVANCED_FLOW = AdvancedFlowEngine()

# ════════════════════════════════════════════════════════════════
#  ORDER BOOK / POSITION BOOK ANALYZERS
# ════════════════════════════════════════════════════════════════

def analyze_order_book(ob: Optional[Dict],
                        price: float) -> Optional[OrderBookData]:
    if not ob or "buckets" not in ob:
        return None
    bp   = float(ob.get("price", price))
    al   = bl = as_ = bs_ = 0.0
    long_clusters  = []
    short_clusters = []
    for b in ob["buckets"]:
        p  = float(b["price"])
        lp = float(b["longCountPercent"])
        sp = float(b["shortCountPercent"])
        if p > bp:
            al += lp; as_ += sp
            if lp > 2.0: long_clusters.append(p)
        else:
            bl += lp; bs_ += sp
            if sp > 2.0: short_clusters.append(p)
    tl   = al + bl; ts = as_ + bs_
    ni   = tl - ts
    pd   = (al + as_) - (bl + bs_)
    bias = ("UPWARD_PRESSURE"   if pd > 2 else
            "DOWNWARD_PRESSURE" if pd < -2 else "BALANCED")
    pdir = ("BULLISH" if pd > 2 else "BEARISH" if pd < -2 else "NEUTRAL")
    tot  = tl + ts
    return OrderBookData(
        price             = bp,
        total_longs       = tl,  total_shorts      = ts,
        net_imbalance     = ni,  pending_delta     = pd,
        breakout_bias     = bias, long_pct         = (tl/tot*100) if tot>0 else 50.0,
        short_pct         = (ts/tot*100) if tot>0 else 50.0,
        pressure_dir      = pdir,
        stop_cluster_above = max(long_clusters)  if long_clusters  else 0.0,
        stop_cluster_below = min(short_clusters) if short_clusters else 0.0,
    )


_pb_history: Dict[str, deque] = {}

def analyze_position_book(pb: Optional[Dict],
                           price: float,
                           instrument: str = "") -> Optional[PositionBookData]:
    if not pb or "buckets" not in pb:
        return None
    bp = float(pb.get("price", price))
    tl = ts = trl = trs = 0.0
    for b in pb["buckets"]:
        p  = float(b["price"])
        lp = float(b["longCountPercent"])
        sp = float(b["shortCountPercent"])
        tl += lp; ts += sp
        if p > bp and ((p-bp)/bp*100) > 0.5:  trl += lp
        elif p < bp and p > 0 and ((bp-p)/p*100) > 0.5: trs += sp
    tot  = tl + ts
    lp_  = (tl/tot*100) if tot > 0 else 50.0
    sp_  = (ts/tot*100) if tot > 0 else 50.0
    skew = lp_ - sp_
    key  = instrument
    if key not in _pb_history:
        _pb_history[key] = deque(maxlen=20)
    hist        = list(_pb_history[key])
    skew_change = (skew - hist[-1]) if hist else 0.0
    _pb_history[key].append(skew)
    tlp  = (trl/tot*100) if tot > 0 else 0.0
    tsp  = (trs/tot*100) if tot > 0 else 0.0
    tuw  = tlp + tsp
    ci   = abs(skew) * (tuw / 10)
    pain = bp * (1 + skew / 200)
    sqz  = ("HIGH" if ci > 300 else "MODERATE" if ci > 150 else "LOW")
    ct   = ("BEARISH" if skew > 15 else "BULLISH" if skew < -15 else "NEUTRAL")
    return PositionBookData(
        price=bp, long_pct=lp_, short_pct=sp_, skew=skew,
        skew_change=skew_change, contrarian_signal=ct,
        trapped_longs_pct=tlp, trapped_shorts_pct=tsp,
        total_underwater=tuw, crowded_trade_index=ci,
        squeeze_potential=sqz, pain_threshold=pain,
    )

# ════════════════════════════════════════════════════════════════
#  VOLUME PROFILE
# ════════════════════════════════════════════════════════════════

def calculate_volume_profile(candles: List[Candle],
                              num_levels: int = 24
                              ) -> Optional[VolumeProfile]:
    if not candles:
        return None
    hi  = max(c.high for c in candles)
    lo  = min(c.low  for c in candles)
    rng = hi - lo
    if rng <= 0:
        return None
    step   = rng / num_levels
    levels = [{"price": lo + (i+0.5)*step, "volume": 0.0}
              for i in range(num_levels)]
    for c in candles:
        li  = max(0, min(int((c.low  - lo)/step), num_levels-1))
        hi2 = max(0, min(int((c.high - lo)/step), num_levels-1))
        n   = hi2 - li + 1
        vpl = c.volume / n if n > 0 else c.volume
        for idx in range(li, hi2+1):
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
    hvn = sorted([lv["price"] for lv in levels if lv["volume"] > avg*1.5])
    lvn = sorted([lv["price"] for lv in levels if lv["volume"] < avg*0.3])
    return VolumeProfile(poc=poc["price"], vah=vah, val=val,
                         hvn=hvn[:5], lvn=lvn[:5], total_volume=tv)

# ════════════════════════════════════════════════════════════════
#  CURRENCY STRENGTH ENGINE
# ════════════════════════════════════════════════════════════════

class CurrencyStrengthEngine:
    _history: Dict[str, deque] = {c: deque(maxlen=50) for c in CURRENCIES}

    async def calculate(self,
                        session:   aiohttp.ClientSession,
                        timeframe: str = "H1",
                        periods:   int = 24
                        ) -> Dict[str, CurrencyStrength]:
        changes: Dict[str, List[float]] = {c: [] for c in CURRENCIES}
        for pair in FOREX_PAIRS[:14]:
            try:
                candles = await fetch_candles(
                    session, pair, timeframe, periods+1)
                if len(candles) >= 2:
                    parts = pair.split("_")
                    if len(parts) == 2:
                        base, quote = parts
                        op, cl = candles[0].open, candles[-1].close
                        if op > 0:
                            pct = (cl - op) / op * 100
                            if base  in changes: changes[base].append(pct)
                            if quote in changes: changes[quote].append(-pct)
                await asyncio.sleep(0.02)
            except Exception:
                continue

        raw_vals  = {}
        strengths = {}
        for currency in CURRENCIES:
            avg = (sum(changes[currency]) / len(changes[currency])
                   if changes[currency] else 0.0)
            raw_vals[currency] = avg
            self._history[currency].append(avg)

        for currency in CURRENCIES:
            hist = list(self._history[currency])
            if len(hist) >= 5:
                mean = float(np.mean(hist))
                std  = float(np.std(hist)) if np.std(hist) > 0 else 1.0
                z    = (raw_vals[currency] - mean) / std
            else:
                z = 0.0
            hist = list(self._history[currency])
            mom  = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
            trend = ("STRONG"  if raw_vals[currency] > 0.1 else
                     "WEAK"    if raw_vals[currency] < -0.1 else "NEUTRAL")
            strengths[currency] = CurrencyStrength(
                currency        = currency,
                strength        = raw_vals[currency],
                strength_zscore = z,
                trend           = trend,
                rank            = 0,
                momentum        = mom,
            )
        for i, cs in enumerate(sorted(
                strengths.values(),
                key=lambda x: x.strength, reverse=True)):
            strengths[cs.currency].rank = i + 1
        return strengths


class CurrencyStrengthCache:
    def __init__(self, ttl: int = STRENGTH_CACHE_TTL):
        self._engine      = CurrencyStrengthEngine()
        self.cache: Optional[Dict] = None
        self.last_update: Optional[datetime] = None
        self.ttl          = ttl
        self._lock        = asyncio.Lock()

    async def get(self, session: aiohttp.ClientSession) -> Dict:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if (self.cache is None or self.last_update is None or
                    (now - self.last_update).total_seconds() > self.ttl):
                self.cache       = await self._engine.calculate(session)
                self.last_update = now
            return self.cache


STRENGTH_CACHE = CurrencyStrengthCache()

# ════════════════════════════════════════════════════════════════
#  ML TRAINING DATA PERSISTENCE
#  Ensures ML data survives restarts — never resets
# ════════════════════════════════════════════════════════════════

class MLDataStore:
    """
    Persistent store for ML training samples.
    Survives restarts. Accumulates forever.
    """

    def __init__(self):
        self.data: List[Dict] = load_json(ML_TRAINING_DATA_FILE, [])
        log.info(f"ML data store: {len(self.data)} samples loaded")

    def add(self, sample: Dict):
        """Add a training sample."""
        self.data.append(sample)
        # Save every 10 samples to avoid excessive I/O
        if len(self.data) % 10 == 0:
            self._save()

    def update_outcome(self, prediction_id: str,
                       outcome: str, pips: float):
        """Update outcome when trade resolves."""
        for s in self.data:
            if s.get("prediction_id") == prediction_id:
                s["outcome"]    = outcome
                s["pips_gained"] = pips
                break
        self._save()

    def get_training_data(self) -> List[Dict]:
        """Return all samples with resolved outcomes."""
        return [s for s in self.data
                if s.get("outcome") in ("WIN", "LOSS")
                and len(s.get("features", [])) == 20]

    def _save(self):
        save_json(ML_TRAINING_DATA_FILE, self.data[-2000:])

    @property
    def n_samples(self) -> int:
        return len(self.get_training_data())

    @property
    def n_total(self) -> int:
        return len(self.data)


ML_DATA_STORE = MLDataStore()

# ════════════════════════════════════════════════════════════════
#  HEALTH SERVER
# ════════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ForexQuant v8.1 Beast Mode - ONLINE")
    def log_message(self, format, *args):
        pass

def start_health_server():
    port   = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()

# ════════════════════════════════════════════════════════════════
#  FULL ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════

async def full_analysis(http_session: aiohttp.ClientSession,
                        pair:         str,
                        tf:           str) -> Dict:
    tf_cfg = TIMEFRAMES.get(tf, TIMEFRAMES["H1"])

    results = await asyncio.gather(
        fetch_candles(http_session, pair, tf,  tf_cfg["candles"]),
        fetch_candles(http_session, pair, "D",  60),
        fetch_candles(http_session, pair, "H4", 120),
        fetch_candles(http_session, pair, "M15", 96),
        fetch_order_book(http_session, pair),
        fetch_position_book(http_session, pair),
        return_exceptions=True
    )

    candles       = results[0] if not isinstance(results[0], Exception) else []
    daily_candles = results[1] if not isinstance(results[1], Exception) else []
    h4_candles    = results[2] if not isinstance(results[2], Exception) else []
    m15_candles   = results[3] if not isinstance(results[3], Exception) else []
    ob_raw        = results[4] if not isinstance(results[4], Exception) else None
    pb_raw        = results[5] if not isinstance(results[5], Exception) else None

    if not candles or len(candles) < 30:
        raise ValueError(f"Insufficient candle data for {pair} {tf}")

    of  = OF_ENGINE.analyze(candles, pair)
    vp  = calculate_volume_profile(candles)
    ob  = analyze_order_book(ob_raw, of.price)
    pb  = analyze_position_book(pb_raw, of.price, pair)
    af  = ADVANCED_FLOW.analyze(candles, ob_raw)

    cs = None
    if pair in FOREX_PAIRS:
        try:
            cs = await STRENGTH_CACHE.get(http_session)
        except Exception:
            pass

    return {
        "candles":       candles,
        "daily_candles": daily_candles,
        "h4_candles":    h4_candles,
        "m15_candles":   m15_candles,
        "of":            of,
        "vp":            vp,
        "ob":            ob,
        "pb":            pb,
        "af":            af,
        "cs":            cs,
        "ob_raw":        ob_raw,
        "pb_raw":        pb_raw,
    }

# ════════════════════════════════════════════════════════════════
#  END OF PART 1
# ════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
#  FOREX QUANT v8.1 — PART 2 OF 4 (FIXED)
#  ICT Structure Engine, Confluence Scoring, Strategy Engine,
#  ML Meta-Labeling Engine, Pattern Memory, QuantBrain,
#  Historical Backtest Engine, Prediction Tracker
#
#  FIXES APPLIED:
#  - OB strength threshold reduced from 40 to 20
#  - ICT Continuation allows pullback zone entry (not just OB/OTE)
#  - Regime filter relaxed — Continuation allowed in RANGING with penalty
#  - Hurst threshold for Mean Reversion raised to 0.52
#  - Mean Reversion fires on EITHER band breach OR VWAP z-score extreme
#  - Breakout VR threshold raised to 0.80, displacement lookback relaxed
#  - AMD fires during any Kill Zone (not just MANIPULATION phase)
#  - Regime contradiction penalty reduced from 25 total to 12 total
#  - Gold/Silver/NAS minimum pips reduced to realistic levels
#  - validate_signal_geometry uses instrument-appropriate minimums
#  - Confidence penalties recalibrated so multiple strategies can fire
#  - ICT Continuation no longer requires HTF=BULLISH/BEARISH strictly
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  ICT MARKET STRUCTURE ENGINE
# ════════════════════════════════════════════════════════════════

class ICTStructureEngine:

    def detect_swing_points(self,
                            candles:  List[Candle],
                            strength: int = 3
                            ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        highs: List[SwingPoint] = []
        lows:  List[SwingPoint] = []
        n = len(candles)
        if n < strength * 2 + 1:
            return highs, lows
        for i in range(strength, n - strength):
            c = candles[i]
            if all(c.high >= candles[i-j].high and
                   c.high >= candles[i+j].high
                   for j in range(1, strength + 1)):
                highs.append(SwingPoint(
                    price=c.high, index=i,
                    time=c.time, is_high=True,
                    strength=strength))
            if all(c.low <= candles[i-j].low and
                   c.low <= candles[i+j].low
                   for j in range(1, strength + 1)):
                lows.append(SwingPoint(
                    price=c.low, index=i,
                    time=c.time, is_high=False,
                    strength=strength))
        return highs, lows

    def analyze_structure(self,
                          candles:        List[Candle],
                          instrument:     str,
                          swing_strength: int = 3) -> MarketStructure:
        if len(candles) < 20:
            return MarketStructure()
        highs, lows = self.detect_swing_points(candles, swing_strength)
        if not highs or not lows:
            return MarketStructure()
        trend              = self._determine_trend(highs, lows)
        last_bos, last_choch = self._detect_bos_choch(
            candles, highs, lows, trend)
        int_highs, int_lows = self.detect_swing_points(candles, 1)
        internal_trend      = self._determine_trend(int_highs, int_lows)
        return MarketStructure(
            swing_highs    = highs[-10:],
            swing_lows     = lows[-10:],
            trend          = trend,
            last_bos       = last_bos,
            last_choch     = last_choch,
            internal_trend = internal_trend,
            htf_bias       = trend,
        )

    def _determine_trend(self,
                         highs: List[SwingPoint],
                         lows:  List[SwingPoint]) -> str:
        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"
        rh = sorted(highs, key=lambda x: x.index)[-4:]
        rl = sorted(lows,  key=lambda x: x.index)[-4:]
        hh = sum(1 for i in range(1, len(rh))
                 if rh[i].price > rh[i-1].price)
        lh = sum(1 for i in range(1, len(rh))
                 if rh[i].price < rh[i-1].price)
        hl = sum(1 for i in range(1, len(rl))
                 if rl[i].price > rl[i-1].price)
        ll = sum(1 for i in range(1, len(rl))
                 if rl[i].price < rl[i-1].price)
        bull = hh + hl
        bear = lh + ll
        if bull > bear and bull >= 2: return "BULLISH"
        if bear > bull and bear >= 2: return "BEARISH"
        return "NEUTRAL"

    def _detect_bos_choch(self,
                          candles: List[Candle],
                          highs:   List[SwingPoint],
                          lows:    List[SwingPoint],
                          trend:   str
                          ) -> Tuple[Optional[StructureType],
                                     Optional[StructureType]]:
        last_bos = last_choch = None
        if not highs or not lows or len(candles) < 5:
            return last_bos, last_choch
        price = candles[-1].close
        for sh in reversed(sorted(highs, key=lambda x: x.index)[-5:]):
            if price > sh.price:
                if trend == "BULLISH":
                    last_bos   = StructureType.BULLISH_BOS
                else:
                    last_choch = StructureType.BULLISH_CHOCH
                break
        for sl in reversed(sorted(lows, key=lambda x: x.index)[-5:]):
            if price < sl.price:
                if trend == "BEARISH":
                    last_bos   = StructureType.BEARISH_BOS
                else:
                    last_choch = StructureType.BEARISH_CHOCH
                break
        return last_bos, last_choch

    def map_liquidity_levels(self,
                             candles:       List[Candle],
                             daily_candles: Optional[List[Candle]],
                             instrument:    str) -> List[LiquidityLevel]:
        levels: List[LiquidityLevel] = []
        pip = pip_value(instrument)
        tol = pip * 5
        if len(candles) < 20:
            return levels
        highs = [c.high for c in candles]
        lows  = [c.low  for c in candles]
        levels.extend(self._find_equal_levels(
            highs, candles, tol, True))
        levels.extend(self._find_equal_levels(
            lows,  candles, tol, False))
        if daily_candles and len(daily_candles) >= 2:
            pd_c = daily_candles[-2]
            for price, ltype in [
                    (pd_c.high, "PDH"), (pd_c.low, "PDL")]:
                levels.append(LiquidityLevel(
                    price=price, level_type=ltype, strength=3,
                    zone_upper=price+tol, zone_lower=price-tol))
        ash, asl = self._get_asian_range(candles)
        if ash:
            levels.append(LiquidityLevel(
                price=ash, level_type="ASH", strength=2,
                zone_upper=ash+tol, zone_lower=ash-tol))
        if asl:
            levels.append(LiquidityLevel(
                price=asl, level_type="ASL", strength=2,
                zone_upper=asl+tol, zone_lower=asl-tol))
        return levels

    def _find_equal_levels(self,
                           values:    List[float],
                           candles:   List[Candle],
                           tolerance: float,
                           is_high:   bool) -> List[LiquidityLevel]:
        levels = []
        used   = set()
        for i in range(len(values)):
            if i in used:
                continue
            cluster = [i]
            for j in range(i+1, min(i+30, len(values))):
                if j not in used and abs(values[j]-values[i]) <= tolerance:
                    cluster.append(j)
                    used.add(j)
            if len(cluster) >= 2:
                avg   = sum(values[k] for k in cluster) / len(cluster)
                ltype = "EQUAL_HIGHS" if is_high else "EQUAL_LOWS"
                levels.append(LiquidityLevel(
                    price=avg, level_type=ltype,
                    strength=len(cluster),
                    zone_upper=avg+tolerance,
                    zone_lower=avg-tolerance))
                used.add(i)
        return levels[:3]

    def _get_asian_range(self,
                         candles: List[Candle]
                         ) -> Tuple[Optional[float], Optional[float]]:
        asian = []
        for c in candles[-24:]:
            if not c.time:
                continue
            try:
                dt = datetime.fromisoformat(
                    c.time.replace("Z", "+00:00"))
                if 0 <= dt.hour < 7:
                    asian.append(c)
            except Exception:
                continue
        if not asian:
            return None, None
        return (max(c.high for c in asian),
                min(c.low  for c in asian))

    def detect_sweep(self,
                     candles:    List[Candle],
                     levels:     List[LiquidityLevel],
                     instrument: str) -> List[LiquidityLevel]:
        swept  = []
        recent = candles[-10:]
        pip    = pip_value(instrument)
        tol    = pip * 2
        for level in levels:
            if level.swept:
                continue
            for c in recent:
                if (level.level_type in ("EQUAL_HIGHS","PDH","ASH")
                        and c.high > level.price - tol
                        and c.close < level.price
                        and c.body_abs > 0):
                    level.swept      = True
                    level.sweep_time = c.time
                    swept.append(level)
                    break
                elif (level.level_type in ("EQUAL_LOWS","PDL","ASL")
                      and c.low < level.price + tol
                      and c.close > level.price
                      and c.body_abs > 0):
                    level.swept      = True
                    level.sweep_time = c.time
                    swept.append(level)
                    break
        return swept

    def detect_displacement(self,
                            candles:  List[Candle],
                            lookback: int = 5) -> Optional[Candle]:
        if len(candles) < lookback + 3:
            return None
        recent   = candles[-(lookback + 5):]
        avg_body = sum(c.body_abs for c in recent[:-lookback]) / max(
            1, len(recent) - lookback)
        # Check recent candles for displacement
        for c in reversed(recent[-lookback:]):
            if c.is_displacement_vs(avg_body):
                return c
        # FALLBACK: accept if most recent candle is 1.3x average
        # (catches early breakout displacement)
        if candles and avg_body > 0:
            if candles[-1].body_abs >= avg_body * 1.3:
                return candles[-1]
        return None

    def detect_fvgs(self,
                    candles:    List[Candle],
                    instrument: str,
                    lookback:   int = 30) -> List[FairValueGap]:
        fvgs: List[FairValueGap] = []
        pip  = pip_value(instrument)
        if len(candles) < 3:
            return fvgs
        recent = candles[-min(lookback, len(candles)):]
        cp     = candles[-1].close
        for i in range(1, len(recent) - 1):
            prev = recent[i-1]
            curr = recent[i]
            nxt  = recent[i+1]
            if nxt.low > prev.high:
                gap_size = nxt.low - prev.high
                strength = min(100.0, gap_size / pip * 2)
                if strength >= 8:   # lowered from 10
                    mid    = prev.high + gap_size / 2
                    filled = cp <= prev.high
                    fvgs.append(FairValueGap(
                        upper=nxt.low, lower=prev.high,
                        mid=mid, direction="BULLISH",
                        time=curr.time, filled=filled,
                        strength=strength))
            elif nxt.high < prev.low:
                gap_size = prev.low - nxt.high
                strength = min(100.0, gap_size / pip * 2)
                if strength >= 8:   # lowered from 10
                    mid    = nxt.high + gap_size / 2
                    filled = cp >= prev.low
                    fvgs.append(FairValueGap(
                        upper=prev.low, lower=nxt.high,
                        mid=mid, direction="BEARISH",
                        time=curr.time, filled=filled,
                        strength=strength))
        fvgs.sort(
            key=lambda x: (not x.filled, x.strength), reverse=True)
        return fvgs[:8]

    def detect_order_blocks(self,
                            candles:    List[Candle],
                            structure:  MarketStructure,
                            instrument: str,
                            lookback:   int = 30) -> List[OrderBlock]:
        obs:  List[OrderBlock] = []
        pip  = pip_value(instrument)
        if len(candles) < 5:
            return obs
        recent   = candles[-min(lookback, len(candles)):]
        avg_body = (sum(c.body_abs for c in recent) /
                    max(1, len(recent)))
        fvgs     = self.detect_fvgs(candles, instrument, lookback)

        for i in range(1, len(recent) - 2):
            c    = recent[i]
            nxt1 = recent[i+1]

            # ── Bullish OB ────────────────────────────────────────
            if (not c.is_bullish and
                    nxt1.is_displacement_vs(avg_body) and
                    nxt1.is_bullish):
                broke_struct = (
                    structure.trend in ("BULLISH",) or
                    structure.last_bos == StructureType.BULLISH_BOS or
                    structure.last_choch == StructureType.BULLISH_CHOCH or
                    structure.internal_trend == "BULLISH")

                left_fvg = any(
                    f.direction == "BULLISH" and
                    f.lower >= c.high - pip * 8   # relaxed from 5
                    for f in fvgs)

                # FIX: Reduced strength threshold from 40 to 20
                strength = 0.0
                if broke_struct: strength += 35
                if left_fvg:     strength += 35
                strength += min(30.0, nxt1.body_abs / pip * 0.2)

                if strength >= 20:   # was 40 — now much more permissive
                    obs.append(OrderBlock(
                        high=c.high, low=c.low,
                        mid=c.low + (c.high - c.low) / 2,
                        direction="DEMAND", time=c.time,
                        valid=True, left_fvg=left_fvg,
                        broke_structure=broke_struct,
                        strength=strength))

            # ── Bearish OB ────────────────────────────────────────
            elif (c.is_bullish and
                    nxt1.is_displacement_vs(avg_body) and
                    not nxt1.is_bullish):
                broke_struct = (
                    structure.trend in ("BEARISH",) or
                    structure.last_bos == StructureType.BEARISH_BOS or
                    structure.last_choch == StructureType.BEARISH_CHOCH or
                    structure.internal_trend == "BEARISH")

                left_fvg = any(
                    f.direction == "BEARISH" and
                    f.upper <= c.low + pip * 8   # relaxed from 5
                    for f in fvgs)

                strength = 0.0
                if broke_struct: strength += 35
                if left_fvg:     strength += 35
                strength += min(30.0, nxt1.body_abs / pip * 0.2)

                if strength >= 20:   # was 40
                    obs.append(OrderBlock(
                        high=c.high, low=c.low,
                        mid=c.low + (c.high - c.low) / 2,
                        direction="SUPPLY", time=c.time,
                        valid=True, left_fvg=left_fvg,
                        broke_structure=broke_struct,
                        strength=strength))

        # Invalidate violated OBs
        cp = candles[-1].close
        for ob in obs:
            if ob.direction == "DEMAND" and cp < ob.low - pip * 3:
                ob.valid = False
            if ob.direction == "SUPPLY" and cp > ob.high + pip * 3:
                ob.valid = False

        valid = [ob for ob in obs if ob.valid]
        valid.sort(key=lambda x: x.strength, reverse=True)
        return valid[:6]

    def detect_breaker_blocks(self,
                              obs:        List[OrderBlock],
                              instrument: str) -> List[OrderBlock]:
        breakers = []
        for ob in obs:
            if not ob.valid:
                breakers.append(OrderBlock(
                    high=ob.high, low=ob.low, mid=ob.mid,
                    direction=(
                        "SUPPLY" if ob.direction == "DEMAND"
                        else "DEMAND"),
                    time=ob.time, valid=True,
                    strength=ob.strength * 0.7))
        return breakers

    def get_ote_zone(self,
                     highs:     List[SwingPoint],
                     lows:      List[SwingPoint],
                     direction: str) -> Optional[Tuple[float, float]]:
        if not highs or not lows:
            return None
        if direction == "LONG":
            sh   = sorted(highs, key=lambda x: x.index)
            sl   = sorted(lows,  key=lambda x: x.index)
            if not sh or not sl: return None
            lh   = sh[-1]
            pls  = [l for l in sl if l.index < lh.index]
            if not pls: return None
            ll   = pls[-1]
            fibs = MATH.fibonacci_levels(lh.price, ll.price, "LONG")
            return (fibs.get("0.786", 0.0), fibs.get("0.618", 0.0))
        else:
            sl   = sorted(lows,  key=lambda x: x.index)
            sh   = sorted(highs, key=lambda x: x.index)
            if not sl or not sh: return None
            ll   = sl[-1]
            phs  = [h for h in sh if h.index < ll.index]
            if not phs: return None
            lh   = phs[-1]
            fibs = MATH.fibonacci_levels(lh.price, ll.price, "SHORT")
            return (fibs.get("0.618", 0.0), fibs.get("0.786", 0.0))

    def assess_amd_phase(self,
                         candles:    List[Candle],
                         session:    SessionData,
                         asian_high: Optional[float],
                         asian_low:  Optional[float]) -> str:
        if not asian_high or not asian_low or not candles:
            return session.amd_phase
        price = candles[-1].close
        if price > asian_high:   return "DISTRIBUTION_UP"
        elif price < asian_low:  return "DISTRIBUTION_DOWN"
        return "ACCUMULATION"

    def full_analysis(self,
                      candles:       List[Candle],
                      daily_candles: Optional[List[Candle]],
                      session:       SessionData,
                      instrument:    str) -> Dict:
        structure    = self.analyze_structure(candles, instrument)
        liq_levels   = self.map_liquidity_levels(
            candles, daily_candles, instrument)
        swept        = self.detect_sweep(candles, liq_levels, instrument)
        displacement = self.detect_displacement(candles)
        fvgs         = self.detect_fvgs(candles, instrument)
        obs          = self.detect_order_blocks(
            candles, structure, instrument)
        breakers     = self.detect_breaker_blocks(obs, instrument)
        ote_long     = self.get_ote_zone(
            structure.swing_highs, structure.swing_lows, "LONG")
        ote_short    = self.get_ote_zone(
            structure.swing_highs, structure.swing_lows, "SHORT")
        asian_h, asian_l = self._get_asian_range(candles)
        amd_phase    = self.assess_amd_phase(
            candles, session, asian_h, asian_l)
        return {
            "structure":    structure,
            "liq_levels":   liq_levels,
            "swept":        swept,
            "displacement": displacement,
            "fvgs":         fvgs,
            "obs":          obs,
            "breakers":     breakers,
            "ote_long":     ote_long,
            "ote_short":    ote_short,
            "asian_high":   asian_h,
            "asian_low":    asian_l,
            "amd_phase":    amd_phase,
        }


ICT_ENGINE = ICTStructureEngine()

# ════════════════════════════════════════════════════════════════
#  REGIME ENGINE  (FIXED: Hurst fallback + relaxed thresholds)
# ════════════════════════════════════════════════════════════════

class RegimeEngine:

    def detect(self,
               candles:       List[Candle],
               daily_candles: Optional[List[Candle]] = None,
               instrument:    str = "") -> RegimeData:
        if len(candles) < 30:
            return RegimeData()

        closes  = [c.close for c in candles]
        returns = [closes[i] / closes[i-1] - 1.0
                   for i in range(1, len(closes))]

        hurst   = MATH.hurst_exponent(closes[-100:])
        frac    = MATH.fractal_dimension(closes[-50:])
        entropy = MATH.price_entropy(returns[-50:])
        ac1     = MATH.autocorrelation(returns[-30:], lag=1)
        vr      = MATH.variance_ratio(closes[-50:], q=5)
        er      = MATH.efficiency_ratio(closes, period=20)
        vol_r   = MATH.volatility_ratio(candles, 14, 50)
        lr      = MATH.linear_regression(closes, period=50)
        slope   = lr.get("slope", 0.0)
        r2      = lr.get("r_squared", 0.0)

        adr_pct = 0.0
        adr_val = 0.0
        if daily_candles and len(daily_candles) >= 5:
            adr_pct, adr_val = MATH.adr_consumed(
                daily_candles, candles[-1].close)

        contradictory = MATH.check_regime_contradiction(hurst, er, ac1)
        regime        = self._classify(
            hurst, frac, entropy, ac1, vr, er, vol_r, r2)
        rec_strat     = self._recommend(regime, vol_r, er)

        return RegimeData(
            regime               = regime,
            hurst_exponent       = hurst,
            fractal_dimension    = frac,
            entropy              = entropy,
            autocorr_lag1        = ac1,
            variance_ratio       = vr,
            efficiency_ratio     = er,
            volatility_ratio     = vol_r,
            adr_consumed_pct     = adr_pct,
            adr_today            = adr_val,
            adr_average          = adr_val,
            trend_slope          = slope,
            r_squared            = r2,
            recommended_strategy = rec_strat,
            contradictory        = contradictory,
        )

    def _classify(self, hurst, frac, entropy, ac1,
                  vr, er, vol_r, r2) -> RegimeType:
        # Hard overrides first
        if vol_r < 0.5:
            return RegimeType.COMPRESSION
        if vol_r > 2.5:
            return RegimeType.EXPANSION
        if entropy > 0.9 and vol_r > 1.5:
            return RegimeType.CHAOTIC

        tv = bv = 0

        # Hurst — FIX: use ER to break ties when Hurst is ambiguous
        if hurst > 0.58:
            tv += 2
        elif hurst < 0.42:
            bv += 2
        elif 0.46 <= hurst <= 0.54:
            # Ambiguous Hurst — let ER and R2 decide
            if er > 0.40 and r2 > 0.50:
                tv += 2   # ER says trending
            elif er < 0.25 and r2 < 0.30:
                bv += 2   # ER says ranging
            # else: no vote — truly ambiguous
        elif hurst > 0.54:
            tv += 1
        else:
            bv += 1

        if frac < 1.35:    tv += 1
        elif frac > 1.65:  bv += 1

        if ac1 > 0.15:     tv += 1
        elif ac1 < -0.15:  bv += 1

        if vr > 1.3:       tv += 1
        elif vr < 0.7:     bv += 1

        if er > 0.45:      tv += 1
        elif er < 0.25:    bv += 1

        if r2 > 0.65:      tv += 1
        elif r2 < 0.25:    bv += 1

        if entropy < 0.5:  tv += 1
        elif entropy > 0.8: bv += 1

        if tv >= bv * 1.5:
            return (RegimeType.STRONG_TREND
                    if tv >= 5 else RegimeType.WEAK_TREND)
        if bv >= tv * 1.5:
            return RegimeType.RANGING
        # FIX: Default to WEAK_TREND instead of RANGING
        # when votes are close but trending signals present
        if tv > bv:
            return RegimeType.WEAK_TREND
        return RegimeType.RANGING

    def _recommend(self, regime: RegimeType,
                   vol_r: float,
                   er: float) -> Optional[StrategyType]:
        if regime in (RegimeType.STRONG_TREND, RegimeType.WEAK_TREND):
            return StrategyType.ICT_CONTINUATION
        if regime == RegimeType.RANGING:
            return StrategyType.ICT_REVERSAL
        if regime == RegimeType.COMPRESSION:
            return StrategyType.BREAKOUT
        return None

    def is_strategy_valid(self, strategy: StrategyType,
                           regime: RegimeData,
                           session: SessionData) -> bool:
        """
        FIX: Much more permissive strategy activation.
        Only CHAOTIC blocks everything.
        Other regimes allow strategies with confidence penalties
        applied INSIDE the strategy rather than blanket rejection.
        """
        # Hard block: chaotic market
        if regime.regime == RegimeType.CHAOTIC:
            return False

        if strategy == StrategyType.ICT_CONTINUATION:
            # FIX: Allow in all non-chaotic regimes
            # Confidence penalty applied inside strategy for RANGING
            return True

        if strategy == StrategyType.ICT_REVERSAL:
            # Works in any regime
            return True

        if strategy == StrategyType.AMD_SESSION:
            # FIX: Only require Kill Zone — remove AMD phase requirement
            return session.is_kill_zone

        if strategy == StrategyType.RETAIL_CONTRARIAN:
            return True

        if strategy == StrategyType.BREAKOUT:
            # FIX: Allow in COMPRESSION and RANGING
            return regime.regime in (
                RegimeType.COMPRESSION,
                RegimeType.RANGING,
                RegimeType.WEAK_TREND)

        if strategy == StrategyType.MEAN_REVERSION:
            # FIX: Allow when Hurst is below 0.52 (was 0.50)
            return regime.hurst_exponent < 0.52

        if strategy == StrategyType.CORRELATION_DIV:
            return regime.regime != RegimeType.CHAOTIC

        return True


REGIME_ENGINE = RegimeEngine()

# ════════════════════════════════════════════════════════════════
#  CONFLUENCE SCORING ENGINE  (9 INDEPENDENT CATEGORIES)
# ════════════════════════════════════════════════════════════════

class ConfluenceEngine:

    def score(self,
              of:            OrderFlowData,
              af:            Optional[AdvancedFlowData],
              vp:            Optional[VolumeProfile],
              ob:            Optional[OrderBookData],
              pb:            Optional[PositionBookData],
              cs:            Optional[Dict],
              ict:           Optional[Dict],
              regime:        Optional[RegimeData],
              session:       Optional[SessionData],
              instrument:    str,
              current_price: float
              ) -> Tuple[List[ConfluenceFactor], float, str]:

        factors: List[ConfluenceFactor] = []

        def add(name, direction, strength, desc, category, weight=1.0):
            factors.append(ConfluenceFactor(
                name=name, direction=direction,
                strength=min(100.0, max(0.0, float(strength))),
                description=desc,
                category=category,
                weight=weight))

        # ── CAT 1: Order Flow ────────────────────────────────────
        of_score, of_dir = self._score_order_flow(of)
        add("Order Flow", of_dir, of_score,
            self._describe_of(of, of_dir), "FLOW", 1.2)

        # ── CAT 2: Market Structure ──────────────────────────────
        if ict:
            struct = ict.get("structure")
            if struct:
                s_dir, s_str, s_desc = self._score_structure(struct)
                add("Market Structure", s_dir, s_str,
                    s_desc, "STRUCTURE", 1.5)

        # ── CAT 3: Institutional Books ───────────────────────────
        i_dir, i_str, i_desc = self._score_institutional(ob, pb)
        if i_str > 0:
            add("Institutional Books", i_dir, i_str,
                i_desc, "INSTITUTIONAL", 1.3)

        # ── CAT 4: Smart Money ───────────────────────────────────
        if af:
            sm_dir, sm_str, sm_desc = self._score_smart_money(af, of)
            if sm_str > 0:
                add("Smart Money", sm_dir, sm_str,
                    sm_desc, "SMART_MONEY", 1.4)

        # ── CAT 5: Currency Strength ─────────────────────────────
        if cs and "_" in instrument:
            cs_dir, cs_str, cs_desc = self._score_cs(cs, instrument)
            if cs_str > 0:
                add("Currency Strength", cs_dir, cs_str,
                    cs_desc, "STRENGTH", 1.1)

        # ── CAT 6: Liquidity / ICT ───────────────────────────────
        if ict:
            lz_dir, lz_str, lz_desc = self._score_liquidity(
                ict, current_price, instrument)
            if lz_str > 0:
                add("Liquidity Zones", lz_dir, lz_str,
                    lz_desc, "LIQUIDITY", 1.6)

        # ── CAT 7: Regime ────────────────────────────────────────
        if regime:
            rv_dir, rv_str, rv_desc = self._score_regime(regime, of)
            if rv_str > 0:
                add("Regime", rv_dir, rv_str,
                    rv_desc, "REGIME", 0.9)

        # ── CAT 8: Session ───────────────────────────────────────
        if session:
            ss_dir, ss_str, ss_desc = self._score_session(
                session, instrument)
            if ss_str > 0:
                add("Session Timing", ss_dir, ss_str,
                    ss_desc, "SESSION", 1.0)

        # ── CAT 9: Volume Profile ────────────────────────────────
        if vp:
            vp_dir, vp_str, vp_desc = self._score_vp(vp, current_price)
            if vp_str > 0:
                add("Volume Profile", vp_dir, vp_str,
                    vp_desc, "VOLUME_PROFILE", 1.1)

        # ── Direction & confidence ────────────────────────────────
        bull = [f for f in factors if f.direction == "BULLISH"]
        bear = [f for f in factors if f.direction == "BEARISH"]

        bull_score = sum(f.strength * f.weight for f in bull)
        bear_score = sum(f.strength * f.weight for f in bear)

        if bull_score == 0 and bear_score == 0:
            return factors, 0.0, "NONE"

        direction = "LONG" if bull_score >= bear_score else "SHORT"
        aligned   = bull if direction == "LONG" else bear
        opposing  = bear if direction == "LONG" else bull

        base  = 40.0 + (len(aligned) / 9) * 40.0
        bonus = 0.0
        if aligned:
            avg_str = (sum(f.strength * f.weight for f in aligned)
                       / len(aligned))
            bonus  += avg_str * 0.15

        penalty = len(opposing) * 3.5  # FIX: reduced from 4.0

        if regime:
            if regime.regime in (RegimeType.STRONG_TREND,
                                  RegimeType.WEAK_TREND):
                bonus += 5.0
            # FIX: Reduced contradiction penalty from 10 to 5
            if regime.contradictory:
                penalty += 5.0   # was 10.0

        if session and session.is_kill_zone:
            bonus += 6.0

        if ict:
            if ict.get("swept"):        bonus += 8.0
            if ict.get("displacement"): bonus += 5.0

        confidence = float(np.clip(base + bonus - penalty, 35.0, 94.0))
        return factors, confidence, direction

    # ── Scorers ───────────────────────────────────────────────────

    def _score_order_flow(self,
                          of: OrderFlowData) -> Tuple[float, str]:
        bull = bear = 0.0
        if of.cvd > 0:                       bull += 25
        elif of.cvd < 0:                     bear += 25
        if of.imbalance_zscore > 1.5:        bull += 20
        elif of.imbalance_zscore < -1.5:     bear += 20
        elif of.imbalance_zscore > 0.8:      bull += 10
        elif of.imbalance_zscore < -0.8:     bear += 10
        if of.vwap_zscore > 0.5:             bull += 15
        elif of.vwap_zscore < -0.5:          bear += 15
        if of.delta_momentum > 0:            bull += 20
        elif of.delta_momentum < 0:          bear += 20
        if of.buying_climax:                 bear += 20
        if of.selling_climax:                bull += 20
        total = bull + bear
        if total == 0:                  return 0.0, "NEUTRAL"
        if bull > bear:                 return min(100.0, bull), "BULLISH"
        if bear > bull:                 return min(100.0, bear), "BEARISH"
        return 0.0, "NEUTRAL"

    def _describe_of(self, of: OrderFlowData, direction: str) -> str:
        z = of.imbalance_zscore
        if abs(z) > 2.0:
            return (f"Statistically extreme "
                    f"{'buying' if direction=='BULLISH' else 'selling'} "
                    f"(z={z:+.1f}σ) — CVD {fmt_signed(of.cvd)}")
        return (f"{'Buying' if direction=='BULLISH' else 'Selling'} "
                f"dominant — CVD {fmt_signed(of.cvd)}, "
                f"imbalance {of.imbalance:+.1f}%")

    def _score_structure(self,
                         struct: MarketStructure
                         ) -> Tuple[str, float, str]:
        if struct.trend == "BULLISH":
            if struct.last_bos == StructureType.BULLISH_BOS:
                return "BULLISH", 90.0, "Confirmed bullish BOS"
            if struct.last_choch == StructureType.BULLISH_CHOCH:
                return "BULLISH", 85.0, "Bullish CHOCH — reversal confirmed"
            return "BULLISH", 70.0, "Bullish structure (HH + HL)"
        if struct.trend == "BEARISH":
            if struct.last_bos == StructureType.BEARISH_BOS:
                return "BEARISH", 90.0, "Confirmed bearish BOS"
            if struct.last_choch == StructureType.BEARISH_CHOCH:
                return "BEARISH", 85.0, "Bearish CHOCH — reversal confirmed"
            return "BEARISH", 70.0, "Bearish structure (LH + LL)"
        return "NEUTRAL", 0.0, "Mixed structure"

    def _score_institutional(self,
                              ob: Optional[OrderBookData],
                              pb: Optional[PositionBookData]
                              ) -> Tuple[str, float, str]:
        bull = bear = 0.0
        if ob:
            if ob.pressure_dir == "BULLISH":   bull += 30
            elif ob.pressure_dir == "BEARISH": bear += 30
        if pb:
            if pb.contrarian_signal == "BULLISH":
                base = 40.0 + (15.0 if pb.skew_change < -2 else 0.0)
                bull += base
            elif pb.contrarian_signal == "BEARISH":
                base = 40.0 + (15.0 if pb.skew_change > 2 else 0.0)
                bear += base
            if pb.squeeze_potential == "HIGH":
                if pb.long_pct > pb.short_pct: bear += 20
                else:                           bull += 20
        total = bull + bear
        if total == 0: return "NEUTRAL", 0.0, "No institutional signal"
        if bull >= bear:
            return ("BULLISH", min(100.0, bull),
                    f"Books favour bulls "
                    f"(retail {pb.short_pct:.0f}% short)"
                    if pb else "Order book bullish")
        return ("BEARISH", min(100.0, bear),
                f"Books favour bears "
                f"(retail {pb.long_pct:.0f}% long)"
                if pb else "Order book bearish")

    def _score_smart_money(self,
                           af: AdvancedFlowData,
                           of: OrderFlowData) -> Tuple[str, float, str]:
        bull = bear = 0.0
        if "BULLISH" in af.informed_signal:   bull += 40
        elif "BEARISH" in af.informed_signal: bear += 40
        if af.aggressor_side == "BUYERS":     bull += 25
        elif af.aggressor_side == "SELLERS":  bear += 25
        if af.absorption_ratio > 1.5:         bull += 20
        elif af.absorption_ratio < 0.67:      bear += 20
        if af.iceberg_count > 0:
            if of.cvd > 0: bull += 15
            else:          bear += 15
        if af.vpin >= 0.5:
            if of.cvd > 0: bull += 10
            else:          bear += 10
        total = bull + bear
        if total == 0: return "NEUTRAL", 0.0, ""
        direction = "BULLISH" if bull >= bear else "BEARISH"
        desc = (f"Smart money {af.smart_money_activity} — "
                f"Informed: {af.informed_signal} | "
                f"Aggressor: {af.aggressor_side}")
        return direction, min(100.0, max(bull, bear)), desc

    def _score_cs(self,
                  cs:         Dict,
                  instrument: str) -> Tuple[str, float, str]:
        parts = instrument.split("_")
        if len(parts) != 2: return "NEUTRAL", 0.0, ""
        bcs = cs.get(parts[0]); qcs = cs.get(parts[1])
        if not bcs or not qcs: return "NEUTRAL", 0.0, ""
        diff   = bcs.strength - qcs.strength
        z_diff = bcs.strength_zscore - qcs.strength_zscore
        if abs(diff) < 0.05: return "NEUTRAL", 0.0, "Equal strength"
        strength = min(100.0, abs(diff)*30 + abs(z_diff)*10)
        if diff > 0:
            return ("BULLISH", strength,
                    f"{parts[0]} ({bcs.strength:+.2f}%) stronger — "
                    f"{bcs.trend}")
        return ("BEARISH", strength,
                f"{parts[1]} ({qcs.strength:+.2f}%) stronger — "
                f"{qcs.trend}")

    def _score_liquidity(self,
                         ict:        Dict,
                         price:      float,
                         instrument: str) -> Tuple[str, float, str]:
        bull = bear = 0.0
        desc_parts = []
        swept = ict.get("swept", [])
        for level in swept:
            if level.level_type in ("EQUAL_LOWS","PDL","ASL"):
                bull += 30
                desc_parts.append(
                    f"Lows swept @ {fmt_price(level.price, instrument)}")
            elif level.level_type in ("EQUAL_HIGHS","PDH","ASH"):
                bear += 30
                desc_parts.append(
                    f"Highs swept @ {fmt_price(level.price, instrument)}")
        disp = ict.get("displacement")
        if disp:
            if disp.is_bullish: bull += 25
            else:               bear += 25
            desc_parts.append("Displacement confirmed")
        pip = pip_value(instrument)
        for ob_block in ict.get("obs", [])[:3]:
            dist = abs(ob_block.mid - price)
            if dist < pip * 60:   # increased from 50
                if ob_block.direction == "DEMAND" and price > ob_block.low:
                    bull += min(20.0, ob_block.strength * 0.2)
                    desc_parts.append(
                        f"Demand OB @ "
                        f"{fmt_price(ob_block.low, instrument)}")
                elif ob_block.direction == "SUPPLY" and price < ob_block.high:
                    bear += min(20.0, ob_block.strength * 0.2)
                    desc_parts.append(
                        f"Supply OB @ "
                        f"{fmt_price(ob_block.high, instrument)}")
        for fvg in ict.get("fvgs", [])[:3]:
            if not fvg.filled:
                dist = abs(fvg.mid - price)
                if dist < pip * 40:   # increased from 30
                    if fvg.direction == "BULLISH" and price > fvg.lower:
                        bull += 15
                    elif fvg.direction == "BEARISH" and price < fvg.upper:
                        bear += 15
        total = bull + bear
        if total == 0: return "NEUTRAL", 0.0, ""
        desc = " | ".join(desc_parts[:3]) if desc_parts else "ICT aligned"
        if bull >= bear: return "BULLISH", min(100.0, bull), desc
        return "BEARISH", min(100.0, bear), desc

    def _score_regime(self,
                      regime: RegimeData,
                      of:     OrderFlowData) -> Tuple[str, float, str]:
        if regime.regime == RegimeType.CHAOTIC:
            return "NEUTRAL", 0.0, "Chaotic — no trade"
        if regime.regime == RegimeType.COMPRESSION:
            return ("NEUTRAL", 30.0,
                    f"Compression (VR:{regime.volatility_ratio:.2f}) "
                    f"— breakout imminent")
        if regime.regime in (RegimeType.STRONG_TREND,
                              RegimeType.WEAK_TREND):
            # FIX: contradiction penalty reduced from 15 to 8
            penalty = " ⚠️ Contradictory" if regime.contradictory else ""
            dir_    = "BULLISH" if of.cvd > 0 else "BEARISH"
            score   = 50.0 - (8.0 if regime.contradictory else 0.0)
            return (dir_, score,
                    f"{regime.regime.value} "
                    f"(H:{regime.hurst_exponent:.2f} "
                    f"ER:{regime.efficiency_ratio:.2f}){penalty}")
        # RANGING — still gives a moderate score
        dir_ = "BULLISH" if of.cvd > 0 else "BEARISH"
        return (dir_, 25.0,
                f"Ranging (H:{regime.hurst_exponent:.2f})")

    def _score_session(self,
                       session:    SessionData,
                       instrument: str) -> Tuple[str, float, str]:
        if session.session_type == SessionType.OFF_HOURS:
            return ("NEUTRAL", 10.0, "Off-hours — reduced liquidity ⚠️")
        if session.is_kill_zone:
            return ("NEUTRAL", 65.0,
                    f"🎯 {session.kill_zone_name}")
        if session.session_type == SessionType.OVERLAP:
            return ("NEUTRAL", 55.0, "London-NY Overlap — max liquidity")
        if session.session_type == SessionType.LONDON:
            return ("NEUTRAL", 40.0, "London session")
        if session.session_type == SessionType.NEW_YORK:
            return ("NEUTRAL", 35.0, "New York session")
        return ("NEUTRAL", 20.0, f"{session.session_name}")

    def _score_vp(self,
                  vp:    VolumeProfile,
                  price: float) -> Tuple[str, float, str]:
        in_va = vp.val <= price <= vp.vah
        if price > vp.poc:
            return ("BULLISH", 40.0,
                    f"Above POC {fmt_price(vp.poc, '')} — buyers control"
                    + (" (VA)" if in_va else " (outside VA)"))
        elif price < vp.poc:
            return ("BEARISH", 40.0,
                    f"Below POC {fmt_price(vp.poc, '')} — sellers control"
                    + (" (VA)" if in_va else " (outside VA)"))
        return ("NEUTRAL", 20.0, "At POC — equilibrium")


CONFLUENCE = ConfluenceEngine()

# ════════════════════════════════════════════════════════════════
#  SIGNAL GEOMETRY VALIDATOR  (FIXED: ATR-relative minimums)
# ════════════════════════════════════════════════════════════════

def get_min_pips_fixed(instrument: str,
                       atr_pips:   float = 0.0
                       ) -> Tuple[float, float]:
    """
    FIX: Instrument-appropriate minimums, with ATR fallback.
    Returns (min_target_pips, min_stop_pips).
    """
    if "XAU" in instrument:
        # Gold: use ATR-relative if available
        if atr_pips > 0:
            return max(40.0, atr_pips * 0.4), max(25.0, atr_pips * 0.25)
        return 50.0, 30.0
    if "XAG" in instrument:
        if atr_pips > 0:
            return max(20.0, atr_pips * 0.4), max(12.0, atr_pips * 0.25)
        return 20.0, 12.0
    if "NAS" in instrument:
        if atr_pips > 0:
            return max(10.0, atr_pips * 0.4), max(6.0, atr_pips * 0.25)
        return 10.0, 6.0
    if "JPY" in instrument:
        return 8.0, 5.0    # JPY pairs smaller pip values
    # Standard forex
    return 10.0, 6.0       # reduced from 12/8


def validate_signal_geometry_fixed(direction:  str,
                                    entry:      float,
                                    target:     float,
                                    stop:       float,
                                    instrument: str,
                                    atr:        float = 0.0,
                                    min_rr:     float = MIN_RR_RATIO
                                    ) -> SignalValidation:
    """
    Fixed version with ATR-relative minimums and proper direction logic.
    """
    pip      = pip_value(instrument)
    atr_pips = atr / pip if atr > 0 else 0.0
    min_tgt, min_stp = get_min_pips_fixed(instrument, atr_pips)

    if direction == "LONG":
        if target <= entry:
            return SignalValidation(
                False,
                f"LONG target {fmt_price(target, instrument)} must be "
                f"ABOVE entry {fmt_price(entry, instrument)}",
                0.0, 0.0, 0.0, entry, target, stop)
        if stop >= entry:
            return SignalValidation(
                False,
                f"LONG stop {fmt_price(stop, instrument)} must be "
                f"BELOW entry {fmt_price(entry, instrument)}",
                0.0, 0.0, 0.0, entry, target, stop)
        reward = target - entry
        risk   = entry  - stop

    elif direction == "SHORT":
        if target >= entry:
            return SignalValidation(
                False,
                f"SHORT target {fmt_price(target, instrument)} must be "
                f"BELOW entry {fmt_price(entry, instrument)}",
                0.0, 0.0, 0.0, entry, target, stop)
        if stop <= entry:
            return SignalValidation(
                False,
                f"SHORT stop {fmt_price(stop, instrument)} must be "
                f"ABOVE entry {fmt_price(entry, instrument)}",
                0.0, 0.0, 0.0, entry, target, stop)
        reward = entry  - target
        risk   = stop   - entry
    else:
        return SignalValidation(
            False, f"Invalid direction: {direction}",
            0.0, 0.0, 0.0, entry, target, stop)

    reward_pips = reward / pip
    risk_pips   = risk   / pip

    if reward_pips < min_tgt:
        return SignalValidation(
            False,
            f"Target too close: {reward_pips:.1f}p "
            f"(min {min_tgt:.0f}p for {instrument})",
            0.0, reward_pips, risk_pips, entry, target, stop)

    if risk_pips < min_stp:
        return SignalValidation(
            False,
            f"Stop too tight: {risk_pips:.1f}p "
            f"(min {min_stp:.0f}p for {instrument})",
            0.0, reward_pips, risk_pips, entry, target, stop)

    if risk == 0:
        return SignalValidation(
            False, "Zero risk", 0.0, reward_pips, risk_pips,
            entry, target, stop)

    rr = reward / risk   # CORRECT: reward ÷ risk

    if rr < min_rr:
        return SignalValidation(
            False,
            f"R/R {rr:.2f} below minimum {min_rr} "
            f"(reward {reward_pips:.1f}p / risk {risk_pips:.1f}p)",
            rr, reward_pips, risk_pips, entry, target, stop)

    return SignalValidation(
        True, "Valid", rr, reward_pips, risk_pips,
        entry, target, stop)

# ════════════════════════════════════════════════════════════════
#  STRATEGY ENGINE  (ALL 7 STRATEGIES — FIXED)
# ════════════════════════════════════════════════════════════════

class StrategyEngine:

    def run_all(self,
                candles:       List[Candle],
                of:            OrderFlowData,
                af:            Optional[AdvancedFlowData],
                vp:            Optional[VolumeProfile],
                ob:            Optional[OrderBookData],
                pb:            Optional[PositionBookData],
                cs:            Optional[Dict],
                ict:           Dict,
                regime:        RegimeData,
                session:       SessionData,
                instrument:    str,
                daily_candles: Optional[List[Candle]] = None
                ) -> Optional[StrategySignal]:

        candidates: List[StrategySignal] = []
        atr = MATH.calculate_atr(candles)

        for strat_type in StrategyType:
            if not REGIME_ENGINE.is_strategy_valid(
                    strat_type, regime, session):
                continue

            sig = self._run_one(
                strat_type, candles, of, af, vp, ob, pb,
                cs, ict, regime, session, instrument,
                daily_candles, atr)

            if not sig:
                continue
            if sig.confidence < MIN_CONFIDENCE:
                continue

            # CRITICAL: validate geometry using FIXED validator
            val = validate_signal_geometry_fixed(
                sig.direction, sig.entry, sig.target,
                sig.stop, instrument, atr)

            if not val.is_valid:
                log.debug(
                    f"Signal rejected [{strat_type.value}] "
                    f"{instrument}: {val.reason}")
                continue

            # Store validated geometry
            sig.rr_ratio    = val.rr_ratio
            sig.reward_pips = val.reward_pips
            sig.risk_pips   = val.risk_pips

            # Compute management levels
            sig.breakeven_price = calculate_breakeven_price(
                sig.direction, sig.entry, sig.target)
            sig.trailing_price  = calculate_trailing_price(
                sig.direction, sig.entry, sig.target)

            candidates.append(sig)

        if not candidates:
            return None

        def priority(s: StrategySignal) -> float:
            sc = s.confidence
            if s.regime_aligned:   sc += 5
            if s.session_aligned:  sc += 5
            if s.htf_aligned:      sc += 8
            sc += s.rr_ratio * 2   # bonus for better R/R
            return sc

        candidates.sort(key=priority, reverse=True)
        return candidates[0]

    def _run_one(self, strategy_type, candles, of, af, vp, ob, pb,
                  cs, ict, regime, session, instrument,
                  daily_candles, atr):
        try:
            if strategy_type == StrategyType.ICT_CONTINUATION:
                return self._ict_continuation(
                    candles, of, af, vp, ob, pb, cs,
                    ict, regime, session, instrument, atr)
            if strategy_type == StrategyType.ICT_REVERSAL:
                return self._ict_reversal(
                    candles, of, af, vp, ob, pb, cs,
                    ict, regime, session, instrument, atr)
            if strategy_type == StrategyType.AMD_SESSION:
                return self._amd_session(
                    candles, of, af, ict,
                    regime, session, instrument, atr)
            if strategy_type == StrategyType.RETAIL_CONTRARIAN:
                return self._retail_contrarian(
                    candles, of, af, ob, pb,
                    ict, regime, session, instrument, atr)
            if strategy_type == StrategyType.BREAKOUT:
                return self._breakout(
                    candles, of, af,
                    regime, session, instrument, atr)
            if strategy_type == StrategyType.MEAN_REVERSION:
                return self._mean_reversion(
                    candles, of, af, vp,
                    regime, session, instrument, atr)
        except Exception as e:
            log.warning(f"Strategy {strategy_type.value}: {e}")
        return None

    # ── Strategy 1: ICT Trend Continuation (FIXED) ──────────────
    def _ict_continuation(self, candles, of, af, vp, ob, pb,
                           cs, ict, regime, session, instrument, atr):
        struct = ict.get("structure", MarketStructure())
        price  = of.price
        pip    = pip_value(instrument)

        # Require some directional bias
        if struct.trend == "NEUTRAL" and struct.internal_trend == "NEUTRAL":
            return None

        # Use best available trend direction
        direction = None
        if struct.trend == "BULLISH":
            direction = "LONG"
        elif struct.trend == "BEARISH":
            direction = "SHORT"
        elif struct.internal_trend == "BULLISH":
            direction = "LONG"
        elif struct.internal_trend == "BEARISH":
            direction = "SHORT"

        if not direction:
            return None

        # FIX: HTF bias — soft enforcement, not hard rejection
        htf_bias    = struct.htf_bias
        conf_penalty = 0.0

        if htf_bias == "NEUTRAL":
            conf_penalty += 5.0   # small penalty, not rejection
        elif direction == "LONG" and htf_bias == "BEARISH":
            conf_penalty += 15.0  # significant penalty but not rejection
        elif direction == "SHORT" and htf_bias == "BULLISH":
            conf_penalty += 15.0

        # FIX: Regime contradiction penalty (reduced)
        if regime.contradictory:
            conf_penalty += 8.0   # was 15.0

        # FIX: Regime penalty for RANGING (not rejection)
        if regime.regime == RegimeType.RANGING:
            conf_penalty += 8.0

        # ── FIX: Entry zone — three options ──────────────────────
        entry_zone    = None
        in_ote        = False
        in_pullback   = False

        # Option 1: Valid OB with relaxed strength
        for ob_block in ict.get("obs", []):
            if (direction == "LONG" and
                    ob_block.direction == "DEMAND" and
                    ob_block.valid and
                    ob_block.strength >= 20):   # was checking broke_structure
                if ob_block.low <= price <= ob_block.high + pip * 30:
                    entry_zone = ob_block
                    break
            elif (direction == "SHORT" and
                    ob_block.direction == "SUPPLY" and
                    ob_block.valid and
                    ob_block.strength >= 20):
                if ob_block.low - pip * 30 <= price <= ob_block.high:
                    entry_zone = ob_block
                    break

        # Option 2: OTE zone
        ote = (ict.get("ote_long") if direction == "LONG"
               else ict.get("ote_short"))
        if ote and ote[0] > 0 and ote[1] > 0:
            in_ote = min(ote[0], ote[1]) <= price <= max(ote[0], ote[1])

        # Option 3: FIX: Pullback into trend (50% retracement zone)
        if (not entry_zone and not in_ote and
                struct.swing_highs and struct.swing_lows):
            sh = sorted(struct.swing_highs, key=lambda x: x.index)
            sl = sorted(struct.swing_lows,  key=lambda x: x.index)
            if direction == "LONG" and sh and sl:
                last_high = sh[-1]
                last_low  = sl[-1]
                # In a pullback if below 50% of last impulse up
                if last_low.price < last_high.price:
                    pullback_zone_high = (last_low.price +
                        (last_high.price - last_low.price) * 0.65)
                    pullback_zone_low  = (last_low.price +
                        (last_high.price - last_low.price) * 0.25)
                    if pullback_zone_low <= price <= pullback_zone_high:
                        in_pullback = True
            elif direction == "SHORT" and sh and sl:
                last_high = sh[-1]
                last_low  = sl[-1]
                if last_high.price > last_low.price:
                    pb_low  = (last_high.price -
                        (last_high.price - last_low.price) * 0.65)
                    pb_high = (last_high.price -
                        (last_high.price - last_low.price) * 0.25)
                    if pb_low <= price <= pb_high:
                        in_pullback = True

        # FIX: Accept if ANY of the three entry conditions is met
        if not entry_zone and not in_ote and not in_pullback:
            return None

        # Score confluence
        factors, conf, conf_dir = CONFLUENCE.score(
            of, af, vp, ob, pb, cs, ict,
            regime, session, instrument, price)

        conf -= conf_penalty

        # Confidence direction alignment (soft penalty)
        if conf_dir not in ("LONG", "SHORT"):
            conf -= 8.0
        elif (conf_dir == "LONG") != (direction == "LONG"):
            conf -= 10.0

        # ── Compute levels ────────────────────────────────────────
        if direction == "LONG":
            if entry_zone:
                entry = entry_zone.mid
                stop  = entry_zone.low - pip * 3
            elif in_ote and ote:
                entry = price
                stop  = min(ote[0], ote[1]) - pip * 5
            else:
                entry = price
                stop  = price - atr * 1.0

            # Target: next liquidity pool above or ATR projection
            liq    = ict.get("liq_levels", [])
            highs_above = sorted([
                l.price for l in liq
                if l.price > entry + pip * 10
                and not l.swept
                and l.level_type in ("EQUAL_HIGHS","PDH","ASH")])
            if highs_above:
                target = highs_above[0]
            elif vp and vp.vah > entry + pip * 15:
                target = vp.vah
            else:
                target = entry + atr * 2.5

        else:  # SHORT
            if entry_zone:
                entry = entry_zone.mid
                stop  = entry_zone.high + pip * 3
            elif in_ote and ote:
                entry = price
                stop  = max(ote[0], ote[1]) + pip * 5
            else:
                entry = price
                stop  = price + atr * 1.0

            liq    = ict.get("liq_levels", [])
            lows_below = sorted([
                l.price for l in liq
                if l.price < entry - pip * 10
                and not l.swept
                and l.level_type in ("EQUAL_LOWS","PDL","ASL")],
                reverse=True)
            if lows_below:
                target = lows_below[0]
            elif vp and vp.val < entry - pip * 15:
                target = vp.val
            else:
                target = entry - atr * 2.5

        # Pre-validate; if fails try ATR fallback
        val = validate_signal_geometry_fixed(
            direction, entry, target, stop, instrument, atr)
        if not val.is_valid:
            # ATR-based fallback levels
            if direction == "LONG":
                target = entry + atr * 2.5
                stop   = entry - atr * 1.0
            else:
                target = entry - atr * 2.5
                stop   = entry + atr * 1.0
            val = validate_signal_geometry_fixed(
                direction, entry, target, stop, instrument, atr)
            if not val.is_valid:
                return None

        if conf < MIN_CONFIDENCE:
            return None

        # Build reasons
        reasons = []
        if entry_zone:
            reasons.append(
                f"{'Demand' if direction=='LONG' else 'Supply'} OB "
                f"@ {fmt_price(entry_zone.low, instrument)}–"
                f"{fmt_price(entry_zone.high, instrument)}")
        if in_ote:
            reasons.append("Price in OTE zone (62-79% retracement)")
        if in_pullback:
            reasons.append("Pullback into trend continuation zone")
        reasons += [f.description for f in factors
                    if f.direction in (
                        "BULLISH" if direction=="LONG" else "BEARISH",
                        "NEUTRAL") and f.strength > 25][:5]

        return StrategySignal(
            strategy       = StrategyType.ICT_CONTINUATION,
            direction      = direction,
            confidence     = conf,
            entry          = entry,
            target         = target,
            stop           = stop,
            quality        = self._quality(conf, len(factors)),
            reasons        = reasons[:6],
            factors        = factors,
            features       = [],
            regime_aligned = regime.regime in (
                RegimeType.STRONG_TREND, RegimeType.WEAK_TREND,
                RegimeType.EXPANSION),
            session_aligned = (session.is_kill_zone or
                               session.session_type !=
                               SessionType.OFF_HOURS),
            htf_aligned    = htf_bias in (
                "BULLISH" if direction == "LONG" else "BEARISH",
                "NEUTRAL"),
            adr_ok         = regime.adr_consumed_pct < 80,
        )

    # ── Strategy 2: ICT Liquidity Reversal ──────────────────────
    def _ict_reversal(self, candles, of, af, vp, ob, pb,
                       cs, ict, regime, session, instrument, atr):
        swept = ict.get("swept", [])
        disp  = ict.get("displacement")
        fvgs  = ict.get("fvgs", [])
        price = of.price
        pip   = pip_value(instrument)

        if not swept or not disp:
            return None

        direction = "LONG" if disp.is_bullish else "SHORT"

        factors, conf, _ = CONFLUENCE.score(
            of, af, vp, ob, pb, cs, ict,
            regime, session, instrument, price)

        swept_level = swept[-1]
        if direction == "LONG":
            stop = swept_level.price - pip * 5
            bull_fvgs = [f for f in fvgs
                         if f.direction == "BULLISH"
                         and f.mid > price and not f.filled]
            if bull_fvgs:
                target = min(bull_fvgs, key=lambda x: x.mid).mid
            elif vp and vp.vah > price + pip * 15:
                target = vp.vah
            else:
                target = price + atr * 2.5
        else:
            stop = swept_level.price + pip * 5
            bear_fvgs = [f for f in fvgs
                         if f.direction == "BEARISH"
                         and f.mid < price and not f.filled]
            if bear_fvgs:
                target = max(bear_fvgs, key=lambda x: x.mid).mid
            elif vp and vp.val < price - pip * 15:
                target = vp.val
            else:
                target = price - atr * 2.5

        # Validate; fallback to ATR
        val = validate_signal_geometry_fixed(
            direction, price, target, stop, instrument, atr)
        if not val.is_valid:
            if direction == "LONG":
                target = price + atr * 2.5
                stop   = price - atr * 1.0
            else:
                target = price - atr * 2.5
                stop   = price + atr * 1.0
            val = validate_signal_geometry_fixed(
                direction, price, target, stop, instrument, atr)
            if not val.is_valid:
                return None

        conf += 10
        conf  = min(94.0, conf)
        if conf < MIN_CONFIDENCE:
            return None

        return StrategySignal(
            strategy       = StrategyType.ICT_REVERSAL,
            direction      = direction,
            confidence     = conf,
            entry          = price,
            target         = target,
            stop           = stop,
            quality        = self._quality(conf, len(factors)),
            reasons        = (
                [f"Liquidity swept @ "
                 f"{fmt_price(swept_level.price, instrument)}",
                 f"{'Bullish' if disp.is_bullish else 'Bearish'} "
                 f"displacement"] +
                [f.description for f in factors
                 if f.strength > 25][:4]),
            factors        = factors,
            features       = [],
            regime_aligned = True,
            session_aligned = True,
            htf_aligned    = True,
            adr_ok         = regime.adr_consumed_pct < 75,
        )

    # ── Strategy 3: AMD Session Model (FIXED) ───────────────────
    def _amd_session(self, candles, of, af, ict,
                      regime, session, instrument, atr):
        # FIX: Only require Kill Zone — removed AMD phase string check
        if not session.is_kill_zone:
            return None

        asian_h = ict.get("asian_high")
        asian_l = ict.get("asian_low")
        price   = of.price
        pip     = pip_value(instrument)

        if not asian_h or not asian_l:
            return None

        asian_range = asian_h - asian_l
        if asian_range < pip * 5:
            return None

        disp      = ict.get("displacement")
        direction = None
        entry = stop = target = 0.0

        # Check for sweep + displacement pattern
        swept = ict.get("swept", [])

        # FIX: Also check if price is NEAR the Asian boundary
        # not just beyond it — catches the setup as it forms
        near_low  = price < asian_l + pip * 10
        near_high = price > asian_h - pip * 10

        if (disp and disp.is_bullish and
                (price < asian_l + pip * 15 or
                 any(l.level_type == "ASL" for l in swept))):
            direction = "LONG"
            entry     = price
            stop      = min(price, asian_l) - atr * 0.6
            target    = asian_h + asian_range * 0.5

        elif (disp and not disp.is_bullish and
                (price > asian_h - pip * 15 or
                 any(l.level_type == "ASH" for l in swept))):
            direction = "SHORT"
            entry     = price
            stop      = max(price, asian_h) + atr * 0.6
            target    = asian_l - asian_range * 0.5

        if not direction:
            return None

        val = validate_signal_geometry_fixed(
            direction, entry, target, stop, instrument, atr)
        if not val.is_valid:
            # Try wider stop
            if direction == "LONG":
                stop   = entry - atr * 1.0
                target = entry + atr * 2.5
            else:
                stop   = entry + atr * 1.0
                target = entry - atr * 2.5
            val = validate_signal_geometry_fixed(
                direction, entry, target, stop, instrument, atr)
            if not val.is_valid:
                return None

        conf = min(90.0,
                   65.0 +
                   (10 if disp else 0) +
                   (8  if af and af.vpin >= 0.4 else 0) +
                   (7  if swept else 0))
        if conf < MIN_CONFIDENCE:
            return None

        return StrategySignal(
            strategy       = StrategyType.AMD_SESSION,
            direction      = direction,
            confidence     = conf,
            entry          = entry,
            target         = target,
            stop           = stop,
            quality        = SetupQuality.A,
            reasons        = [
                f"AMD Model: {session.kill_zone_name}",
                f"Asian range reference: "
                f"{fmt_price(asian_l, instrument)}–"
                f"{fmt_price(asian_h, instrument)}",
                f"{'Sweep detected + ' if swept else ''}"
                f"Displacement "
                f"{'bullish' if disp and disp.is_bullish else 'bearish'}",
                f"Range: {pips_diff(instrument, asian_range):.0f} pips",
            ],
            factors        = [],
            features       = [],
            regime_aligned = True,
            session_aligned = True,
            htf_aligned    = True,
            adr_ok         = regime.adr_consumed_pct < 70,
        )

    # ── Strategy 4: Retail Contrarian ────────────────────────────
    def _retail_contrarian(self, candles, of, af, ob_data, pb,
                            ict, regime, session, instrument, atr):
        if not pb or abs(pb.skew) < 20:
            return None

        price     = of.price
        direction = None

        if pb.contrarian_signal == "BULLISH" and pb.short_pct > 60:
            direction = "LONG"
        elif pb.contrarian_signal == "BEARISH" and pb.long_pct > 60:
            direction = "SHORT"

        if not direction:
            return None

        swept     = ict.get("swept", [])
        liq_bonus = 15.0 if swept else 0.0
        sm_bonus  = 0.0
        if af:
            if (direction == "LONG" and
                    "BULLISH" in af.informed_signal):
                sm_bonus = 15.0
            elif (direction == "SHORT" and
                    "BEARISH" in af.informed_signal):
                sm_bonus = 15.0

        conf = min(90.0,
                   55.0 + liq_bonus + sm_bonus +
                   (8.0 if abs(pb.skew_change) > 2 else 0.0) +
                   (10.0 if pb.squeeze_potential == "HIGH" else 0.0))

        if direction == "LONG":
            stop   = price - atr * 1.2
            target = (pb.pain_threshold
                      if pb.pain_threshold > price + atr * 1.5
                      else price + atr * 2.0)
        else:
            stop   = price + atr * 1.2
            target = (pb.pain_threshold
                      if pb.pain_threshold < price - atr * 1.5
                      else price - atr * 2.0)

        val = validate_signal_geometry_fixed(
            direction, price, target, stop, instrument, atr)
        if not val.is_valid:
            return None
        if conf < MIN_CONFIDENCE:
            return None

        return StrategySignal(
            strategy       = StrategyType.RETAIL_CONTRARIAN,
            direction      = direction,
            confidence     = conf,
            entry          = price,
            target         = target,
            stop           = stop,
            quality        = self._quality(conf, 4),
            reasons        = [
                f"Retail {pb.long_pct:.0f}% LONG / "
                f"{pb.short_pct:.0f}% SHORT",
                f"Contrarian: {pb.contrarian_signal}",
                f"Squeeze: {pb.squeeze_potential}",
                f"Skew Δ: {pb.skew_change:+.1f}%",
            ],
            factors        = [],
            features       = [],
            regime_aligned = True,
            session_aligned = (
                session.session_type != SessionType.OFF_HOURS),
            htf_aligned    = True,
            adr_ok         = True,
        )

    # ── Strategy 5: Breakout Compression (FIXED) ────────────────
    def _breakout(self, candles, of, af, regime,
                   session, instrument, atr):
        # FIX: raised from 0.7 to 0.8 — more permissive
        if regime.volatility_ratio >= 0.80:
            return None

        price = of.price
        pip   = pip_value(instrument)

        # FIX: Try multiple lookback windows for displacement
        disp = None
        for lookback in [3, 5, 7]:
            disp = ICT_ENGINE.detect_displacement(
                candles, lookback=lookback)
            if disp:
                break

        # FIX: Also accept strong current candle as proxy
        if not disp and len(candles) >= 10:
            avg_body = (sum(c.body_abs for c in candles[-11:-1])
                        / 10)
            if (candles[-1].body_abs >= avg_body * 1.3 and
                    candles[-1].body_ratio >= 0.55):
                disp = candles[-1]

        if not disp:
            return None

        direction = "LONG" if disp.is_bullish else "SHORT"

        if direction == "LONG":
            stop   = price - atr * 1.0
            target = price + atr * 3.0
        else:
            stop   = price + atr * 1.0
            target = price - atr * 3.0

        val = validate_signal_geometry_fixed(
            direction, price, target, stop, instrument, atr)
        if not val.is_valid:
            return None

        conf = min(88.0,
                   60.0 +
                   (10 if af and af.vpin >= 0.35 else 0) +
                   (10 if regime.volatility_ratio < 0.5 else
                    5  if regime.volatility_ratio < 0.65 else 0) +
                   (8  if session.is_kill_zone else 0))

        if conf < MIN_CONFIDENCE:
            return None

        return StrategySignal(
            strategy       = StrategyType.BREAKOUT,
            direction      = direction,
            confidence     = conf,
            entry          = price,
            target         = target,
            stop           = stop,
            quality        = self._quality(conf, 3),
            reasons        = [
                f"Compression breakout "
                f"(VR:{regime.volatility_ratio:.2f})",
                f"{'Bullish' if disp.is_bullish else 'Bearish'} "
                f"displacement ({disp.body_ratio:.0%} body)",
                f"Energy release imminent",
            ],
            factors        = [],
            features       = [],
            regime_aligned = regime.regime in (
                RegimeType.COMPRESSION, RegimeType.RANGING),
            session_aligned = (
                session.session_type != SessionType.OFF_HOURS),
            htf_aligned    = True,
            adr_ok         = regime.adr_consumed_pct < 60,
        )

    # ── Strategy 6: Statistical Mean Reversion (FIXED) ──────────
    def _mean_reversion(self, candles, of, af, vp,
                         regime, session, instrument, atr):
        # FIX: raised threshold from 0.50 to 0.52
        if regime.hurst_exponent >= 0.52:
            return None

        price  = of.price
        closes = [c.close for c in candles]
        bands  = MATH.std_bands(closes, 50)
        if not bands:
            return None

        direction = None

        # FIX: Accept EITHER 2-sigma band breach OR strong VWAP z-score
        if price >= bands["upper_2"]:
            direction = "SHORT"
        elif price <= bands["lower_2"]:
            direction = "LONG"
        elif of.vwap_zscore >= 2.2:
            direction = "SHORT"
        elif of.vwap_zscore <= -2.2:
            direction = "LONG"

        if not direction:
            return None

        # Target: VWAP or 20-period mean (whichever is closer to entry)
        mean_20 = bands["mean"]
        vwap    = of.vwap

        if direction == "LONG":
            stop   = price - atr * 0.8
            # Target closest of VWAP or mean that is above current price
            candidates = []
            if vwap > price + atr * 0.5:
                candidates.append(vwap)
            if mean_20 > price + atr * 0.5:
                candidates.append(mean_20)
            target = (min(candidates) if candidates
                      else price + atr * 1.5)
        else:
            stop   = price + atr * 0.8
            candidates = []
            if vwap < price - atr * 0.5:
                candidates.append(vwap)
            if mean_20 < price - atr * 0.5:
                candidates.append(mean_20)
            target = (max(candidates) if candidates
                      else price - atr * 1.5)

        val = validate_signal_geometry_fixed(
            direction, price, target, stop, instrument, atr)
        if not val.is_valid:
            # Wider target
            if direction == "LONG":
                target = price + atr * 2.0
            else:
                target = price - atr * 2.0
            val = validate_signal_geometry_fixed(
                direction, price, target, stop, instrument, atr)
            if not val.is_valid:
                return None

        conf = min(80.0,
                   55.0 +
                   (10 if regime.autocorr_lag1 < -0.2 else 0) +
                   (10 if abs(of.vwap_zscore) > 2.5 else
                    5  if abs(of.vwap_zscore) > 2.0 else 0) +
                   (8  if price <= bands["lower_2"] or
                          price >= bands["upper_2"] else 0))

        if conf < MIN_CONFIDENCE:
            return None

        return StrategySignal(
            strategy       = StrategyType.MEAN_REVERSION,
            direction      = direction,
            confidence     = conf,
            entry          = price,
            target         = target,
            stop           = stop,
            quality        = self._quality(conf, 3),
            reasons        = [
                f"Price "
                f"{'above +2σ' if price >= bands.get('upper_2',999) else 'below -2σ' if price <= bands.get('lower_2',-999) else 'at VWAP extreme'}"
                f" (z={of.vwap_zscore:+.1f}σ)",
                f"Mean-reverting regime "
                f"(H:{regime.hurst_exponent:.2f})",
                f"Target: "
                f"{fmt_price(target, instrument)} "
                f"({'VWAP' if target == vwap else 'mean'})",
            ],
            factors        = [],
            features       = [],
            regime_aligned = regime.regime == RegimeType.RANGING,
            session_aligned = True,
            htf_aligned    = True,
            adr_ok         = True,
        )

    @staticmethod
    def _quality(conf: float, n: int) -> SetupQuality:
        if conf >= 82 and n >= 6: return SetupQuality.A_PLUS
        if conf >= 72 and n >= 5: return SetupQuality.A
        if conf >= 62 and n >= 4: return SetupQuality.B
        if conf >= 52 and n >= 3: return SetupQuality.C
        return SetupQuality.NO_TRADE


STRATEGY_ENGINE = StrategyEngine()

# ════════════════════════════════════════════════════════════════
#  ML META-LABELING ENGINE  (UNCHANGED — FULLY FUNCTIONAL)
# ════════════════════════════════════════════════════════════════

class MLMetaEngine:
    """
    Meta-labeling: predicts whether the rule-based signal succeeds.
    Persists across restarts. Activates at ML_MIN_SAMPLES = 50.
    """

    def __init__(self):
        self.rf  = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=5, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
        self.gb  = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42)
        self.lr  = LogisticRegression(
            max_iter=3000, C=0.5, random_state=42)
        self.scaler     = StandardScaler()
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

        self.is_trained        = False
        self.is_calibrated     = False
        self.n_samples         = 0
        self.out_of_sample_acc = 0.0
        self.calibrated_brier  = 1.0
        self.feature_importance: Dict[str, float] = {}
        self.accuracy_history:   List[float] = []
        self.weights             = [0.50, 0.35, 0.15]
        self._load()

    def _load(self):
        if os.path.exists(ML_MODEL_FILE):
            try:
                with open(ML_MODEL_FILE, "rb") as f:
                    d = pickle.load(f)
                self.rf                = d.get("rf",    self.rf)
                self.gb                = d.get("gb",    self.gb)
                self.lr                = d.get("lr",    self.lr)
                self.scaler            = d.get("scaler",self.scaler)
                self.calibrator        = d.get("cal",   self.calibrator)
                self.is_trained        = d.get("trained",     False)
                self.is_calibrated     = d.get("calibrated",  False)
                self.n_samples         = d.get("n_samples",   0)
                self.out_of_sample_acc = d.get("oos_acc",     0.0)
                self.feature_importance= d.get("feat_imp",    {})
                self.accuracy_history  = d.get("acc_history", [])
                self.weights           = d.get("weights",
                                              [0.50, 0.35, 0.15])
                self.calibrated_brier  = d.get("brier",       1.0)
                log.info(
                    f"ML loaded: {self.n_samples} samples | "
                    f"Trained: {self.is_trained} | "
                    f"OOS: {self.out_of_sample_acc:.2%}")
            except Exception as e:
                log.error(f"ML load error: {e}")
        else:
            log.info(
                f"ML: No model file. "
                f"Training at {ML_MIN_SAMPLES} resolved samples.")

    def _save(self):
        try:
            with open(ML_MODEL_FILE, "wb") as f:
                pickle.dump({
                    "rf":          self.rf,
                    "gb":          self.gb,
                    "lr":          self.lr,
                    "scaler":      self.scaler,
                    "cal":         self.calibrator,
                    "trained":     self.is_trained,
                    "calibrated":  self.is_calibrated,
                    "n_samples":   self.n_samples,
                    "oos_acc":     self.out_of_sample_acc,
                    "feat_imp":    self.feature_importance,
                    "acc_history": self.accuracy_history,
                    "weights":     self.weights,
                    "brier":       self.calibrated_brier,
                }, f)
        except Exception as e:
            log.error(f"ML save error: {e}")

    def extract_features(self,
                         of:      OrderFlowData,
                         af:      Optional[AdvancedFlowData],
                         vp:      Optional[VolumeProfile],
                         ob:      Optional[OrderBookData],
                         pb:      Optional[PositionBookData],
                         regime:  Optional[RegimeData],
                         session: Optional[SessionData],
                         signal:  Optional[StrategySignal],
                         ict:     Optional[Dict]) -> List[float]:
        tv   = of.total_volume if of.total_volume > 0 else 1.0
        f0   = float(np.clip(of.imbalance_zscore / 3.0, -1.0, 1.0))
        f1   = float(np.clip(of.cvd / tv, -1.0, 1.0))
        f2   = float(np.clip(of.vwap_zscore / 3.0, -1.0, 1.0))
        f3   = float(np.clip(of.vol_ratio / 3.0, 0.0, 1.0))
        f4   = float(of.efficiency_ratio)
        f5   = af.vpin if af else 0.0
        f6   = af.toxicity if af else 0.0
        f7   = float(np.clip(
            (af.absorption_ratio - 1.0) / 2.0, -0.5, 1.0)
            if af else 0.0)
        f8   = (af.institutional_score / 100) if af else 0.0
        f9   = float(np.clip(pb.skew / 100, -1.0, 1.0)) if pb else 0.0
        f10  = float(np.clip(
            pb.skew_change / 10, -1.0, 1.0)) if pb else 0.0
        regime_map = {
            RegimeType.STRONG_TREND: 1.0,
            RegimeType.WEAK_TREND:   0.7,
            RegimeType.RANGING:      0.3,
            RegimeType.COMPRESSION:  0.0,
            RegimeType.EXPANSION:    0.8,
            RegimeType.CHAOTIC:     -1.0,
        }
        f11  = regime_map.get(regime.regime, 0.0) if regime else 0.0
        f12  = float(np.clip(
            (regime.hurst_exponent - 0.5) * 4, -1.0, 1.0)
            if regime else 0.0)
        f13  = float(np.clip(
            regime.volatility_ratio / 3.0, 0.0, 1.0)
            if regime else 0.33)
        f14  = float(np.clip(
            regime.adr_consumed_pct / 100, 0.0, 1.0)
            if regime else 0.5)
        sess_map = {
            SessionType.OVERLAP:   1.0,
            SessionType.LONDON:    0.8,
            SessionType.NEW_YORK:  0.7,
            SessionType.TOKYO:     0.4,
            SessionType.SYDNEY:    0.3,
            SessionType.OFF_HOURS: 0.1,
        }
        f15  = sess_map.get(
            session.session_type, 0.5) if session else 0.5
        f15 += 0.2 if (session and session.is_kill_zone) else 0.0
        f15  = min(1.0, f15)
        f16  = 1.0 if (ict and ict.get("swept")) else 0.0
        disp = ict.get("displacement") if ict else None
        f17  = float(disp.body_ratio if disp else 0.0)
        f18  = float(np.clip(
            signal.rr_ratio / 5.0, 0.0, 1.0) if signal else 0.0)
        f19  = float(signal.confidence / 100 if signal else 0.5)
        raw = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
               f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]
        return [0.0 if (math.isnan(v) or math.isinf(v)) else float(v)
                for v in raw]

    def train(self, training_data: Optional[List[Dict]] = None) -> Dict:
        if training_data is None:
            samples = ML_DATA_STORE.get_training_data()
        else:
            samples = [s for s in training_data
                       if s.get("outcome") in ("WIN","LOSS")
                       and len(s.get("features",[])) == 20]

        self.n_samples = len(samples)

        if self.n_samples < ML_MIN_SAMPLES:
            log.info(
                f"ML: {self.n_samples}/{ML_MIN_SAMPLES} resolved. "
                f"Need {ML_MIN_SAMPLES - self.n_samples} more.")
            return {
                "status":  f"Need {ML_MIN_SAMPLES-self.n_samples} more",
                "samples": self.n_samples,
                "trained": False,
            }

        X = [s["features"] for s in samples]
        y = [1 if s["outcome"]=="WIN" else 0 for s in samples]
        Xa = np.array(X, dtype=np.float32)
        ya = np.array(y, dtype=np.int32)
        Xs = self.scaler.fit_transform(Xa)

        n       = len(Xs)
        fold_n  = n // 5
        buffer  = max(5, fold_n // 10)
        oos_p   = np.zeros(n)
        oos_m   = np.zeros(n, dtype=bool)

        for fold in range(5):
            ts   = fold * fold_n
            te   = ts + fold_n if fold < 4 else n
            tend = max(0, ts - buffer)
            if tend < max(10, ML_MIN_SAMPLES // 4):
                continue
            Xtr, ytr = Xs[:tend], ya[:tend]
            Xte      = Xs[ts:te]
            if len(np.unique(ytr)) < 2:
                continue
            self.rf.fit(Xtr, ytr)
            oos_p[ts:te] = self.rf.predict_proba(Xte)[:,1]
            oos_m[ts:te] = True

        if len(np.unique(ya)) < 2:
            return {"status": "single_class",
                    "samples": n, "trained": False}

        self.rf.fit(Xs, ya)
        self.gb.fit(Xs, ya)
        self.lr.fit(Xs, ya)

        if oos_m.sum() > 0:
            oos_y   = ya[oos_m]
            oos_bin = (oos_p[oos_m] >= 0.5).astype(int)
            self.out_of_sample_acc = float(
                accuracy_score(oos_y, oos_bin))

        if oos_m.sum() >= 20:
            oos_raw = oos_p[oos_m]
            oos_y   = ya[oos_m]
            self.calibrator.fit(oos_raw, oos_y)
            self.is_calibrated = True
            cal_p   = self.calibrator.predict(oos_raw)
            self.calibrated_brier = float(
                brier_score_loss(oos_y, cal_p))

        FNAMES = [
            "of_imb_z","cvd_norm","vwap_z","vol_ratio",
            "efficiency","vpin","toxicity","absorption",
            "inst_score","retail_skew","skew_change","regime",
            "hurst","volatility_ratio","adr_consumed","session_qual",
            "liq_swept","displacement","rr_ratio","raw_conf"
        ]
        if hasattr(self.rf, "feature_importances_"):
            self.feature_importance = {
                n: float(v) for n, v in
                zip(FNAMES, self.rf.feature_importances_)}

        self.is_trained = True
        self.accuracy_history.append(self.out_of_sample_acc)
        self._save()

        log.info(
            f"ML trained: {self.n_samples} samples | "
            f"OOS: {self.out_of_sample_acc:.2%} | "
            f"Brier: {self.calibrated_brier:.3f}")

        return {
            "status":       "trained",
            "samples":      self.n_samples,
            "oos_accuracy": self.out_of_sample_acc,
            "brier":        self.calibrated_brier,
            "calibrated":   self.is_calibrated,
            "trained":      True,
        }

    def predict(self, features: List[float]
                ) -> Tuple[float, float, str]:
        if not self.is_trained or len(features) != 20:
            return 0.5, 0.5, "🧠 ML: Accumulating data..."
        try:
            X   = np.array([features], dtype=np.float32)
            Xs  = self.scaler.transform(X)
            rfp = self.rf.predict_proba(Xs)[0][1]
            gbp = self.gb.predict_proba(Xs)[0][1]
            lrp = self.lr.predict_proba(Xs)[0][1]
            raw = (rfp*self.weights[0] +
                   gbp*self.weights[1] +
                   lrp*self.weights[2])
            cal = (float(self.calibrator.predict([raw])[0])
                   if self.is_calibrated else raw)
            if cal >= 0.75:   lbl = "🧠 ML: VERY HIGH ✅"
            elif cal >= 0.62: lbl = "🧠 ML: HIGH ⚡"
            elif cal >= 0.52: lbl = "🧠 ML: MODERATE"
            elif cal >= 0.40: lbl = "🧠 ML: LOW ⚠️"
            else:              lbl = "🧠 ML: UNFAVORABLE 🔴"
            return float(raw), float(cal), lbl
        except Exception as e:
            log.error(f"ML predict: {e}")
            return 0.5, 0.5, "🧠 ML: Error"

    def get_stats(self) -> Dict:
        n_total    = ML_DATA_STORE.n_total
        n_resolved = ML_DATA_STORE.n_samples
        return {
            "is_trained":      self.is_trained,
            "is_calibrated":   self.is_calibrated,
            "n_samples":       n_resolved,
            "n_total":         n_total,
            "n_needed":        max(0, ML_MIN_SAMPLES - n_resolved),
            "oos_accuracy":    self.out_of_sample_acc,
            "brier_score":     self.calibrated_brier,
            "accuracy_history": self.accuracy_history[-10:],
            "top_features":    sorted(
                self.feature_importance.items(),
                key=lambda x: x[1], reverse=True)[:8],
            "weights":         self.weights,
            "latest_acc":      (self.accuracy_history[-1]
                                if self.accuracy_history else 0.0),
        }


ML_ENGINE = MLMetaEngine()

# ════════════════════════════════════════════════════════════════
#  PATTERN MEMORY ENGINE  (UNCHANGED)
# ════════════════════════════════════════════════════════════════

class PatternMemoryEngine:

    def __init__(self):
        self.patterns: List[Dict] = load_json(PATTERN_MEMORY_FILE, [])
        log.info(f"Pattern memory: {len(self.patterns)} patterns")

    def _save(self):
        save_json(PATTERN_MEMORY_FILE, self.patterns[-800:])

    def record(self,
               pred:    QuantPrediction,
               of:      OrderFlowData,
               af:      Optional[AdvancedFlowData],
               regime:  RegimeData,
               session: SessionData):
        self.patterns.append({
            "id":         pred.prediction_id,
            "trade_id":   pred.trade_id,
            "pair":       pred.pair,
            "timeframe":  pred.timeframe,
            "timestamp":  pred.timestamp,
            "direction":  pred.direction,
            "strategy":   pred.strategy,
            "outcome":    "PENDING",
            "pips":       0.0,
            "confidence": pred.confidence,
            "cal_prob":   pred.calibrated_prob,
            "quality":    pred.quality,
            "session":    session.session_name,
            "is_kz":      session.is_kill_zone,
            "regime":     regime.regime.value,
            "hurst":      regime.hurst_exponent,
            "vpin":       af.vpin if af else 0.0,
            "toxicity":   af.toxicity if af else 0.0,
            "inst_score": af.institutional_score if af else 0.0,
            "sm_activity":af.smart_money_activity if af else "LOW",
            "aggressor":  af.aggressor_side if af else "NEUTRAL",
            "imbalance":  of.imbalance,
            "cvd_pos":    of.cvd > 0,
            "rr_ratio":   pred.rr_ratio,
        })
        self._save()

    def update(self, prediction_id: str, outcome: str, pips: float):
        for p in self.patterns:
            if p.get("id") == prediction_id:
                p["outcome"] = outcome
                p["pips"]    = pips
                break
        self._save()

    def find_similar(self,
                     pair:      str,
                     direction: str,
                     strategy:  str,
                     af:        Optional[AdvancedFlowData],
                     of:        Optional[OrderFlowData],
                     regime:    Optional[RegimeData],
                     session:   Optional[SessionData]) -> Dict:
        completed = [p for p in self.patterns
                     if p.get("outcome") in ("WIN","LOSS")
                     and p.get("pair") == pair]
        if not completed:
            return {"found": False, "count": 0}
        scored = []
        for p in completed:
            score = 0
            if p.get("direction") == direction:         score += 25
            if p.get("strategy")  == strategy:          score += 20
            if session and p.get("session") == session.session_name:
                score += 15
            if p.get("is_kz") and session and session.is_kill_zone:
                score += 10
            if regime and p.get("regime") == regime.regime.value:
                score += 15
            if af and abs(p.get("vpin",0)-af.vpin) < 0.1:
                score += 10
            if af and p.get("aggressor") == af.aggressor_side:
                score += 5
            if of and p.get("cvd_pos",(of.cvd>0)) == (of.cvd>0):
                score += 5
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [p for s,p in scored[:15] if s >= 40]
        if not top:
            return {"found": False, "count": 0}
        wins     = sum(1 for p in top if p.get("outcome")=="WIN")
        losses   = len(top) - wins
        wr       = wins / len(top)
        avg_pips = (sum(abs(p.get("pips",0))
                       for p in top if p.get("outcome")=="WIN")
                    / max(wins, 1))
        return {
            "found":     True,
            "count":     len(top),
            "wins":      wins,
            "losses":    losses,
            "win_rate":  wr,
            "avg_pips":  avg_pips,
            "narrative": self._narrative(
                pair, direction, strategy, top, wr, avg_pips),
        }

    def _narrative(self, pair, direction, strategy,
                   patterns, wr, avg_pips) -> str:
        n  = len(patterns)
        pd = pair_display(pair)
        em = "🔥" if wr >= 0.70 else ("✅" if wr >= 0.55 else "⚠️")
        side = "bulls" if direction=="LONG" else "bears"
        return (
            f"{'─'*30}\n{em} <b>Pattern Memory ({n} similar)</b>\n\n"
            f"Historical: <b>{wr:.0%} WR</b> | "
            f"<b>{avg_pips:.1f} avg pips</b> on winners.\n\n"
            f"{'Strong precedent for ' if wr>=0.60 else 'Mixed — '}"
            f"{side}. "
            f"{'High-probability by history.' if wr>=0.65 else 'Manage risk carefully.'}"
        )


PATTERN_MEMORY = PatternMemoryEngine()

# ════════════════════════════════════════════════════════════════
#  QUANT BRAIN  (UNCHANGED EXCEPT USES FIXED VALIDATOR)
# ════════════════════════════════════════════════════════════════

class QuantBrain:

    def __init__(self):
        self.history: List[Dict] = load_json(PREDICTIONS_FILE, [])
        log.info(f"QuantBrain: {len(self.history)} historical signals")

    def _save(self):
        save_json(PREDICTIONS_FILE, self.history[-2000:])

    async def generate(self,
                       candles:       List[Candle],
                       of:            OrderFlowData,
                       af:            Optional[AdvancedFlowData],
                       vp:            Optional[VolumeProfile],
                       ob:            Optional[OrderBookData],
                       pb:            Optional[PositionBookData],
                       cs:            Optional[Dict],
                       instrument:    str,
                       daily_candles: Optional[List[Candle]] = None,
                       m15_candles:   Optional[List[Candle]] = None,
                       h4_candles:    Optional[List[Candle]] = None
                       ) -> Dict:
        now     = datetime.now(timezone.utc)
        session = SESSION_ENGINE.get_session(now)
        price   = candles[-1].close

        regime  = REGIME_ENGINE.detect(
            candles, daily_candles, instrument)
        ict     = ICT_ENGINE.full_analysis(
            candles, daily_candles, session, instrument)

        # HTF bias from H4
        htf_bias = "NEUTRAL"
        if h4_candles and len(h4_candles) >= 20:
            h4_struct  = ICT_ENGINE.analyze_structure(
                h4_candles, instrument, 3)
            htf_bias   = h4_struct.trend
            ict["structure"].htf_bias = htf_bias

        # Pivots
        pivots = None
        if daily_candles and len(daily_candles) >= 2:
            pd_c   = daily_candles[-2]
            pivots = MATH.calculate_pivots(
                pd_c.high, pd_c.low, pd_c.close)

        closes    = [c.close for c in candles]
        std_bands = MATH.std_bands(closes, 50)
        atr       = MATH.calculate_atr(candles)
        exp_move  = MATH.expected_move(price, atr, 12)

        # Run strategy engine
        signal = STRATEGY_ENGINE.run_all(
            candles, of, af, vp, ob, pb, cs,
            ict, regime, session, instrument, daily_candles)

        if signal is None or signal.quality == SetupQuality.NO_TRADE:
            return {
                "has_setup":  False,
                "regime":     regime,
                "session":    session,
                "ict":        ict,
                "htf_bias":   htf_bias,
                "std_bands":  std_bands,
                "exp_move":   exp_move,
                "pivots":     pivots,
            }

        # Session minimum confidence
        min_conf = SESSION_ENGINE.get_min_confidence(session)
        if signal.confidence < min_conf:
            return {
                "has_setup": False,
                "regime":    regime,
                "session":   session,
                "ict":       ict,
                "htf_bias":  htf_bias,
                "std_bands": std_bands,
                "exp_move":  exp_move,
                "pivots":    pivots,
            }

        # ML features and prediction
        features = ML_ENGINE.extract_features(
            of, af, vp, ob, pb, regime, session, signal, ict)
        signal.features = features

        ml_raw, ml_cal, ml_label = ML_ENGINE.predict(features)
        signal.ml_probability  = ml_raw
        signal.calibrated_prob = ml_cal

        if ML_ENGINE.is_trained and ml_cal < 0.38:
            signal.confidence = max(MIN_CONFIDENCE,
                                    signal.confidence * 0.85)

        # Pattern memory
        sim = PATTERN_MEMORY.find_similar(
            instrument, signal.direction,
            signal.strategy.value, af, of, regime, session)

        # Key levels
        key_levels = self._key_levels(
            ict, vp, pivots, std_bands, exp_move, instrument, price)

        return {
            "has_setup":       True,
            "signal":          signal,
            "direction":       signal.direction,
            "quality":         signal.quality.value,
            "strategy":        signal.strategy.value,
            "target":          signal.target,
            "stop":            signal.stop,
            "confidence":      signal.confidence,
            "calibrated_prob": signal.calibrated_prob,
            "ml_raw":          ml_raw,
            "ml_label":        ml_label,
            "ml_prob":         ml_cal,
            "reasons":         signal.reasons,
            "factors":         signal.factors,
            "features":        features,
            "rr_ratio":        signal.rr_ratio,
            "reward_pips":     signal.reward_pips,
            "risk_pips":       signal.risk_pips,
            "key_levels":      key_levels,
            "regime":          regime,
            "session":         session,
            "ict":             ict,
            "htf_bias":        htf_bias,
            "std_bands":       std_bands,
            "exp_move":        exp_move,
            "pivots":          pivots,
            "sim":             sim,
            "breakeven_price": signal.breakeven_price,
            "trailing_price":  signal.trailing_price,
        }

    def _key_levels(self, ict, vp, pivots, std_bands,
                    exp_move, instrument, price) -> List[Dict]:
        levels = []
        if vp:
            levels += [
                {"price": vp.poc, "type": "POC",
                 "desc": "Point of Control"},
                {"price": vp.vah, "type": "VAH",
                 "desc": "Value Area High"},
                {"price": vp.val, "type": "VAL",
                 "desc": "Value Area Low"},
            ]
        if pivots:
            levels += [
                {"price": pivots.daily_pp, "type": "PP",
                 "desc": "Daily Pivot"},
                {"price": pivots.daily_r1, "type": "R1",
                 "desc": "Daily R1"},
                {"price": pivots.daily_s1, "type": "S1",
                 "desc": "Daily S1"},
                {"price": pivots.camarilla_r4, "type": "CR4",
                 "desc": "Camarilla R4"},
                {"price": pivots.camarilla_s4, "type": "CS4",
                 "desc": "Camarilla S4"},
            ]
        for lv in ict.get("liq_levels", [])[:4]:
            levels.append({
                "price": lv.price, "type": lv.level_type,
                "desc":  ("✅ Swept" if lv.swept else "Unswept")
                         + " liquidity"
            })
        if std_bands:
            levels += [
                {"price": std_bands.get("upper_2", 0),
                 "type": "+2σ", "desc": "2-Sigma upper"},
                {"price": std_bands.get("lower_2", 0),
                 "type": "-2σ", "desc": "2-Sigma lower"},
            ]
        near = [l for l in levels
                if l.get("price", 0) > 0 and
                pips_diff(instrument, abs(l["price"]-price)) < 250]
        near.sort(key=lambda x: abs(x["price"] - price))
        return near[:8]

    def add_to_history(self, pred: QuantPrediction):
        self.history.append({
            "prediction_id":    pred.prediction_id,
            "trade_id":         pred.trade_id,
            "pair":             pred.pair,
            "timeframe":        pred.timeframe,
            "timestamp":        pred.timestamp,
            "current_price":    pred.current_price,
            "direction":        pred.direction,
            "target_price":     pred.target_price,
            "invalidation_price": pred.invalidation_price,
            "confidence":       pred.confidence,
            "calibrated_prob":  pred.calibrated_prob,
            "quality":          pred.quality,
            "strategy":         pred.strategy,
            "reasons":          pred.reasons,
            "key_levels":       pred.key_levels,
            "features":         pred.features,
            "ml_confidence":    pred.ml_confidence,
            "outcome":          "PENDING",
            "pips_gained":      0.0,
            "mae_pips":         0.0,
            "mfe_pips":         0.0,
            "regime":           pred.regime_at_signal,
            "session":          pred.session_at_signal,
            "rr_ratio":         pred.rr_ratio,
            "reward_pips":      pred.reward_pips,
            "risk_pips":        pred.risk_pips,
            "was_sent":         pred.was_sent_to_users,
            "sent_quality":     pred.sent_quality,
        })
        self._save()


QB = QuantBrain()

# ════════════════════════════════════════════════════════════════
#  HISTORICAL BACKTEST ENGINE  (UNCHANGED LOGIC, USES FIXED VAL)
# ════════════════════════════════════════════════════════════════

class HistoricalBacktestEngine:

    def __init__(self):
        self.results: Dict = load_json(HISTORICAL_BT_FILE, {})
        self.is_running    = False
        self._last_run:    Optional[datetime] = None
        log.info(f"Backtest: {len(self.results)} stored results")

    def _save(self):
        save_json(HISTORICAL_BT_FILE, self.results)

    def stale(self) -> bool:
        if not self.results or not self._last_run:
            return True
        return (datetime.now(timezone.utc) -
                self._last_run).total_seconds() > 86400

    async def run(self, pairs=None, tf="H1") -> Dict:
        if self.is_running:
            return {"status": "already_running"}
        self.is_running = True
        pairs_to_test   = pairs or ASSET_CATEGORIES["forex_major"][:5]
        all_results     = {}
        log.info(f"Backtest: {len(pairs_to_test)} pairs on {tf}")
        async with aiohttp.ClientSession() as session:
            for pair in pairs_to_test:
                try:
                    result = await self._backtest_pair(
                        session, pair, tf)
                    if result:
                        key               = f"{pair}_{tf}"
                        all_results[key]  = result
                        self.results[key] = result
                        log.info(
                            f"BT {pair}: "
                            f"{result['win_rate']:.1%} WR | "
                            f"E:{result['expectancy']:+.1f}p")
                    await asyncio.sleep(1.5)
                except Exception as e:
                    log.error(f"BT {pair}: {e}")
        self._save()
        await self._feed_ml(all_results)
        self.is_running = False
        self._last_run  = datetime.now(timezone.utc)
        return all_results

    async def _backtest_pair(self, http_session, pair, tf):
        candles = await fetch_candles(http_session, pair, tf, 500)
        if len(candles) < 80:
            return None
        pip    = pip_value(pair)
        spread = pip * (2 if "JPY" in pair else
                        4 if "XAU" in pair else
                        3 if "NAS" in pair else 1.5)
        signals  = []
        window   = 60
        step     = 8
        se_local = SessionEngine()

        for i in range(window, len(candles)-25, step):
            analysis = candles[max(0, i-window):i]
            if len(analysis) < 30:
                continue
            try:
                of     = OF_ENGINE.analyze(analysis, pair)
                regime = REGIME_ENGINE.detect(analysis)
                sess   = se_local.get_session()
                atr    = MATH.calculate_atr(analysis)

                for st in [StrategyType.ICT_CONTINUATION,
                           StrategyType.ICT_REVERSAL,
                           StrategyType.BREAKOUT,
                           StrategyType.MEAN_REVERSION]:
                    if not REGIME_ENGINE.is_strategy_valid(
                            st, regime, sess):
                        continue
                    sig = self._quick_signal(
                        analysis, of, regime, pair, st, atr)
                    if not sig:
                        continue
                    # Use fixed validator in backtest
                    val = validate_signal_geometry_fixed(
                        sig["direction"], sig["entry"],
                        sig["target"], sig["stop"],
                        pair, atr)
                    if not val.is_valid:
                        continue
                    entry  = sig["entry"] + spread
                    future = candles[i:i+25]
                    outcome, p_pips, mae, mfe = self._simulate(
                        future, entry, sig["target"],
                        sig["stop"], sig["direction"], pair, spread)
                    try:
                        dt = datetime.fromisoformat(
                            analysis[-1].time.replace("Z","+00:00"))
                        sn = se_local.get_session(dt).session_name
                    except Exception:
                        sn = "Unknown"
                    dummy_sig = StrategySignal(
                        strategy=st,
                        direction=sig["direction"],
                        confidence=sig["confidence"],
                        entry=entry,
                        target=sig["target"],
                        stop=sig["stop"],
                        quality=SetupQuality.B,
                        reasons=[], factors=[], features=[],
                        rr_ratio=val.rr_ratio,
                        reward_pips=val.reward_pips,
                        risk_pips=val.risk_pips)
                    feats = ML_ENGINE.extract_features(
                        of, None, None, None, None,
                        regime, sess, dummy_sig, None)
                    signals.append({
                        "pair":       pair,
                        "timeframe":  tf,
                        "direction":  sig["direction"],
                        "strategy":   st.value,
                        "quality":    "B (Moderate)",
                        "confidence": sig["confidence"],
                        "outcome":    outcome,
                        "pips":       p_pips,
                        "mae":        mae,
                        "mfe":        mfe,
                        "session":    sn,
                        "regime":     regime.regime.value,
                        "features":   feats,
                        "rr_ratio":   val.rr_ratio,
                    })
            except Exception:
                continue

        if not signals:
            return None

        wins     = sum(1 for s in signals if s["outcome"]=="WIN")
        losses   = sum(1 for s in signals if s["outcome"]=="LOSS")
        total    = wins + losses
        wr       = wins / total if total > 0 else 0.0
        avg_win  = (sum(abs(s["pips"]) for s in signals
                       if s["outcome"]=="WIN") / max(wins, 1))
        avg_loss = (sum(abs(s["pips"]) for s in signals
                       if s["outcome"]=="LOSS") / max(losses, 1))
        exp      = wr*avg_win - (1-wr)*avg_loss

        session_wr  = self._breakdown(signals, "session")
        strategy_wr = self._breakdown(signals, "strategy")
        regime_wr   = self._breakdown(signals, "regime")

        best_sess  = (max(session_wr.items(),
                         key=lambda x: x[1]["win_rate"])[0]
                     if session_wr else "N/A")
        worst_sess = (min(session_wr.items(),
                         key=lambda x: x[1]["win_rate"])[0]
                     if session_wr else "N/A")

        return {
            "pair":          pair,
            "timeframe":     tf,
            "total_signals": len(signals),
            "wins":          wins,
            "losses":        losses,
            "win_rate":      wr,
            "avg_pips":      avg_win,
            "avg_loss_pips": avg_loss,
            "expectancy":    exp,
            "best_session":  best_sess,
            "worst_session": worst_sess,
            "session_wr":    session_wr,
            "strategy_wr":   strategy_wr,
            "regime_wr":     regime_wr,
            "signals":       signals,
            "last_run":      datetime.now(timezone.utc).isoformat(),
        }

    def _breakdown(self, signals, key) -> Dict:
        out  = {}
        keys = set(s.get(key,"?") for s in signals)
        for k in keys:
            ss = [s for s in signals
                  if s.get(key)==k
                  and s["outcome"] in ("WIN","LOSS")]
            if ss:
                sw = sum(1 for s in ss if s["outcome"]=="WIN")
                out[k] = {"wins":     sw,
                          "total":    len(ss),
                          "win_rate": sw/len(ss)}
        return out

    def _quick_signal(self, candles, of, regime,
                       pair, strategy, atr):
        price = candles[-1].close
        if strategy == StrategyType.ICT_CONTINUATION:
            if of.imbalance_zscore > 1.0 and of.cvd > 0:
                return {"direction": "LONG",
                        "confidence": 58,
                        "entry":  price,
                        "target": price + atr * 2.5,
                        "stop":   price - atr * 1.0}
            if of.imbalance_zscore < -1.0 and of.cvd < 0:
                return {"direction": "SHORT",
                        "confidence": 58,
                        "entry":  price,
                        "target": price - atr * 2.5,
                        "stop":   price + atr * 1.0}
        elif strategy == StrategyType.MEAN_REVERSION:
            # FIX: Use 0.52 threshold in backtest too
            if regime.hurst_exponent < 0.52 and of.vwap_zscore < -2.0:
                return {"direction": "LONG",
                        "confidence": 60,
                        "entry":  price,
                        "target": of.vwap if of.vwap > price + atr else price + atr*1.5,
                        "stop":   price - atr * 0.8}
            if regime.hurst_exponent < 0.52 and of.vwap_zscore > 2.0:
                return {"direction": "SHORT",
                        "confidence": 60,
                        "entry":  price,
                        "target": of.vwap if of.vwap < price - atr else price - atr*1.5,
                        "stop":   price + atr * 0.8}
        elif strategy == StrategyType.BREAKOUT:
            # FIX: Use 0.80 threshold
            if regime.volatility_ratio < 0.80:
                disp = ICT_ENGINE.detect_displacement(candles, 3)
                if not disp and len(candles) >= 10:
                    avg_b = sum(c.body_abs for c in candles[-11:-1])/10
                    if candles[-1].body_abs >= avg_b * 1.3:
                        disp = candles[-1]
                if disp:
                    if disp.is_bullish:
                        return {"direction": "LONG",
                                "confidence": 62,
                                "entry":  price,
                                "target": price + atr * 3.0,
                                "stop":   price - atr * 1.0}
                    else:
                        return {"direction": "SHORT",
                                "confidence": 62,
                                "entry":  price,
                                "target": price - atr * 3.0,
                                "stop":   price + atr * 1.0}
        return None

    def _simulate(self, future, entry, target, stop,
                  direction, pair, spread):
        mae = mfe = 0.0
        pip = pip_value(pair)
        for c in future:
            if direction == "LONG":
                fav  = (c.high - entry) / pip
                adv  = (entry - c.low) / pip
                mfe  = max(mfe, fav); mae = max(mae, adv)
                if c.high >= target:
                    return "WIN",  (target-entry)/pip, mae, mfe
                if c.low  <= stop:
                    return "LOSS", -(entry-stop)/pip, mae, mfe
            else:
                fav  = (entry - c.low) / pip
                adv  = (c.high - entry) / pip
                mfe  = max(mfe, fav); mae = max(mae, adv)
                if c.low  <= target:
                    return "WIN",  (entry-target)/pip, mae, mfe
                if c.high >= stop:
                    return "LOSS", -(stop-entry)/pip, mae, mfe
        last   = future[-1].close if future else entry
        p_pips = ((last-entry) if direction=="LONG"
                  else (entry-last)) / pip
        return ("WIN" if p_pips > 0 else "LOSS"), p_pips, mae, mfe

    async def _feed_ml(self, results: Dict):
        all_signals: List[Dict] = []
        for key, result in results.items():
            if isinstance(result, dict):
                for s in result.get("signals", []):
                    ML_DATA_STORE.add({
                        "prediction_id": f"BT_{key}_{len(all_signals)}",
                        "features":      s.get("features", []),
                        "outcome":       s.get("outcome", ""),
                        "pips_gained":   s.get("pips", 0),
                        "strategy":      s.get("strategy", ""),
                        "session":       s.get("session", ""),
                        "regime":        s.get("regime", ""),
                    })
                    all_signals.append(s)
        if ML_DATA_STORE.n_samples >= ML_MIN_SAMPLES:
            result = ML_ENGINE.train()
            log.info(f"ML trained from BT: {result}")
        else:
            log.info(
                f"BT: {len(all_signals)} signals. "
                f"ML store: {ML_DATA_STORE.n_samples}/"
                f"{ML_MIN_SAMPLES}")

    def format_report(self, live_predictions: List[Dict]) -> str:
        live_done = [p for p in live_predictions
                     if p.get("outcome") in ("WIN","LOSS")]
        lw  = sum(1 for p in live_done if p.get("outcome")=="WIN")
        ll  = len(live_done) - lw
        lt  = lw + ll
        wr  = lw / lt if lt > 0 else 0.0
        avg_win  = (sum(p.get("pips_gained",0)
                       for p in live_done
                       if p.get("outcome")=="WIN") / max(lw,1))
        avg_loss = (sum(abs(p.get("pips_gained",0))
                       for p in live_done
                       if p.get("outcome")=="LOSS") / max(ll,1))
        exp      = wr*avg_win - (1-wr)*avg_loss
        ml_s     = ML_ENGINE.get_stats()
        perf     = ("🔥 ELITE"    if wr >= 0.70 else
                    "✅ STRONG"   if wr >= 0.60 else
                    "⚡ MODERATE" if wr >= 0.50 else
                    "⚠️ BUILDING")

        msg = (
            f"{'='*35}\n📊 <b>QUANT v8.1 PERFORMANCE</b>\n{'='*35}\n\n"
            f"<b>Live Results:</b>\n"
            f"├ Total:      <b>{lt}</b>\n"
            f"├ Wins:       ✅ <b>{lw}</b>\n"
            f"├ Losses:     ❌ <b>{ll}</b>\n"
            f"├ Win Rate:   <b>{wr:.1%}</b>\n"
            f"├ Avg Win:    <b>+{avg_win:.1f} pips</b>\n"
            f"├ Avg Loss:   <b>-{avg_loss:.1f} pips</b>\n"
            f"├ Expectancy: <b>{exp:+.1f} pips/trade</b>\n"
            f"└ Rating:     <b>{perf}</b>\n\n"
        )

        if self.results:
            msg += f"{'─'*35}\n<b>📈 Backtest:</b>\n"
            for key, r in list(self.results.items())[:5]:
                if isinstance(r, dict) and "win_rate" in r:
                    msg += (
                        f"💱 <b>{pair_display(r['pair'])}</b>: "
                        f"{r['win_rate']:.1%} WR | "
                        f"E:{r.get('expectancy',0):+.1f}p\n")
            msg += "\n"

        msg += f"{'─'*35}\n<b>🧠 ML Engine:</b>\n"
        if ml_s["is_trained"]:
            msg += (
                f"├ Status:    ✅ Active\n"
                f"├ Resolved:  {ml_s['n_samples']}\n"
                f"├ Total:     {ml_s['n_total']}\n"
                f"├ OOS Acc:   <b>{ml_s['oos_accuracy']:.1%}</b>\n"
                f"├ Brier:     {ml_s['brier_score']:.3f}\n"
                f"└ Calib:     {'✅' if ml_s['is_calibrated'] else '❌'}\n\n"
            )
        else:
            msg += (
                f"├ Status:    ⏳ Accumulating\n"
                f"├ Resolved:  {ml_s['n_samples']}/{ML_MIN_SAMPLES}\n"
                f"├ Total:     {ml_s['n_total']} tracked\n"
                f"└ Need:      {ml_s['n_needed']} more\n\n"
            )

        msg += (
            f"{'─'*35}\n"
            f"<i>⚠️ Stop-loss every trade. Max 1-2% risk.</i>\n"
            f"{'='*35}"
        )
        return msg

    def get_pair_result(self,
                        pair: str, tf: str = "H1") -> Optional[Dict]:
        return self.results.get(f"{pair}_{tf}")


HISTORICAL_BT = HistoricalBacktestEngine()

# ════════════════════════════════════════════════════════════════
#  PREDICTION TRACKER  (UNCHANGED — FULLY FUNCTIONAL)
# ════════════════════════════════════════════════════════════════

class PredictionTracker:

    def __init__(self):
        self.active: List[QuantPrediction] = []
        self._load()

    def _load(self):
        if not os.path.exists(ACTIVE_PREDICTIONS_FILE):
            return
        try:
            raw = load_json(ACTIVE_PREDICTIONS_FILE, [])
            self.active = []
            for p in raw:
                if p.get("status") == "ACTIVE":
                    try:
                        p.setdefault("trade_id", generate_trade_id(
                            p.get("pair","UNKNOWN"),
                            p.get("direction","LONG")))
                        p.setdefault("calibrated_prob", 0.5)
                        p.setdefault("strategy", "Unknown")
                        p.setdefault("chat_ids",
                            [p.get("chat_id")]
                            if p.get("chat_id") else [])
                        p.setdefault("rr_ratio", 0.0)
                        p.setdefault("reward_pips", 0.0)
                        p.setdefault("risk_pips", 0.0)
                        p.setdefault("breakeven_price", 0.0)
                        p.setdefault("trailing_price", 0.0)
                        p.setdefault("mae_pips", 0.0)
                        p.setdefault("mfe_pips", 0.0)
                        p.setdefault("regime_at_signal", "")
                        p.setdefault("session_at_signal", "")
                        p.setdefault("breakeven_notified", False)
                        p.setdefault("trailing_notified", False)
                        p.setdefault("was_sent_to_users", True)
                        p.setdefault("sent_quality",
                                     p.get("quality",""))
                        p.setdefault("key_levels", [])
                        p.pop("chat_id", None)
                        self.active.append(QuantPrediction(**p))
                    except Exception as e:
                        log.warning(f"Skip pred on load: {e}")
            log.info(f"Active predictions: {len(self.active)}")
        except Exception as e:
            log.error(f"Load predictions: {e}")

    def _save(self):
        data = []
        for p in self.active:
            data.append({
                "trade_id":          p.trade_id,
                "prediction_id":     p.prediction_id,
                "pair":              p.pair,
                "timeframe":         p.timeframe,
                "timestamp":         p.timestamp,
                "current_price":     p.current_price,
                "direction":         p.direction,
                "target_price":      p.target_price,
                "invalidation_price":p.invalidation_price,
                "breakeven_price":   p.breakeven_price,
                "trailing_price":    p.trailing_price,
                "confidence":        p.confidence,
                "calibrated_prob":   p.calibrated_prob,
                "quality":           p.quality,
                "strategy":          p.strategy,
                "reasons":           p.reasons,
                "key_levels":        p.key_levels,
                "factors_aligned":   p.factors_aligned,
                "features":          p.features,
                "rr_ratio":          p.rr_ratio,
                "reward_pips":       p.reward_pips,
                "risk_pips":         p.risk_pips,
                "status":            p.status,
                "outcome":           p.outcome,
                "hit_time":          p.hit_time,
                "chat_ids":          p.chat_ids,
                "ml_confidence":     p.ml_confidence,
                "ml_used":           p.ml_used,
                "pips_gained":       p.pips_gained,
                "mae_pips":          p.mae_pips,
                "mfe_pips":          p.mfe_pips,
                "regime_at_signal":  p.regime_at_signal,
                "session_at_signal": p.session_at_signal,
                "breakeven_notified":p.breakeven_notified,
                "trailing_notified": p.trailing_notified,
                "was_sent_to_users": p.was_sent_to_users,
                "sent_quality":      p.sent_quality,
            })
        save_json(ACTIVE_PREDICTIONS_FILE, data)

    def add(self, pred: QuantPrediction):
        self.active.append(pred)
        self._save()
        log.info(
            f"Tracking [{pred.trade_id}] "
            f"{pred.pair} {pred.direction} | "
            f"Sent:{pred.was_sent_to_users}")

    async def check(self, bot) -> List[Dict]:
        notifications = []
        if not self.active:
            return notifications

        async with aiohttp.ClientSession() as http_session:
            for pred in self.active[:]:
                try:
                    cp = await fetch_current_price(
                        http_session, pred.pair)
                    if cp is None:
                        continue

                    pip      = pip_value(pred.pair)
                    resolved = False

                    # MAE/MFE update
                    if pred.direction == "LONG":
                        fav = (cp - pred.current_price) / pip
                        adv = (pred.current_price - cp) / pip
                    else:
                        fav = (pred.current_price - cp) / pip
                        adv = (cp - pred.current_price) / pip
                    pred.mfe_pips = max(pred.mfe_pips, fav)
                    pred.mae_pips = max(pred.mae_pips, adv)

                    # Breakeven alert
                    if (pred.was_sent_to_users and
                            not pred.breakeven_notified and
                            pred.breakeven_price > 0):
                        be_hit = (
                            (pred.direction == "LONG" and
                             cp >= pred.breakeven_price) or
                            (pred.direction == "SHORT" and
                             cp <= pred.breakeven_price))
                        if be_hit:
                            pred.breakeven_notified = True
                            notifications.append({
                                "pred": pred,
                                "type": "BREAKEVEN",
                                "cp":   cp})

                    # Trailing alert
                    if (pred.was_sent_to_users and
                            not pred.trailing_notified and
                            pred.trailing_price > 0):
                        trail_hit = (
                            (pred.direction == "LONG" and
                             cp >= pred.trailing_price) or
                            (pred.direction == "SHORT" and
                             cp <= pred.trailing_price))
                        if trail_hit:
                            pred.trailing_notified = True
                            notifications.append({
                                "pred": pred,
                                "type": "TRAILING",
                                "cp":   cp})

                    # Target / Stop
                    if pred.direction == "LONG":
                        if cp >= pred.target_price:
                            pred.status      = "TARGET_HIT"
                            pred.outcome     = "WIN"
                            pred.pips_gained = (
                                (pred.target_price -
                                 pred.current_price) / pip)
                            if pred.was_sent_to_users:
                                notifications.append({
                                    "pred": pred,
                                    "type": "TARGET_HIT",
                                    "cp":   cp})
                            resolved = True
                        elif cp <= pred.invalidation_price:
                            pred.status      = "STOP_HIT"
                            pred.outcome     = "LOSS"
                            pred.pips_gained = -(
                                (pred.current_price -
                                 pred.invalidation_price) / pip)
                            if pred.was_sent_to_users:
                                notifications.append({
                                    "pred": pred,
                                    "type": "STOP_HIT",
                                    "cp":   cp})
                            resolved = True
                    else:
                        if cp <= pred.target_price:
                            pred.status      = "TARGET_HIT"
                            pred.outcome     = "WIN"
                            pred.pips_gained = (
                                (pred.current_price -
                                 pred.target_price) / pip)
                            if pred.was_sent_to_users:
                                notifications.append({
                                    "pred": pred,
                                    "type": "TARGET_HIT",
                                    "cp":   cp})
                            resolved = True
                        elif cp >= pred.invalidation_price:
                            pred.status      = "STOP_HIT"
                            pred.outcome     = "LOSS"
                            pred.pips_gained = -(
                                (pred.invalidation_price -
                                 pred.current_price) / pip)
                            if pred.was_sent_to_users:
                                notifications.append({
                                    "pred": pred,
                                    "type": "STOP_HIT",
                                    "cp":   cp})
                            resolved = True

                    # 24h expiry
                    if not resolved:
                        try:
                            pt = datetime.fromisoformat(
                                pred.timestamp.replace("Z","+00:00"))
                            if (datetime.now(timezone.utc) - pt >
                                    timedelta(hours=24)):
                                pred.status  = "EXPIRED"
                                pred.outcome = "EXPIRED"
                                if pred.was_sent_to_users:
                                    notifications.append({
                                        "pred": pred,
                                        "type": "EXPIRED",
                                        "cp":   cp})
                                resolved = True
                        except Exception:
                            pass

                    if resolved:
                        pred.hit_time = (
                            datetime.now(timezone.utc).isoformat())
                        self.active.remove(pred)

                    await asyncio.sleep(0.1)

                except Exception as e:
                    log.error(f"Check {pred.trade_id}: {e}")

        if notifications:
            self._save()
            for n in notifications:
                p = n["pred"]
                if n["type"] in ("TARGET_HIT","STOP_HIT","EXPIRED"):
                    for h in QB.history:
                        if h.get("prediction_id") == p.prediction_id:
                            h["outcome"]     = p.outcome or "EXPIRED"
                            h["pips_gained"] = p.pips_gained
                            h["mae_pips"]    = p.mae_pips
                            h["mfe_pips"]    = p.mfe_pips
                            break
                    QB._save()
                    ML_DATA_STORE.update_outcome(
                        p.prediction_id,
                        p.outcome or "EXPIRED",
                        p.pips_gained)
                    PATTERN_MEMORY.update(
                        p.prediction_id,
                        p.outcome or "EXPIRED",
                        p.pips_gained)
                    if ML_DATA_STORE.n_samples >= ML_MIN_SAMPLES:
                        asyncio.create_task(self._async_retrain())

        return notifications

    async def _async_retrain(self):
        try:
            result = ML_ENGINE.train()
            log.info(f"ML auto-retrain: {result.get('status')}")
        except Exception as e:
            log.error(f"ML retrain: {e}")


PT = PredictionTracker()

# ════════════════════════════════════════════════════════════════
#  FULL ANALYSIS PIPELINE  (UNCHANGED)
# ════════════════════════════════════════════════════════════════

async def full_analysis(http_session: aiohttp.ClientSession,
                        pair:         str,
                        tf:           str) -> Dict:
    tf_cfg = TIMEFRAMES.get(tf, TIMEFRAMES["H1"])

    results = await asyncio.gather(
        fetch_candles(http_session, pair, tf,  tf_cfg["candles"]),
        fetch_candles(http_session, pair, "D",  60),
        fetch_candles(http_session, pair, "H4", 120),
        fetch_candles(http_session, pair, "M15", 96),
        fetch_order_book(http_session, pair),
        fetch_position_book(http_session, pair),
        return_exceptions=True
    )

    candles       = (results[0] if not isinstance(results[0], Exception)
                     else [])
    daily_candles = (results[1] if not isinstance(results[1], Exception)
                     else [])
    h4_candles    = (results[2] if not isinstance(results[2], Exception)
                     else [])
    m15_candles   = (results[3] if not isinstance(results[3], Exception)
                     else [])
    ob_raw        = (results[4] if not isinstance(results[4], Exception)
                     else None)
    pb_raw        = (results[5] if not isinstance(results[5], Exception)
                     else None)

    if not candles or len(candles) < 30:
        raise ValueError(
            f"Insufficient candle data for {pair} {tf}")

    of  = OF_ENGINE.analyze(candles, pair)
    vp  = calculate_volume_profile(candles)
    ob  = analyze_order_book(ob_raw, of.price)
    pb  = analyze_position_book(pb_raw, of.price, pair)
    af  = ADVANCED_FLOW.analyze(candles, ob_raw)

    cs = None
    if pair in FOREX_PAIRS:
        try:
            cs = await STRENGTH_CACHE.get(http_session)
        except Exception:
            pass

    return {
        "candles":       candles,
        "daily_candles": daily_candles,
        "h4_candles":    h4_candles,
        "m15_candles":   m15_candles,
        "of":            of,
        "vp":            vp,
        "ob":            ob,
        "pb":            pb,
        "af":            af,
        "cs":            cs,
        "ob_raw":        ob_raw,
        "pb_raw":        pb_raw,
    }

# ════════════════════════════════════════════════════════════════
#  END OF PART 2 (FIXED)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  FOREX QUANT v8.1 — PART 3 OF 4
#  Performance Engine, Subscription Manager, Session Advisor,
#  Message Formatters, Chart Functions, Alert Engine,
#  Telegram Command Handlers, Callback Handler
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  PERFORMANCE INTELLIGENCE ENGINE
# ════════════════════════════════════════════════════════════════

class PerformanceEngine:
    """
    Tracks expectancy, MAE/MFE, regime/session/strategy breakdown,
    drawdown, Kelly criterion, adaptive risk recommendations.
    """

    def __init__(self):
        self.data: Dict = load_json(PERFORMANCE_FILE, {
            "trades":             [],
            "consecutive_losses": 0,
            "consecutive_wins":   0,
            "peak_equity":        0.0,
            "current_equity":     0.0,
            "max_drawdown":       0.0,
        })
        log.info(
            f"Performance engine: "
            f"{len(self.data.get('trades',[]))} trades")

    def _save(self):
        save_json(PERFORMANCE_FILE, self.data)

    def record(self, pred: QuantPrediction):
        """Record resolved prediction in performance engine."""
        if pred.outcome not in ("WIN", "LOSS"):
            return
        trade = {
            "id":        pred.prediction_id,
            "trade_id":  pred.trade_id,
            "pair":      pred.pair,
            "strategy":  pred.strategy,
            "direction": pred.direction,
            "outcome":   pred.outcome,
            "pips":      pred.pips_gained,
            "mae":       pred.mae_pips,
            "mfe":       pred.mfe_pips,
            "rr_ratio":  pred.rr_ratio,
            "regime":    pred.regime_at_signal,
            "session":   pred.session_at_signal,
            "quality":   pred.quality,
            "conf":      pred.confidence,
            "cal_prob":  pred.calibrated_prob,
            "time":      pred.hit_time or pred.timestamp,
        }
        self.data["trades"].append(trade)

        # Consecutive tracking
        if pred.outcome == "WIN":
            self.data["consecutive_wins"]   += 1
            self.data["consecutive_losses"]  = 0
        else:
            self.data["consecutive_losses"] += 1
            self.data["consecutive_wins"]    = 0

        # Equity curve (pip-based simulation)
        self.data["current_equity"] = (
            self.data.get("current_equity", 0.0) + pred.pips_gained)
        peak = self.data.get("peak_equity", 0.0)
        if self.data["current_equity"] > peak:
            self.data["peak_equity"] = self.data["current_equity"]
            peak = self.data["current_equity"]
        dd = peak - self.data["current_equity"]
        if dd > self.data.get("max_drawdown", 0.0):
            self.data["max_drawdown"] = dd

        self._save()

    def get_risk_mode(self) -> Tuple[str, float, str]:
        cl = self.data.get("consecutive_losses", 0)
        cw = self.data.get("consecutive_wins",   0)
        if cl >= MAX_CONSECUTIVE_LOSSES:
            return ("RECOVERY", REDUCED_RISK_PCT,
                    f"⚠️ {cl} consecutive losses — "
                    f"{REDUCED_RISK_PCT}% risk. A+ quality only.")
        if cl >= 3:
            return ("CAUTION", REDUCED_RISK_PCT,
                    f"⚠️ {cl} consecutive losses — risk reduced to "
                    f"{REDUCED_RISK_PCT}%.")
        if cw >= 5:
            return ("HOT_STREAK", DEFAULT_RISK_PCT,
                    f"🔥 {cw} wins in a row — stay disciplined!")
        return ("NORMAL", DEFAULT_RISK_PCT, "✅ Normal risk mode.")

    def get_expectancy(self,
                       filter_key:   Optional[str] = None,
                       filter_value: Optional[str] = None) -> Dict:
        trades = self.data.get("trades", [])
        if filter_key and filter_value:
            trades = [t for t in trades
                      if t.get(filter_key) == filter_value]
        wins   = [t for t in trades if t.get("outcome") == "WIN"]
        losses = [t for t in trades if t.get("outcome") == "LOSS"]
        total  = len(wins) + len(losses)
        if total == 0:
            return {"total": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "expectancy": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0,
                    "avg_mae": 0.0, "avg_mfe": 0.0}
        wr       = len(wins) / total
        avg_win  = (sum(t["pips"] for t in wins) / len(wins)
                    if wins else 0.0)
        avg_loss = (sum(abs(t["pips"]) for t in losses) / len(losses)
                    if losses else 0.0)
        exp      = wr * avg_win - (1 - wr) * avg_loss
        all_t    = wins + losses
        avg_mae  = sum(t.get("mae", 0) for t in all_t) / total
        avg_mfe  = sum(t.get("mfe", 0) for t in all_t) / total
        return {
            "total":      total,
            "wins":       len(wins),
            "losses":     len(losses),
            "win_rate":   wr,
            "expectancy": exp,
            "avg_win":    avg_win,
            "avg_loss":   avg_loss,
            "avg_mae":    avg_mae,
            "avg_mfe":    avg_mfe,
        }

    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        out = {}
        for st in StrategyType:
            e = self.get_expectancy("strategy", st.value)
            if e["total"] > 0:
                out[st.value] = e
        return out

    def get_regime_breakdown(self) -> Dict[str, Dict]:
        out = {}
        for rt in RegimeType:
            e = self.get_expectancy("regime", rt.value)
            if e["total"] > 0:
                out[rt.value] = e
        return out

    def get_session_breakdown(self) -> Dict[str, Dict]:
        sessions = [
            "London", "New York", "Tokyo / Asian",
            "London-NY Overlap", "Off Hours", "Sydney"
        ]
        out = {}
        for sn in sessions:
            e = self.get_expectancy("session", sn)
            if e["total"] > 0:
                out[sn] = e
        return out

    def kelly_recommendation(self) -> float:
        exp = self.get_expectancy()
        if exp["total"] < 30:
            return DEFAULT_RISK_PCT
        return max(0.25, min(2.0,
            MATH.kelly_fraction(
                exp["win_rate"],
                exp["avg_win"],
                exp["avg_loss"]) * 100))

    def format_report(self) -> str:
        exp   = self.get_expectancy()
        mode, risk_pct, risk_msg = self.get_risk_mode()
        kelly = self.kelly_recommendation()
        strat = self.get_strategy_breakdown()
        sess  = self.get_session_breakdown()
        regime_b = self.get_regime_breakdown()

        msg = (
            f"{'='*35}\n"
            f"📊 <b>PERFORMANCE INTELLIGENCE</b>\n"
            f"{'='*35}\n\n"
            f"<b>Overall ({exp['total']} trades):</b>\n"
            f"├ Win Rate:   <b>{exp['win_rate']:.1%}</b>\n"
            f"├ Avg Win:    <b>+{exp['avg_win']:.1f} pips</b>\n"
            f"├ Avg Loss:   <b>-{exp['avg_loss']:.1f} pips</b>\n"
            f"├ Expectancy: <b>{exp['expectancy']:+.1f} pips/trade</b>\n"
            f"├ Avg MAE:    {exp['avg_mae']:.1f}p (max adverse)\n"
            f"├ Avg MFE:    {exp['avg_mfe']:.1f}p (max favorable)\n"
            f"└ Max DD:     "
            f"{self.data.get('max_drawdown', 0):.1f} pips\n\n"
            f"<b>Risk:</b> {risk_msg}\n"
            f"<b>Kelly:</b> {kelly:.2f}% per trade\n\n"
        )

        if strat:
            msg += f"{'─'*35}\n<b>By Strategy:</b>\n"
            for s, e in sorted(
                    strat.items(),
                    key=lambda x: x[1]["expectancy"],
                    reverse=True)[:5]:
                msg += (
                    f"• {s[:24]}: "
                    f"{e['win_rate']:.0%} WR | "
                    f"{e['expectancy']:+.1f}p "
                    f"({e['total']})\n")
            msg += "\n"

        if sess:
            msg += f"{'─'*35}\n<b>By Session:</b>\n"
            for sn, e in sorted(
                    sess.items(),
                    key=lambda x: x[1]["expectancy"],
                    reverse=True):
                msg += (
                    f"• {sn}: "
                    f"{e['win_rate']:.0%} WR | "
                    f"{e['expectancy']:+.1f}p\n")
            msg += "\n"

        if regime_b:
            msg += f"{'─'*35}\n<b>By Regime:</b>\n"
            for rn, e in sorted(
                    regime_b.items(),
                    key=lambda x: x[1]["expectancy"],
                    reverse=True):
                msg += (
                    f"• {rn}: "
                    f"{e['win_rate']:.0%} WR | "
                    f"{e['expectancy']:+.1f}p\n")

        return msg


PERF_ENGINE = PerformanceEngine()

# ════════════════════════════════════════════════════════════════
#  SUBSCRIPTION MANAGER
# ════════════════════════════════════════════════════════════════

class SubscriptionManager:

    def __init__(self):
        raw = load_json(USER_SUBSCRIPTIONS_FILE, {})
        self.subs: Dict[int, Dict] = {}
        for k, v in raw.items():
            try:
                self.subs[int(k)] = v
            except Exception:
                pass
        log.info(f"Subscriptions: {len(self.subs)} users")

    def _save(self):
        save_json(USER_SUBSCRIPTIONS_FILE,
                  {str(k): v for k, v in self.subs.items()})

    def subscribe(self, chat_id: int,
                  pairs:     List[str] = None,
                  timeframe: str = "H1"):
        self.subs[chat_id] = {
            "pairs":         pairs or [],
            "timeframe":     timeframe,
            "active":        True,
            "all_pairs":     len(pairs or []) == 0,
            "min_quality":   "B",
            "min_conf":      MIN_ALERT_CONFIDENCE,
            "subscribed_at": datetime.now(timezone.utc).isoformat(),
            "last_alert":    None,
            "alert_count":   0,
        }
        self._save()

    def unsubscribe(self, chat_id: int):
        if chat_id in self.subs:
            self.subs[chat_id]["active"] = False
            self._save()

    def is_active(self, chat_id: int) -> bool:
        s = self.subs.get(chat_id)
        return s is not None and s.get("active", False)

    def get(self, chat_id: int) -> Optional[Dict]:
        return self.subs.get(chat_id)

    def all_active(self) -> List[Tuple[int, Dict]]:
        return [(cid, s) for cid, s in self.subs.items()
                if s.get("active", False)]

    def add_pair(self, chat_id: int, pair: str):
        if chat_id in self.subs:
            pairs = self.subs[chat_id].get("pairs", [])
            if pair not in pairs:
                pairs.append(pair)
            self.subs[chat_id]["pairs"]     = pairs
            self.subs[chat_id]["all_pairs"] = False
            self._save()

    def remove_pair(self, chat_id: int, pair: str):
        if chat_id in self.subs:
            pairs = self.subs[chat_id].get("pairs", [])
            if pair in pairs:
                pairs.remove(pair)
            self.subs[chat_id]["pairs"] = pairs
            self._save()

    def set_all_pairs(self, chat_id: int, enabled: bool):
        if chat_id in self.subs:
            self.subs[chat_id]["all_pairs"] = enabled
            self._save()

    def set_tf(self, chat_id: int, tf: str):
        if chat_id in self.subs:
            self.subs[chat_id]["timeframe"] = tf
            self._save()

    def set_min_quality(self, chat_id: int, quality: str):
        if chat_id in self.subs:
            self.subs[chat_id]["min_quality"] = quality
            self._save()

    def record_alert(self, chat_id: int):
        if chat_id in self.subs:
            self.subs[chat_id]["last_alert"] = (
                datetime.now(timezone.utc).isoformat())
            self.subs[chat_id]["alert_count"] = (
                self.subs[chat_id].get("alert_count", 0) + 1)
            self._save()

    def wants_pair(self, chat_id: int, pair: str) -> bool:
        s = self.subs.get(chat_id)
        if not s or not s.get("active"):
            return False
        if s.get("all_pairs"):
            return True
        return pair in s.get("pairs", [])

    def quality_rank(self, quality_str: str) -> int:
        """Returns numeric rank: A+=4, A=3, B=2, C=1."""
        if "A+" in quality_str: return 4
        if quality_str.startswith("A"): return 3
        if "B" in quality_str: return 2
        return 1

    def meets_quality(self, chat_id: int,
                      quality_str: str) -> bool:
        """Check if signal quality meets subscriber's minimum."""
        s = self.subs.get(chat_id)
        if not s:
            return False
        min_q    = s.get("min_quality", "B")
        min_rank = self.quality_rank(min_q)
        sig_rank = self.quality_rank(quality_str)
        return sig_rank >= min_rank


SUB_MANAGER = SubscriptionManager()

# ════════════════════════════════════════════════════════════════
#  SESSION ADVISOR
# ════════════════════════════════════════════════════════════════

class SessionAdvisor:

    BEST_PAIRS = {
        "London":            ["EUR_USD","GBP_USD","EUR_GBP",
                              "GBP_JPY","XAU_USD"],
        "New York":          ["EUR_USD","GBP_USD","USD_CAD",
                              "USD_JPY","XAU_USD"],
        "London-NY Overlap": ["EUR_USD","GBP_USD","XAU_USD",
                              "EUR_GBP","GBP_JPY"],
        "Tokyo / Asian":     ["USD_JPY","AUD_JPY","EUR_JPY",
                              "GBP_JPY","AUD_USD"],
        "Sydney":            ["AUD_USD","NZD_USD","AUD_JPY",
                              "NZD_JPY","AUD_NZD"],
        "Off Hours":         [],
    }

    def advice(self, pair: str,
               session: Optional[SessionData] = None) -> str:
        if session is None:
            session = SESSION_ENGINE.get_session()
        sn   = session.session_name
        best = self.BEST_PAIRS.get(sn, [])
        wat  = session.wat_time

        if session.session_type == SessionType.OFF_HOURS:
            quality = "⚠️ OFF-HOURS"
            advice  = ("Low liquidity, wide spreads. "
                       "Signals still fire but require higher "
                       "confluence. Reduce position size by 50%.")
        elif session.is_kill_zone:
            quality = "🎯 PRIME TIME"
            advice  = (f"{session.kill_zone_name} — "
                       "highest-probability entry window. "
                       "Institutions most active here.")
        elif session.session_type == SessionType.OVERLAP:
            quality = "⚡ MAX LIQUIDITY"
            advice  = ("London-NY Overlap: tightest spreads, "
                       "biggest institutional moves.")
        elif session.session_type == SessionType.LONDON:
            quality = "🏦 HIGH QUALITY"
            advice  = ("London sets the tone. Most institutional "
                       "order flow here. EUR/GBP pairs optimal.")
        elif session.session_type == SessionType.NEW_YORK:
            quality = "🗽 GOOD QUALITY"
            advice  = ("NY adds USD momentum. Best in first 3h. "
                       "Watch for London reversal setups.")
        elif session.session_type == SessionType.TOKYO:
            quality = "🗼 MODERATE"
            advice  = ("Tokyo: lower volatility, JPY pairs best. "
                       "Range-bound. AMD accumulation phase.")
        else:
            quality = "📊 STANDARD"
            advice  = "Standard conditions."

        msg = (
            f"{'─'*35}\n"
            f"<b>🕐 {wat}</b>\n"
            f"<b>Session:</b> {sn} — {quality}\n"
            f"{'─'*35}\n\n"
            f"{advice}\n\n"
            f"AMD Phase: <b>{session.amd_phase}</b>\n"
        )
        if session.is_dst_london or session.is_dst_ny:
            msg += (
                f"<i>DST: "
                f"{'London BST' if session.is_dst_london else ''}"
                f"{' + ' if session.is_dst_london and session.is_dst_ny else ''}"
                f"{'NY EDT' if session.is_dst_ny else ''}</i>\n")
        if pair in best:
            msg += (f"\n✅ <b>{pair_display(pair)} is OPTIMAL "
                    f"for this session.</b>\n")
        elif sn == "Off Hours":
            msg += (f"\n⚠️ <b>Off-hours — reduced quality for "
                    f"{pair_display(pair)}.</b>\n")
        else:
            msg += (f"\n💡 {pair_display(pair)} tradeable. "
                    f"Best now: "
                    f"{', '.join(pair_display(p) for p in best[:3]) or 'N/A'}\n")
        if session.hours_remaining > 0:
            msg += (f"\n<i>Session closes in "
                    f"{session.hours_remaining:.1f}h</i>\n")
        return msg

    def schedule(self) -> str:
        now     = datetime.now(timezone.utc)
        session = SESSION_ENGINE.get_session(now)
        wat     = SESSION_ENGINE.wat_time(now)
        utc_t   = SESSION_ENGINE.utc_time(now)

        msg = (
            f"{'='*35}\n"
            f"🗓️ <b>TRADING SESSION GUIDE</b>\n"
            f"{'='*35}\n\n"
            f"<b>Now:</b> {wat} | {utc_t}\n"
            f"<b>Current:</b> {session.session_name} "
            f"{'| 🎯 ' + session.kill_zone_name if session.is_kill_zone else ''}\n"
            f"<b>AMD Phase:</b> {session.amd_phase}\n\n"
        )

        sessions_info = [
            ("Sydney",           "21:00–06:00 UTC", "22:00–07:00 WAT",
             "🦘", "AUD/NZD pairs — low volatility"),
            ("Tokyo/Asian",      "00:00–09:00 UTC", "01:00–10:00 WAT",
             "🗼", "JPY pairs — AMD accumulation"),
            ("London Kill Zone", "07:00–10:00 UTC", "08:00–11:00 WAT",
             "🎯", "EUR/GBP/XAU — manipulation then expansion"),
            ("London Full",      "07:00–16:00 UTC", "08:00–17:00 WAT",
             "🏦", "All majors — institutional order flow"),
            ("NY Kill Zone",     "12:00–15:00 UTC", "13:00–16:00 WAT",
             "🎯", "EUR/USD/XAU — peak volume"),
            ("London-NY Overlap","12:00–16:00 UTC", "13:00–17:00 WAT",
             "⚡", "ALL pairs — maximum liquidity"),
            ("New York Full",    "12:00–21:00 UTC", "13:00–22:00 WAT",
             "🗽", "USD pairs — trend continuation"),
            ("Off Hours",        "21:00–00:00 UTC", "22:00–01:00 WAT",
             "😴", "Avoid if possible — signals still fire with ⚠️"),
        ]

        for name, utc_t_s, wat_s, em, note in sessions_info:
            active = (" ◀️ <b>NOW</b>"
                      if session.session_name in name else "")
            msg += (
                f"{em} <b>{name}</b>{active}\n"
                f"   UTC: {utc_t_s} | WAT: {wat_s}\n"
                f"   {note}\n\n")

        msg += (
            f"{'─'*35}\n"
            f"<b>💡 Rules:</b>\n"
            f"• Best entries: Kill Zones\n"
            f"• Best overall: London-NY Overlap\n"
            f"• Off-hours: higher threshold, still active\n"
            f"• AMD: Accumulate→Manipulate→Distribute\n\n"
            f"<b>⚠️ Risk:</b> 1-2% max per trade. "
            f"Stop-loss every trade.")
        return msg


SA = SessionAdvisor()

# ════════════════════════════════════════════════════════════════
#  MESSAGE FORMATTERS
# ════════════════════════════════════════════════════════════════

def fmt_signal_message(pair:    str,
                       tf:      str,
                       result:  Dict,
                       of:      OrderFlowData,
                       af:      Optional[AdvancedFlowData],
                       session: SessionData,
                       regime:  RegimeData,
                       sim:     Optional[Dict] = None) -> str:
    signal  = result["signal"]
    name    = pair_display(pair)
    em      = asset_emoji(pair)
    ts      = session.wat_time
    d       = signal.direction
    de      = "🟢" if d == "LONG" else "🔴"
    dw      = "BUY" if d == "LONG" else "SELL"
    q       = signal.quality.value
    qe      = quality_emoji(q)
    cp      = of.price
    tgt     = signal.target
    stp     = signal.stop
    rr      = signal.rr_ratio        # CORRECT validated R/R
    rp      = signal.reward_pips     # validated reward pips
    rkp     = signal.risk_pips       # validated risk pips
    conf    = signal.confidence
    cal     = result.get("calibrated_prob", 0.5)
    ml_l    = result.get("ml_label", "")
    strat   = signal.strategy.value
    htf     = result.get("htf_bias", "NEUTRAL")
    be_p    = signal.breakeven_price
    trail_p = signal.trailing_price

    mode, risk_pct, _ = PERF_ENGINE.get_risk_mode()
    recovery_warn = ""
    if mode == "RECOVERY":
        recovery_warn = (
            "\n⚠️ <b>RECOVERY MODE — A+ quality only</b>\n")
    off_warn = ""
    if session.session_type == SessionType.OFF_HOURS:
        off_warn = (
            "\n⚠️ <b>OFF-HOURS — Reduced liquidity. "
            "Wider stops recommended.</b>\n")

    trade_id = result.get("trade_id", "FQ-UNKNOWN")

    msg = (
        f"{'='*35}\n"
        f"🎯 <b>SIGNAL — {name}</b>\n"
        f"{'='*35}\n"
        f"{off_warn}{recovery_warn}\n"
        f"🆔 Trade ID: <code>{trade_id}</code>\n"
        f"{em} <b>{name}</b> | "
        f"{TIMEFRAMES.get(tf,{}).get('emoji','')} {tf}\n"
        f"🕐 {ts} | {session.session_name}"
        f"{' | 🎯 ' + session.kill_zone_name if session.is_kill_zone else ''}\n\n"

        f"{'─'*35}\n"
        f"{de} <b>{dw}</b> | {qe} <b>{q}</b>\n"
        f"📊 Confidence:      <b>{conf:.0f}%</b>\n"
        f"🎯 Calibrated Prob: <b>{cal:.0%}</b> "
        f"<i>(actual historical accuracy)</i>\n"
    )

    if ml_l and "Accumulating" not in ml_l and "Error" not in ml_l:
        msg += (
            f"{ml_l} "
            f"({result.get('ml_prob', 0):.0%})\n")

    msg += (
        f"📐 Strategy:  <b>{strat}</b>\n"
        f"📈 HTF Bias:  <b>{htf}</b>\n"
        f"🌊 Regime:    <b>{regime.regime.value}</b>"
    )

    if regime.contradictory:
        msg += " ⚠️ (contradictory signals)"

    msg += (
        f"\nH:{regime.hurst_exponent:.2f} | "
        f"ER:{regime.efficiency_ratio:.2f} | "
        f"VR:{regime.volatility_ratio:.2f}\n\n"

        f"{'─'*35}\n"
        f"<b>📍 Trade Levels:</b>\n"
        f"├ Entry:     <code>{fmt_price(cp,  pair)}</code>\n"
        f"├ 🎯 Target: <code>{fmt_price(tgt, pair)}</code> "
        f"(+{rp:.1f} pips)\n"
        f"└ ⛔ Stop:   <code>{fmt_price(stp, pair)}</code> "
        f"(-{rkp:.1f} pips)\n\n"

        f"⚖️ R/R: <b>1:{rr:.2f}</b> | "
        f"Risk: <b>{risk_pct:.1f}%</b> of account\n\n"

        f"<b>📌 Management Levels:</b>\n"
        f"├ Breakeven @ <code>{fmt_price(be_p, pair)}</code> "
        f"(move SL to entry)\n"
        f"└ Trail SL @ <code>{fmt_price(trail_p, pair)}</code> "
        f"(lock profits)\n\n"

        f"{'─'*35}\n"
        f"<b>Why this signal fired:</b>\n"
    )

    for i, r in enumerate(signal.reasons[:6], 1):
        msg += f"  {i}. {r}\n"
    msg += "\n"

    # ICT context
    ict   = result.get("ict", {})
    swept = ict.get("swept", [])
    disp  = ict.get("displacement")
    amd   = ict.get("amd_phase", "")

    if swept or disp or amd:
        msg += f"{'─'*35}\n<b>🏛️ ICT Context:</b>\n"
        for lv in swept[:2]:
            msg += (
                f"• Swept: {lv.level_type} @ "
                f"<code>{fmt_price(lv.price, pair)}</code>\n")
        if disp:
            msg += (
                f"• Displacement: "
                f"{'🟢 Bullish' if disp.is_bullish else '🔴 Bearish'} "
                f"({disp.body_ratio:.0%} body)\n")
        if amd:
            msg += f"• AMD Phase: {amd}\n"
        msg += "\n"

    # Smart money
    if af:
        sm_e = ("🔥" if "HIGH" in af.smart_money_activity else "📊")
        msg += (
            f"{'─'*35}\n{sm_e} <b>Smart Money:</b>\n"
            f"├ VPIN:      <code>{af.vpin:.2%}</code> "
            f"({af.vpin_level})\n"
            f"├ Informed:  <b>{af.informed_signal}</b>\n"
            f"├ Aggressor: <b>{af.aggressor_side}</b>\n"
            f"└ Inst Score:<b>{af.institutional_score:.0f}%</b>\n\n"
        )
        if af.iceberg_count > 0:
            msg += (
                f"🧊 <b>{af.iceberg_count} iceberg levels detected</b>\n\n")

    # Key levels
    kl = result.get("key_levels", [])
    if kl:
        msg += f"{'─'*35}\n<b>Key Levels:</b>\n"
        for lv in kl[:5]:
            dist = pips_diff(pair, abs(lv["price"] - cp))
            side = "above" if lv["price"] > cp else "below"
            msg += (
                f"• {lv['type']}: "
                f"<code>{fmt_price(lv['price'], pair)}</code> "
                f"({dist:.0f}p {side})\n")
        msg += "\n"

    # Pattern memory
    if sim and sim.get("found"):
        msg += f"{sim['narrative']}\n\n"

    msg += (
        f"{'='*35}\n"
        f"<i>✅ Live tracking ON — alerts on TP/SL + breakeven.\n"
        f"🆔 Reference: <code>{trade_id}</code>\n"
        f"⚠️ Stop-loss at <code>{fmt_price(stp, pair)}</code>. "
        f"Risk max {risk_pct:.1f}%.</i>"
    )
    return msg


def fmt_no_setup_message(pair:    str,
                         tf:      str,
                         of:      OrderFlowData,
                         regime:  RegimeData,
                         session: SessionData,
                         factors: Optional[List] = None) -> str:
    name   = pair_display(pair)
    em     = asset_emoji(pair)
    ts     = session.wat_time
    bull_n = sum(1 for f in (factors or [])
                 if f.direction == "BULLISH")
    bear_n = sum(1 for f in (factors or [])
                 if f.direction == "BEARISH")
    return (
        f"{'='*35}\n⚪ <b>NO CLEAR SETUP</b>\n{'='*35}\n\n"
        f"{em} <b>{name}</b> | "
        f"{TIMEFRAMES.get(tf,{}).get('emoji','')} {tf}\n"
        f"🕐 {ts}\n\n"
        f"{'─'*35}\n<b>Why no signal:</b>\n"
        f"├ 🟢 Bullish factors: {bull_n}\n"
        f"├ 🔴 Bearish factors: {bear_n}\n"
        f"├ No strategy met all criteria\n"
        f"└ Signal geometry validation active\n\n"
        f"<b>Market State:</b>\n"
        f"├ Regime:    {regime.regime.value}\n"
        f"├ Hurst:     {regime.hurst_exponent:.2f} "
        f"({'↗ Trending' if regime.hurst_exponent>0.55 else '↔ Ranging' if regime.hurst_exponent<0.45 else '◌ Random'})\n"
        f"├ Vol Ratio: {regime.volatility_ratio:.2f} "
        f"({'Expanding' if regime.volatility_ratio>1.2 else 'Compressing' if regime.volatility_ratio<0.7 else 'Normal'})\n"
        f"├ ADR Used:  {regime.adr_consumed_pct:.0f}%\n"
        f"└ Contradict:{regime.contradictory}\n\n"
        f"<b>Flow:</b>\n"
        f"├ CVD:       <code>{fmt_signed(of.cvd)}</code>\n"
        f"├ Imbalance: <code>{of.imbalance:+.1f}%</code> "
        f"(z={of.imbalance_zscore:+.1f}σ)\n"
        f"└ VWAP Z:    <code>{of.vwap_zscore:+.1f}σ</code>\n\n"
        f"{'='*35}\n"
        f"<i>💡 Patience is the edge. "
        f"Every rejected signal protects your capital.</i>"
    )


def fmt_outcome_notification(pred:  QuantPrediction,
                             atype: str,
                             cp:    float) -> str:
    """
    Format outcome notification.
    ONLY called when pred.was_sent_to_users == True.
    """
    em   = asset_emoji(pred.pair)
    name = pair_display(pred.pair)
    pip  = pip_value(pred.pair)
    tid  = pred.trade_id

    if atype == "TARGET_HIT":
        return (
            f"{'='*35}\n"
            f"🎯 <b>TARGET HIT! ✅</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{tid}</code>\n"
            f"{em} <b>{name}</b> | {pred.direction}\n"
            f"📐 Strategy: {pred.strategy}\n\n"
            f"├ Entry:   <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ Target:  <code>{fmt_price(pred.target_price,  pred.pair)}</code>\n"
            f"├ Exit:    <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"├ 💰 Profit: <b>+{pred.pips_gained:.1f} pips</b>\n"
            f"├ R/R was: 1:{pred.rr_ratio:.2f}\n"
            f"├ MFE:     {pred.mfe_pips:.1f}p (max unrealized profit)\n"
            f"└ MAE:     {pred.mae_pips:.1f}p (max heat taken)\n\n"
            f"Quality: {pred.quality} | "
            f"Conf: {pred.confidence:.0f}%\n"
            f"{'='*35}\n"
            f"<i>Winning trade! Risk management made this possible.\n"
            f"Reference this trade: <code>{tid}</code></i>"
        )
    elif atype == "STOP_HIT":
        return (
            f"{'='*35}\n"
            f"❌ <b>STOP TRIGGERED</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{tid}</code>\n"
            f"{em} <b>{name}</b> | {pred.direction}\n"
            f"📐 Strategy: {pred.strategy}\n\n"
            f"├ Entry:  <code>{fmt_price(pred.current_price,      pred.pair)}</code>\n"
            f"├ Stop:   <code>{fmt_price(pred.invalidation_price, pred.pair)}</code>\n"
            f"├ Exit:   <code>{fmt_price(cp,                       pred.pair)}</code>\n"
            f"├ Loss:   <b>-{abs(pred.pips_gained):.1f} pips</b>\n"
            f"├ R/R was: 1:{pred.rr_ratio:.2f}\n"
            f"├ MAE:    {pred.mae_pips:.1f}p (went this far against)\n"
            f"└ MFE:    {pred.mfe_pips:.1f}p (closest to target)\n\n"
            f"{'='*35}\n"
            f"<i>The stop-loss protected your capital. "
            f"Losing trades are normal — even the best systems lose.\n"
            f"Reference: <code>{tid}</code></i>"
        )
    elif atype == "BREAKEVEN":
        return (
            f"{'='*35}\n"
            f"📌 <b>BREAKEVEN ALERT</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{tid}</code>\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"Price reached <b>50% of target</b>.\n\n"
            f"├ Entry:     <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ Current:   <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"├ Target:    <code>{fmt_price(pred.target_price,   pred.pair)}</code>\n"
            f"└ Stop:      <code>{fmt_price(pred.invalidation_price, pred.pair)}</code>\n\n"
            f"{'─'*35}\n"
            f"<b>💡 Action: Consider moving stop-loss to entry "
            f"({fmt_price(pred.current_price, pred.pair)}) "
            f"to make this a risk-free trade.</b>\n"
            f"{'='*35}"
        )
    elif atype == "TRAILING":
        return (
            f"{'='*35}\n"
            f"🔄 <b>TRAILING STOP ALERT</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{tid}</code>\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"Price reached <b>75% of target</b>! 🚀\n\n"
            f"├ Entry:     <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ Current:   <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"├ Target:    <code>{fmt_price(pred.target_price,   pred.pair)}</code>\n"
            f"└ Unrealised: <b>+{pred.mfe_pips:.1f} pips</b>\n\n"
            f"{'─'*35}\n"
            f"<b>💡 Action: Consider trailing your stop-loss "
            f"to lock in profits. Move stop "
            f"{'above' if pred.direction=='SHORT' else 'below'} "
            f"the most recent swing "
            f"{'high' if pred.direction=='SHORT' else 'low'}.</b>\n"
            f"{'='*35}"
        )
    else:  # EXPIRED
        p_pips = ((cp - pred.current_price) if pred.direction == "LONG"
                  else (pred.current_price - cp)) / pip
        return (
            f"{'='*35}\n"
            f"⏰ <b>SIGNAL EXPIRED</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{tid}</code>\n"
            f"{em} <b>{name}</b> | {pred.direction}\n\n"
            f"├ Entry:   <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ Target:  <code>{fmt_price(pred.target_price,  pred.pair)}</code>\n"
            f"├ Current: <code>{fmt_price(cp,                  pred.pair)}</code>\n"
            f"└ P/L:     {p_pips:+.1f} pips (24h expiry)\n\n"
            f"<i>Setup expired. Thesis did not play out in time.\n"
            f"Reference: <code>{tid}</code></i>"
        )


def fmt_analysis_messages(pair:    str,
                           tf:      str,
                           of:      OrderFlowData,
                           af:      Optional[AdvancedFlowData],
                           vp:      Optional[VolumeProfile],
                           ob:      Optional[OrderBookData],
                           pb:      Optional[PositionBookData],
                           regime:  RegimeData,
                           session: SessionData,
                           ict:     Optional[Dict] = None) -> List[str]:
    msgs = []
    name = pair_display(pair)
    ts   = session.wat_time
    ti   = TIMEFRAMES.get(tf, {})
    vs   = "above" if of.price > of.vwap else "below"

    m1 = (
        f"{'='*32}\n📊 <b>{name} DEEP ANALYSIS</b>\n{'='*32}\n"
        f"⏰ {ts} | {ti.get('emoji','')} {ti.get('label',tf)}\n\n"
        f"💰 <b>Price:</b> <code>{fmt_price(of.price, pair)}</code>\n"
        f"📐 <b>VWAP:</b>  <code>{fmt_price(of.vwap,  pair)}</code> "
        f"({vs} — {'bullish' if vs=='above' else 'bearish'} bias)\n"
        f"📊 VWAP Z: <code>{of.vwap_zscore:+.1f}σ</code>\n\n"

        f"{'─'*32}\n<b>📈 ORDER FLOW</b>\n{'─'*32}\n"
        f"{'🟢' if of.cvd>0 else '🔴'} "
        f"<b>{'BUYERS' if of.cvd>0 else 'SELLERS'} dominant</b>\n\n"
        f"├ Buy Vol:    <code>{fmt_num(of.buy_volume)}</code> "
        f"({of.buy_pct:.1f}%)\n"
        f"├ Sell Vol:   <code>{fmt_num(of.sell_volume)}</code> "
        f"({of.sell_pct:.1f}%)\n"
        f"├ Imbalance:  <code>{fmt_pct(of.imbalance)}</code> "
        f"(z={of.imbalance_zscore:+.1f}σ)\n"
        f"├ CVD:        <code>{fmt_signed(of.cvd)}</code>\n"
        f"├ Momentum:   <code>{fmt_signed(of.delta_momentum)}</code>\n"
        f"├ Vol Trend:  <code>{fmt_pct(of.volume_trend)}</code>\n"
        f"└ Efficiency: <code>{of.efficiency_ratio:.2f}</code>\n\n"

        f"{'─'*32}\n<b>📊 MARKET REGIME</b>\n{'─'*32}\n"
        f"Regime: <b>{regime.regime.value}</b>"
        f"{' ⚠️ CONTRADICTORY' if regime.contradictory else ''}\n"
        f"├ Hurst:      {regime.hurst_exponent:.3f} "
        f"({'↗ Trending' if regime.hurst_exponent>0.55 else '↔ Mean-Rev' if regime.hurst_exponent<0.45 else '◌ Random'})\n"
        f"├ Fractal:    {regime.fractal_dimension:.3f}\n"
        f"├ Entropy:    {regime.entropy:.3f}\n"
        f"├ Autocorr:   {regime.autocorr_lag1:+.3f}\n"
        f"├ Var Ratio:  {regime.variance_ratio:.3f}\n"
        f"├ Efficiency: {regime.efficiency_ratio:.3f}\n"
        f"├ Vol Ratio:  {regime.volatility_ratio:.2f}\n"
        f"└ ADR Used:   {regime.adr_consumed_pct:.0f}%\n"
    )
    msgs.append(m1)

    if af:
        m2 = (
            f"{'─'*32}\n<b>🧠 SMART MONEY</b>\n{'─'*32}\n\n"
            f"<b>VPIN:</b> <code>{af.vpin:.2%}</code> — {af.vpin_level}\n"
            f"<b>Toxicity:</b> <code>{af.toxicity:.2%}</code> — "
            f"{af.toxicity_level}\n"
            f"<b>Liquidity:</b> {af.liquidity_level}\n"
            f"<b>Depth Bias:</b> {af.depth_bias}\n\n"
            f"<b>Absorption Ratio:</b> {af.absorption_ratio:.2f}\n"
            f"<b>Aggressor:</b> <b>{af.aggressor_side}</b>\n"
            f"<b>Informed Signal:</b> <b>{af.informed_signal}</b>\n\n"
            f"{'🧊 <b>' + str(af.iceberg_count) + ' iceberg levels</b>' if af.iceberg_count > 0 else '• No icebergs'}\n\n"
            f"{'─'*32}\n"
            f"└ <b>Inst Score: {af.institutional_score:.0f}%</b> | "
            f"Smart Money: <b>{af.smart_money_activity}</b>\n"
        )
        msgs.append(m2)

    if ict:
        struct  = ict.get("structure", MarketStructure())
        fvgs    = ict.get("fvgs", [])
        obs     = ict.get("obs",  [])
        swept   = ict.get("swept", [])
        disp    = ict.get("displacement")
        asian_h = ict.get("asian_high")
        asian_l = ict.get("asian_low")

        m3 = (
            f"{'─'*32}\n<b>🏛️ ICT STRUCTURE</b>\n{'─'*32}\n\n"
            f"<b>Trend:</b> {struct.trend} | "
            f"<b>Internal:</b> {struct.internal_trend} | "
            f"<b>HTF:</b> {struct.htf_bias}\n"
            f"<b>BOS:</b> "
            f"{struct.last_bos.value if struct.last_bos else 'None'}\n"
            f"<b>CHOCH:</b> "
            f"{struct.last_choch.value if struct.last_choch else 'None'}\n\n"
        )
        if asian_h and asian_l:
            m3 += (
                f"<b>Asian Range:</b>\n"
                f"├ High: <code>{fmt_price(asian_h, pair)}</code>\n"
                f"├ Low:  <code>{fmt_price(asian_l, pair)}</code>\n"
                f"└ AMD:  <b>{ict.get('amd_phase','?')}</b>\n\n")
        if swept:
            m3 += "<b>Swept Liquidity:</b>\n"
            for lv in swept[:3]:
                m3 += (f"✅ {lv.level_type} @ "
                       f"<code>{fmt_price(lv.price, pair)}</code>\n")
            m3 += "\n"
        if disp:
            m3 += (
                f"<b>Displacement:</b> "
                f"{'🟢 Bullish' if disp.is_bullish else '🔴 Bearish'} "
                f"({disp.body_ratio:.0%} body)\n\n")
        unfilled = [f for f in fvgs if not f.filled][:3]
        if unfilled:
            m3 += "<b>Fair Value Gaps:</b>\n"
            for fvg in unfilled:
                dist = pips_diff(pair, abs(fvg.mid - of.price))
                m3 += (
                    f"{'🟢' if fvg.direction=='BULLISH' else '🔴'} "
                    f"{fmt_price(fvg.lower, pair)}–"
                    f"{fmt_price(fvg.upper, pair)} "
                    f"({dist:.0f}p away)\n")
            m3 += "\n"
        valid_obs = [ob for ob in obs if ob.valid][:3]
        if valid_obs:
            m3 += "<b>Order Blocks:</b>\n"
            for ob_b in valid_obs:
                m3 += (
                    f"{'🟢' if ob_b.direction=='DEMAND' else '🔴'} "
                    f"{ob_b.direction}: "
                    f"{fmt_price(ob_b.low, pair)}–"
                    f"{fmt_price(ob_b.high, pair)} "
                    f"(str:{ob_b.strength:.0f})\n")
        msgs.append(m3)

    m4 = ""
    if vp:
        in_va = vp.val <= of.price <= vp.vah
        m4 += (
            f"{'─'*32}\n<b>📐 VOLUME PROFILE</b>\n{'─'*32}\n\n"
            f"├ POC: <code>{fmt_price(vp.poc, pair)}</code>\n"
            f"├ VAH: <code>{fmt_price(vp.vah, pair)}</code>\n"
            f"├ VAL: <code>{fmt_price(vp.val, pair)}</code>\n"
            f"└ In VA: {'✅' if in_va else '❌'}\n\n")
    if pb:
        m4 += (
            f"{'─'*32}\n<b>👥 POSITION BOOK</b>\n{'─'*32}\n\n"
            f"├ Longs:   {pbar(pb.long_pct)} {pb.long_pct:.1f}%\n"
            f"├ Shorts:  {pbar(pb.short_pct)} {pb.short_pct:.1f}%\n"
            f"├ Skew:    <code>{fmt_pct(pb.skew)}</code>\n"
            f"├ Δ Skew:  <code>{pb.skew_change:+.1f}%</code>\n"
            f"├ Signal:  <b>{pb.contrarian_signal}</b>\n"
            f"└ Squeeze: {pb.squeeze_potential}\n\n")
    if ob:
        m4 += (
            f"{'─'*32}\n<b>📖 ORDER BOOK</b>\n{'─'*32}\n\n"
            f"├ Longs:   {pbar(ob.long_pct)} {ob.long_pct:.1f}%\n"
            f"├ Shorts:  {pbar(ob.short_pct)} {ob.short_pct:.1f}%\n"
            f"└ Pressure:<b>{ob.breakout_bias.replace('_',' ')}</b>\n")
        if ob.stop_cluster_above > 0:
            m4 += (f"├ Stop ↑: "
                   f"<code>{fmt_price(ob.stop_cluster_above, pair)}</code>\n")
        if ob.stop_cluster_below > 0:
            m4 += (f"└ Stop ↓: "
                   f"<code>{fmt_price(ob.stop_cluster_below, pair)}</code>\n")
    if m4:
        msgs.append(m4)

    msgs.append(
        f"\n{'='*32}\n"
        f"<i>Use /signal for prediction\n"
        f"Use /sessions for session guide\n"
        f"Use /math for mathematical explanations</i>"
    )
    return msgs


def fmt_market_summary(pair:        str,
                       of:          OrderFlowData,
                       af:          Optional[AdvancedFlowData],
                       vp:          Optional[VolumeProfile],
                       ob:          Optional[OrderBookData],
                       pb:          Optional[PositionBookData],
                       regime:      RegimeData,
                       session:     SessionData,
                       ict:         Optional[Dict],
                       pred_result: Optional[Dict]) -> str:
    name  = pair_display(pair)
    ts    = session.wat_time
    parts = []

    parts.append(
        f"{'='*35}\n🧠 <b>MARKET INTELLIGENCE BRIEF</b>\n{'='*35}\n"
        f"📍 <b>{name}</b> | 🕐 {ts}\n"
        f"Session: {session.session_name}"
        f"{' | 🎯 ' + session.kill_zone_name if session.is_kill_zone else ''}\n"
        f"Regime: <b>{regime.regime.value}</b>"
        f"{' ⚠️ Contradictory' if regime.contradictory else ''} | "
        f"AMD: <b>{session.amd_phase}</b>"
    )

    # Flow narrative
    cvd_dir = "NET BUYING" if of.cvd > 0 else "NET SELLING"
    imb_z   = of.imbalance_zscore
    flow_str = (
        f"Flow imbalance is <b>statistically extreme</b> "
        f"(z={imb_z:+.1f}σ) — "
        f"{'buyers' if imb_z>0 else 'sellers'} unusually dominant."
        if abs(imb_z) > 2.0 else
        f"Flow shows <b>{cvd_dir}</b> "
        f"(z={imb_z:+.1f}σ — normal conditions)."
    )
    parts.append(
        f"{'─'*35}\n<b>📈 ORDER FLOW</b>\n{'─'*35}\n\n"
        f"{flow_str}\n\n"
        f"VWAP z-score: <b>{of.vwap_zscore:+.1f}σ</b> — "
        f"price is "
        f"{'statistically expensive' if of.vwap_zscore>2 else 'statistically cheap' if of.vwap_zscore<-2 else 'near fair value'}.\n\n"
        f"Volume: "
        f"{'📈 Rising' if of.volume_trend>15 else '📉 Declining' if of.volume_trend<-15 else '➡️ Stable'} "
        f"({of.volume_trend:+.1f}%)"
        + (f"\n\n⚠️ <b>BUYING CLIMAX</b> — Reversal risk HIGH."
           if of.buying_climax else
           f"\n\n⚠️ <b>SELLING CLIMAX</b> — Reversal risk HIGH."
           if of.selling_climax else "")
    )

    # Regime
    h = regime.hurst_exponent
    if h > 0.58:
        regime_narr = (
            f"Hurst ({h:.3f}) confirms <b>trending market</b>. "
            f"Momentum strategies have edge."
            + (" ⚠️ BUT Efficiency Ratio contradicts — caution."
               if regime.contradictory else ""))
    elif h < 0.42:
        regime_narr = (
            f"Hurst ({h:.3f}) confirms <b>mean-reverting</b>. "
            f"Fade extremes, target VWAP.")
    else:
        regime_narr = (
            f"Hurst ({h:.3f}) — near <b>random walk</b>. "
            f"Wait for clearer regime.")
    parts.append(
        f"{'─'*35}\n<b>📊 REGIME</b>\n{'─'*35}\n\n"
        f"{regime_narr}\n\n"
        f"Volatility: "
        f"{'Compressing — breakout imminent 🔥' if regime.volatility_ratio<0.5 else 'Expanding — follow the move 🚀' if regime.volatility_ratio>1.8 else 'Normal conditions'} "
        f"(VR:{regime.volatility_ratio:.2f})"
    )

    # Smart money
    if af:
        sm = af.smart_money_activity
        sm_narr = (
            f"Smart money: <b>{sm}</b>. "
            f"Informed signal: <b>{af.informed_signal}</b>. "
            f"{'Follow institutions.' if 'HIGH' in sm else 'Mixed conditions.'}"
        )
        retail_txt = ""
        if pb:
            retail_txt = (
                f"\n\n<b>Retail:</b>\n"
                f"• {pb.long_pct:.0f}% LONG / {pb.short_pct:.0f}% SHORT\n"
                f"• Contrarian: <b>{pb.contrarian_signal}</b>\n"
                f"• Squeeze: {pb.squeeze_potential}")
        parts.append(
            f"{'─'*35}\n<b>🏦 SMART MONEY vs RETAIL</b>\n{'─'*35}\n\n"
            f"{sm_narr}\n\n"
            f"• VPIN: {af.vpin:.0%} — "
            f"{'Significant informed flow' if af.vpin>=0.5 else 'Low smart money'}\n"
            f"• Aggressor: <b>{af.aggressor_side}</b>\n"
            f"• Absorption: {af.absorption_ratio:.2f}\n"
            + (f"• 🧊 {af.iceberg_count} iceberg levels\n"
               if af.iceberg_count > 0 else "")
            + retail_txt
        )

    # ICT
    if ict:
        struct  = ict.get("structure", MarketStructure())
        swept   = ict.get("swept", [])
        disp    = ict.get("displacement")
        asian_h = ict.get("asian_high")
        asian_l = ict.get("asian_low")
        ict_txt = (
            f"{'─'*35}\n<b>🏛️ ICT</b>\n{'─'*35}\n\n"
            f"Trend: <b>{struct.trend}</b> | "
            f"HTF: <b>{struct.htf_bias}</b>\n\n"
        )
        if swept:
            ict_txt += (
                f"✅ <b>Liquidity Swept:</b> "
                f"{', '.join(lv.level_type for lv in swept[:2])} — "
                f"reversal or acceleration likely.\n\n")
        if disp:
            ict_txt += (
                f"<b>Displacement:</b> "
                f"{'🟢 Bullish' if disp.is_bullish else '🔴 Bearish'}\n\n")
        if asian_h and asian_l:
            pv = ("ABOVE" if of.price>asian_h else
                  "BELOW" if of.price<asian_l else "INSIDE")
            ict_txt += (
                f"Asian Range: {fmt_price(asian_l,pair)}–"
                f"{fmt_price(asian_h,pair)} | "
                f"Price {pv} | AMD: {ict.get('amd_phase','?')}\n")
        parts.append(ict_txt)

    # Conclusion
    bull = bear = 0
    if of.cvd > 0:                              bull += 2
    else:                                        bear += 2
    if of.vwap_zscore > 0.5:                    bull += 1
    elif of.vwap_zscore < -0.5:                 bear += 1
    if af and "BULLISH" in af.informed_signal:  bull += 3
    elif af and "BEARISH" in af.informed_signal: bear += 3
    if pb and pb.contrarian_signal == "BULLISH": bull += 2
    elif pb and pb.contrarian_signal == "BEARISH": bear += 2

    total = bull + bear
    bpct  = bull / total * 100 if total > 0 else 50

    if bpct >= 65:
        bias = "📈 <b>BULLISH BIAS</b>"
        bt   = f"{bull}/{total} signals favour buyers."
    elif bpct <= 35:
        bias = "📉 <b>BEARISH BIAS</b>"
        bt   = f"{bear}/{total} signals favour sellers."
    else:
        bias = "↔️ <b>MIXED</b>"
        bt   = "Evidence balanced — no clear edge."

    tgt_txt = ""
    if pred_result and pred_result.get("has_setup"):
        sig = pred_result["signal"]
        rr  = sig.rr_ratio
        tgt_txt = (
            f"\n\n<b>🎯 Algorithm Signal:</b>\n"
            f"{'LONG' if sig.direction=='LONG' else 'SHORT'} to "
            f"<code>{fmt_price(sig.target, pair)}</code> "
            f"({pred_result['calibrated_prob']:.0%} calibrated prob)\n"
            f"Stop: <code>{fmt_price(sig.stop, pair)}</code> | "
            f"R/R: 1:{rr:.2f} ✅"
        )
    elif vp:
        tgt_txt = (
            f"\n\n<b>Key Levels:</b>\n"
            f"• POC: {fmt_price(vp.poc, pair)}\n"
            f"• VAH: {fmt_price(vp.vah, pair)}\n"
            f"• VAL: {fmt_price(vp.val, pair)}"
        )

    parts.append(
        f"{'='*35}\n<b>📊 CONCLUSION</b>\n{'='*35}\n\n"
        f"{bias}\n{bt}"
        f"{tgt_txt}\n\n"
        f"<i>No analysis is 100% correct. "
        f"Risk management is everything.</i>"
    )

    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════

def chart_signal(candles:    List[Candle],
                 instrument: str,
                 result:     Dict,
                 of:         OrderFlowData,
                 af:         Optional[AdvancedFlowData],
                 regime:     RegimeData,
                 ict:        Optional[Dict] = None) -> io.BytesIO:
    fig = plt.figure(figsize=(14, 13), facecolor='#0d1117')
    gs  = GridSpec(4, 1, figure=fig,
                   height_ratios=[3, 1, 1, 0.9], hspace=0.14)
    C   = {
        'bg':   '#0d1117', 'text': '#c9d1d9', 'grid': '#21262d',
        'up':   '#3fb950', 'dn':   '#f85149', 'tgt':  '#3fb950',
        'stp':  '#f85149', 'ent':  '#f0883e', 'poc':  '#a371f7',
        'vwap': '#58a6ff', 'sm':   '#ffd700', 'fvg':  '#58a6ff',
        'ob':   '#ff9500', 'liq':  '#ff4ff8', 'be':   '#ffaa00',
    }

    signal = result.get("signal")
    if not signal:
        plt.close(fig)
        return chart_no_setup(candles, instrument)

    disp_c = candles[-70:] if len(candles) > 70 else candles
    n      = len(disp_c)
    cp     = candles[-1].close
    trade_id = result.get("trade_id", "")

    # ── Panel 1: Price ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C['bg'])

    for i, c in enumerate(disp_c):
        col = C['up'] if c.is_bullish else C['dn']
        ax1.plot([i,i], [c.low,  c.high],  color=col, lw=0.8)
        ax1.plot([i,i], [c.open, c.close], color=col, lw=3.5)

    # Entry / Target / Stop / Breakeven / Trailing
    for yv, lbl, col, ls in [
        (cp,                  f"ENTRY\n{fmt_price(cp, instrument)}",
         C['ent'], '-'),
        (signal.target,       f"TP\n{fmt_price(signal.target, instrument)}",
         C['tgt'], '--'),
        (signal.stop,         f"SL\n{fmt_price(signal.stop, instrument)}",
         C['stp'], '--'),
        (signal.breakeven_price,
         f"BE\n{fmt_price(signal.breakeven_price, instrument)}",
         C['be'], ':'),
    ]:
        if yv > 0:
            ax1.axhline(y=yv, color=col, linestyle=ls, lw=2, alpha=0.9)
            ax1.text(n+1, yv, lbl, color=col, fontsize=7,
                     fontweight='bold', va='center')

    ax1.axhline(y=of.vwap, color=C['vwap'],
                linestyle=':', lw=1.5, alpha=0.7)

    # ICT overlays
    if ict:
        lo_ = min(c.low  for c in disp_c)
        hi_ = max(c.high for c in disp_c)
        for fvg in ict.get("fvgs", [])[:4]:
            if not fvg.filled and lo_ <= fvg.mid <= hi_:
                fc = C['up'] if fvg.direction=="BULLISH" else C['dn']
                ax1.axhspan(fvg.lower, fvg.upper, alpha=0.1, color=fc)
        for ob_b in ict.get("obs", [])[:3]:
            if ob_b.valid and lo_ <= ob_b.mid <= hi_:
                ax1.axhspan(ob_b.low, ob_b.high, alpha=0.08, color=C['ob'])
        for lv in ict.get("liq_levels", [])[:4]:
            if lo_ <= lv.price <= hi_:
                lsty = '-' if lv.swept else '--'
                ax1.axhline(y=lv.price, color=C['liq'],
                            linestyle=lsty, lw=0.8, alpha=0.5)

    # Arrow
    ac = C['up'] if signal.direction == 'LONG' else C['dn']
    ax1.annotate('', xy=(n-5, signal.target), xytext=(n-5, cp),
                 arrowprops=dict(arrowstyle='->', color=ac, lw=3))

    qe = quality_emoji(signal.quality.value)
    de = "🟢" if signal.direction == 'LONG' else "🔴"
    ax1.set_title(
        f"{asset_emoji(instrument)} {pair_display(instrument)} | "
        f"{de} {signal.direction} | {qe} {signal.quality.value} | "
        f"Conf:{result['confidence']:.0f}% | "
        f"R/R:1:{signal.rr_ratio:.2f} | {trade_id}",
        color=C['text'], fontsize=10, fontweight='bold', pad=10)
    ax1.tick_params(colors=C['text'], labelsize=8)
    ax1.grid(True, alpha=0.12, color=C['grid'])
    ax1.set_xlim(-2, n+18)

    # ── Panel 2: Delta ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C['bg'])
    deltas = [c.delta for c in disp_c]
    ax2.bar(range(n), deltas,
            color=[C['up'] if d>=0 else C['dn'] for d in deltas],
            alpha=0.75, width=0.8)
    ax2.axhline(y=0, color=C['grid'], lw=1)
    ax2.set_title('Order Flow Delta', color=C['text'], fontsize=9, pad=4)
    ax2.tick_params(colors=C['text'], labelsize=7)
    ax2.grid(True, alpha=0.12, color=C['grid'])
    ax2.set_xlim(-2, n+18)

    # ── Panel 3: CVD ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(C['bg'])
    cvd     = list(np.cumsum(deltas))
    cvd_col = C['up'] if cvd and cvd[-1]>=cvd[0] else C['dn']
    ax3.fill_between(range(n), cvd, alpha=0.25, color=cvd_col)
    ax3.plot(range(n), cvd, color=cvd_col, lw=1.5)
    ax3.set_title('CVD', color=C['text'], fontsize=9, pad=4)
    ax3.tick_params(colors=C['text'], labelsize=7)
    ax3.grid(True, alpha=0.12, color=C['grid'])
    ax3.set_xlim(-2, n+18)

    # ── Panel 4: Smart Money ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(C['bg'])
    if af:
        sm_s = []
        for i in range(n):
            wc  = disp_c[max(0, i-10):i+1]
            bv  = sum(c.buy_volume  for c in wc)
            sv  = sum(c.sell_volume for c in wc)
            tv  = bv + sv
            sm_s.append(abs(bv-sv)/tv*100 if tv>0 else 0.0)
        ax4.fill_between(range(n), sm_s, alpha=0.5, color=C['sm'])
        ax4.plot(range(n), sm_s, color=C['sm'], lw=1.5)
        ax4.axhline(y=50, color='red', linestyle='--', lw=1, alpha=0.6)
        ax4.set_title(
            f"Smart Money | VPIN:{af.vpin:.0%} | "
            f"{af.smart_money_activity} | {regime.regime.value}",
            color=C['sm'], fontsize=9, pad=4)
    else:
        ax4.text(0.5, 0.5, f"Regime: {regime.regime.value}",
                 ha='center', va='center', color=C['text'],
                 transform=ax4.transAxes, fontsize=11)
    ax4.tick_params(colors=C['text'], labelsize=7)
    ax4.grid(True, alpha=0.12, color=C['grid'])
    ax4.set_xlim(-2, n+18)

    ts_str = SESSION_ENGINE.wat_time()
    fig.text(0.98, 0.01,
             f'ForexQuant v8.1 | {ts_str}',
             ha='right', color=C['text'], fontsize=7, alpha=0.7)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor=C['bg'], edgecolor='none',
                bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    gc.collect()
    return buf


def chart_no_setup(candles: List[Candle], instrument: str) -> io.BytesIO:
    fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.14)
    C   = {'bg':'#0d1117','text':'#c9d1d9','grid':'#21262d',
           'up':'#3fb950','dn':'#f85149','neu':'#f0883e'}
    d   = candles[-70:] if len(candles) > 70 else candles
    n   = len(d)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C['bg'])
    for i, c in enumerate(d):
        col = C['up'] if c.is_bullish else C['dn']
        ax1.plot([i,i], [c.low, c.high],   color=col, lw=0.8)
        ax1.plot([i,i], [c.open, c.close], color=col, lw=3.5)
    ax1.set_title(
        f"{asset_emoji(instrument)} {pair_display(instrument)} "
        f"| ⚪ NO CLEAR SETUP",
        color=C['neu'], fontsize=13, fontweight='bold', pad=14)
    ax1.tick_params(colors=C['text'], labelsize=8)
    ax1.grid(True, alpha=0.12, color=C['grid'])
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C['bg'])
    deltas = [c.delta for c in d]
    ax2.bar(range(n), deltas,
            color=[C['up'] if x>=0 else C['dn'] for x in deltas],
            alpha=0.7, width=0.8)
    ax2.axhline(y=0, color=C['grid'], lw=1)
    ax2.set_title('Delta', color=C['text'], fontsize=9, pad=4)
    ax2.tick_params(colors=C['text'], labelsize=7)
    ax2.grid(True, alpha=0.12, color=C['grid'])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor=C['bg'], edgecolor='none',
                bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    gc.collect()
    return buf


def chart_strength(strengths: Dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    C  = {'up':'#3fb950','dn':'#f85149','text':'#c9d1d9','grid':'#21262d'}
    sc = sorted(strengths.values(),
                key=lambda x: x.strength, reverse=True)
    cu = [cs.currency for cs in sc]
    va = [cs.strength for cs in sc]
    mo = [cs.momentum for cs in sc]
    bc = [C['up'] if v>0 else C['dn'] for v in va]
    yp = np.arange(len(cu))
    ax.barh(yp, va, color=bc, alpha=0.85, height=0.6)
    ax.set_yticks(yp)
    ax.set_yticklabels(
        cu, color=C['text'], fontsize=13, fontweight='bold')
    ax.axvline(x=0, color=C['grid'], lw=2)
    for i, (curr, v, m) in enumerate(zip(cu, va, mo)):
        em  = "💪" if v>0.2 else ("💀" if v<-0.2 else "➖")
        mom = (f"↑" if m>0.01 else ("↓" if m<-0.01 else ""))
        ax.text(v+(0.06 if v>=0 else -0.06), i,
                f'{em} {v:+.2f}% {mom}', va='center',
                ha='left' if v>=0 else 'right',
                color=C['text'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Strength (%)', color=C['text'], fontsize=11)
    ts  = SESSION_ENGINE.wat_time()
    ax.set_title(
        f'💱 Currency Strength (24H) — {ts}',
        color=C['text'], fontsize=14, fontweight='bold', pad=18)
    ax.tick_params(colors=C['text'])
    ax.grid(True, alpha=0.18, color=C['grid'], axis='x')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110,
                facecolor='#0d1117', edgecolor='none',
                bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    gc.collect()
    return buf


def chart_performance(live_predictions: List[Dict]) -> Optional[io.BytesIO]:
    live = [p for p in live_predictions
            if p.get("outcome") in ("WIN","LOSS")]
    if not live:
        return None
    try:
        fig    = plt.figure(figsize=(14, 10), facecolor='#0d1117')
        gs     = GridSpec(2, 2, figure=fig,
                          hspace=0.45, wspace=0.35)
        C      = {'bg':'#0d1117','text':'#c9d1d9','grid':'#21262d',
                  'up':'#3fb950','dn':'#f85149','acc':'#58a6ff',
                  'gold':'#ffd700'}

        # Equity curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(C['bg'])
        pips_c = []
        run    = 0.0
        for p in live:
            run += p.get("pips_gained", 0.0)
            pips_c.append(run)
        col_ = C['up'] if pips_c[-1]>=0 else C['dn']
        ax1.plot(pips_c, color=col_, lw=2)
        ax1.fill_between(range(len(pips_c)), pips_c,
                         alpha=0.2, color=col_)
        ax1.axhline(y=0, color=C['grid'], lw=1.5, linestyle='--')
        ax1.set_title(
            f"Equity Curve ({pips_c[-1]:+.0f} pips)",
            color=C['text'], fontsize=11, fontweight='bold')
        ax1.tick_params(colors=C['text'], labelsize=8)
        ax1.grid(True, alpha=0.15, color=C['grid'])

        # Win/Loss pie
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(C['bg'])
        lw_  = sum(1 for p in live if p.get("outcome")=="WIN")
        ll_  = len(live) - lw_
        if lw_ + ll_ > 0:
            pv  = [v for v in [lw_, ll_] if v > 0]
            pl  = [lb for v, lb in zip([lw_,ll_],["Wins","Losses"])
                   if v > 0]
            pc  = [C['up'], C['dn']][:len(pv)]
            ax2.pie(pv, labels=pl, colors=pc, autopct='%1.0f%%',
                    startangle=90,
                    textprops={'color': C['text'], 'fontsize': 11})
        wr_ = lw_/(lw_+ll_) if (lw_+ll_) > 0 else 0
        ax2.set_title(f"Win Rate: {wr_:.1%}",
                      color=C['text'], fontsize=11, fontweight='bold')

        # Strategy bars
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(C['bg'])
        st_data = {}
        for p in live:
            st = p.get("strategy","Unknown")[:22]
            if st not in st_data:
                st_data[st] = {"w":0,"l":0}
            if p.get("outcome")=="WIN": st_data[st]["w"] += 1
            else:                       st_data[st]["l"] += 1
        if st_data:
            st_n = list(st_data.keys())
            st_w = [(st_data[s]["w"]/max(st_data[s]["w"]+
                    st_data[s]["l"],1)*100) for s in st_n]
            st_c = [C['up'] if w>=55 else C['dn'] for w in st_w]
            ax3.barh(range(len(st_n)), st_w, color=st_c, alpha=0.85)
            ax3.set_yticks(range(len(st_n)))
            ax3.set_yticklabels(
                [s[:20] for s in st_n],
                color=C['text'], fontsize=8)
            ax3.axvline(x=50, color=C['grid'], lw=1.5, linestyle='--')
            ax3.set_xlim(0, 100)
        ax3.set_title("Win Rate by Strategy",
                      color=C['text'], fontsize=11, fontweight='bold')
        ax3.tick_params(colors=C['text'], labelsize=8)
        ax3.grid(True, alpha=0.15, color=C['grid'], axis='x')

        # ML accuracy
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(C['bg'])
        ml_h = ML_ENGINE.accuracy_history
        ml_s = ML_ENGINE.get_stats()
        if ml_h and len(ml_h) >= 2:
            ax4.plot([v*100 for v in ml_h],
                     color=C['acc'], lw=2.5,
                     marker='o', markersize=5)
            ax4.fill_between(range(len(ml_h)),
                             [v*100 for v in ml_h],
                             alpha=0.2, color=C['acc'])
            ax4.axhline(y=50, color=C['grid'], lw=1.5, linestyle='--')
            ax4.set_ylim(0, 100)
            ax4.set_title(
                f"ML OOS Accuracy: {ml_h[-1]:.1%}",
                color=C['gold'], fontsize=11, fontweight='bold')
        else:
            ax4.text(0.5, 0.5,
                     f"ML accumulating...\n"
                     f"({ml_s['n_samples']}/{ML_MIN_SAMPLES} resolved)\n"
                     f"({ml_s['n_total']} total tracked)",
                     ha='center', va='center',
                     color=C['text'], fontsize=10,
                     transform=ax4.transAxes)
            ax4.set_title("ML Engine",
                          color=C['gold'],
                          fontsize=11, fontweight='bold')
        ax4.tick_params(colors=C['text'], labelsize=8)
        ax4.grid(True, alpha=0.15, color=C['grid'])

        ts_str = SESSION_ENGINE.wat_time()
        fig.text(0.5, 0.97,
                 "FOREX QUANT v8.1 — PERFORMANCE ANALYTICS",
                 ha='center', color=C['text'],
                 fontsize=13, fontweight='bold')
        fig.text(0.98, 0.01, f"Generated: {ts_str}",
                 ha='right', color=C['text'],
                 fontsize=7, alpha=0.7)
        plt.tight_layout(rect=[0,0,1,0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120,
                    facecolor=C['bg'], edgecolor='none',
                    bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        gc.collect()
        return buf
    except Exception as e:
        log.error(f"Performance chart error: {e}")
        return None

# ════════════════════════════════════════════════════════════════
#  AUTO ALERT ENGINE
# ════════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Scans ALL pairs continuously.
    Adaptive interval by session.
    Never stops — off-hours fires with warning.
    Quality filter per subscriber respected.
    Only tracks predictions that were actually sent.
    """

    def __init__(self):
        self.last_alert:  Dict[str, datetime] = {}
        self.cooldown     = timedelta(hours=3)
        self.off_hr_cd    = timedelta(hours=6)
        self._scan_count  = 0

    def _can_alert(self, pair: str, tf: str,
                   session: SessionData) -> bool:
        key  = f"{pair}_{tf}"
        last = self.last_alert.get(key)
        if last is None:
            return True
        cd = (self.off_hr_cd
              if session.session_type == SessionType.OFF_HOURS
              else self.cooldown)
        return datetime.now(timezone.utc) - last > cd

    def _mark(self, pair: str, tf: str):
        self.last_alert[f"{pair}_{tf}"] = datetime.now(timezone.utc)

    async def scan_all(self, app) -> int:
        now     = datetime.now(timezone.utc)
        session = SESSION_ENGINE.get_session(now)
        active  = SUB_MANAGER.all_active()
        if not active:
            return 0

        scan_tfs = (["M15", "H1"] if session.is_kill_zone
                    else ["H1"])

        min_conf = SESSION_ENGINE.get_min_confidence(session)
        sent     = 0
        self._scan_count += 1

        log.info(
            f"Alert scan #{self._scan_count} | "
            f"{session.session_name} | "
            f"TFs:{scan_tfs} | MinConf:{min_conf:.0f}%")

        async with aiohttp.ClientSession() as http_session:
            for pair in ALL_PAIRS:
                for tf in scan_tfs:
                    if not self._can_alert(pair, tf, session):
                        continue

                    # Who wants this pair?
                    recipients = [
                        (cid, sub) for cid, sub in active
                        if SUB_MANAGER.wants_pair(cid, pair)
                    ]
                    if not recipients:
                        continue

                    try:
                        dd = await full_analysis(
                            http_session, pair, tf)
                        of = dd["of"]
                        af = dd["af"]

                        result = await QB.generate(
                            dd["candles"], of, af, dd["vp"],
                            dd["ob"], dd["pb"], dd["cs"], pair,
                            dd["daily_candles"],
                            dd["m15_candles"], dd["h4_candles"])

                        if not result.get("has_setup"):
                            continue

                        signal = result["signal"]
                        conf   = result["confidence"]
                        regime = result["regime"]

                        if conf < min_conf:
                            continue
                        if signal.rr_ratio < MIN_RR_RATIO:
                            continue
                        # Extra gate: off-hours requires A grade minimum
                        if (session.session_type == SessionType.OFF_HOURS
                                and signal.quality == SetupQuality.C):
                            continue

                        # Double-check geometry one final time
                        final_val = validate_signal_geometry(
                            signal.direction, signal.entry,
                            signal.target, signal.stop, pair)
                        if not final_val.is_valid:
                            log.warning(
                                f"Final geometry check failed "
                                f"{pair}: {final_val.reason}")
                            continue

                        self._mark(pair, tf)

                        sim = PATTERN_MEMORY.find_similar(
                            pair, signal.direction,
                            signal.strategy.value, af, of,
                            regime, session)

                        # Generate unique trade ID
                        trade_id = generate_trade_id(pair, signal.direction)
                        result["trade_id"] = trade_id

                        msg = fmt_signal_message(
                            pair, tf, result, of, af,
                            session, regime, sim)

                        # Determine actual recipients (quality filter)
                        actual_recipients = [
                            (cid, sub) for cid, sub in recipients
                            if SUB_MANAGER.meets_quality(
                                cid, signal.quality.value)
                        ]

                        if not actual_recipients:
                            continue

                        # Create prediction object
                        pid     = str(uuid.uuid4())
                        now_iso = now.isoformat()
                        aligned = sum(
                            1 for f in signal.factors
                            if f.direction == (
                                "BULLISH" if signal.direction=="LONG"
                                else "BEARISH"))

                        pobj = QuantPrediction(
                            trade_id           = trade_id,
                            prediction_id      = pid,
                            pair               = pair,
                            timeframe          = tf,
                            timestamp          = now_iso,
                            current_price      = of.price,
                            direction          = signal.direction,
                            target_price       = signal.target,
                            invalidation_price = signal.stop,
                            breakeven_price    = signal.breakeven_price,
                            trailing_price     = signal.trailing_price,
                            confidence         = conf,
                            calibrated_prob    = result.get(
                                "calibrated_prob", 0.5),
                            quality            = signal.quality.value,
                            strategy           = signal.strategy.value,
                            reasons            = signal.reasons,
                            key_levels         = result.get("key_levels",[]),
                            factors_aligned    = aligned,
                            features           = signal.features,
                            rr_ratio           = signal.rr_ratio,
                            reward_pips        = signal.reward_pips,
                            risk_pips          = signal.risk_pips,
                            status             = "ACTIVE",
                            chat_ids           = [],  # filled below
                            ml_confidence      = result.get("ml_prob",0.0),
                            ml_used            = ML_ENGINE.is_trained,
                            regime_at_signal   = regime.regime.value,
                            session_at_signal  = session.session_name,
                            was_sent_to_users  = False,  # set True if sent
                            sent_quality       = signal.quality.value,
                        )

                        # Generate chart once
                        ch = None
                        try:
                            ch = chart_signal(
                                dd["candles"], pair, result,
                                of, af, regime, result.get("ict"))
                        except Exception as e:
                            log.error(f"Chart {pair}: {e}")

                        # Send to recipients
                        sent_chat_ids = []
                        for chat_id, sub in actual_recipients:
                            cap = (
                                f"{asset_emoji(pair)} "
                                f"<b>{pair_display(pair)}</b> | "
                                f"{'🟢' if signal.direction=='LONG' else '🔴'} "
                                f"{signal.direction} | "
                                f"{signal.quality.value} | "
                                f"🆔 {trade_id}")
                            ok_p = False
                            if ch:
                                ch.seek(0)
                                ok_p = await send_photo(
                                    app.bot, chat_id, ch,
                                    caption=cap,
                                    parse_mode="HTML") is not None
                            ok_m = await send_msg(
                                app.bot, chat_id, msg,
                                parse_mode="HTML") is not None
                            if ok_p or ok_m:
                                sent_chat_ids.append(chat_id)
                                SUB_MANAGER.record_alert(chat_id)
                                sent += 1

                        if sent_chat_ids:
                            pobj.chat_ids         = sent_chat_ids
                            pobj.was_sent_to_users = True

                        # Always track prediction in system
                        QB.add_to_history(pobj)
                        PT.add(pobj)
                        PATTERN_MEMORY.record(
                            pobj, of, af, regime, session)

                        # Add to ML data store
                        ML_DATA_STORE.add({
                            "prediction_id": pid,
                            "features":      signal.features,
                            "outcome":       "",
                            "pips_gained":   0.0,
                            "strategy":      signal.strategy.value,
                            "session":       session.session_name,
                            "regime":        regime.regime.value,
                        })

                        await asyncio.sleep(1.5)

                    except Exception as e:
                        log.error(f"Alert scan {pair} {tf}: {e}")
                        continue

        if sent > 0:
            log.info(f"Alerts sent: {sent}")
        return sent


AE = AlertEngine()


async def alert_loop(app):
    log.info("Alert scanner started")
    while True:
        try:
            session  = SESSION_ENGINE.get_session()
            interval = SESSION_ENGINE.get_scan_interval(session)
            n        = await AE.scan_all(app)
            if n > 0:
                log.info(f"Alerts fired: {n}")
            await asyncio.sleep(interval)
        except Exception as e:
            log.error(f"Alert loop: {e}")
            await asyncio.sleep(60)


async def monitor_loop(app):
    log.info("Monitor loop started")
    while True:
        try:
            notes = await PT.check(app.bot)
            for n in notes:
                pred = n["pred"]
                # Safety check: only notify if was_sent_to_users
                if not pred.was_sent_to_users:
                    continue
                msg = fmt_outcome_notification(pred, n["type"], n["cp"])
                # Send to ALL chat_ids that received the original signal
                for chat_id in pred.chat_ids:
                    await send_msg(app.bot, chat_id, msg,
                                   parse_mode="HTML")
                    await asyncio.sleep(0.3)
                # Record in performance engine
                if n["type"] in ("TARGET_HIT", "STOP_HIT"):
                    PERF_ENGINE.record(pred)
            await asyncio.sleep(MONITORING_INTERVAL)
        except Exception as e:
            log.error(f"Monitor loop: {e}")
            await asyncio.sleep(MONITORING_INTERVAL)


async def daily_summary_loop(app):
    log.info("Daily summary loop started")
    while True:
        try:
            now    = datetime.now(timezone.utc)
            target = now.replace(
                hour=21, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            await asyncio.sleep(wait)

            exp      = PERF_ENGINE.get_expectancy()
            ml_stats = ML_ENGINE.get_stats()
            strat_b  = PERF_ENGINE.get_strategy_breakdown()
            mode, risk_pct, risk_msg = PERF_ENGINE.get_risk_mode()
            session  = SESSION_ENGINE.get_session()
            wat      = SESSION_ENGINE.wat_time()

            best_strat = (max(strat_b.items(),
                             key=lambda x: x[1]["expectancy"],
                             default=(None, {}))[0])

            msg = (
                f"{'='*35}\n"
                f"📊 <b>DAILY PERFORMANCE SUMMARY</b>\n"
                f"{'='*35}\n\n"
                f"🕐 {wat}\n\n"
                f"<b>Resolved Today:</b>\n"
                f"├ Total:      {exp['total']}\n"
                f"├ Win Rate:   {exp['win_rate']:.1%}\n"
                f"├ Expectancy: {exp['expectancy']:+.1f} pips/trade\n"
                f"└ Avg MFE:    {exp['avg_mfe']:.1f}p\n\n"
            )
            if best_strat:
                msg += (
                    f"🏆 Best: <b>{best_strat[:25]}</b>\n\n")
            msg += (
                f"<b>🧠 ML:</b> "
                f"{'✅ Active' if ml_stats['is_trained'] else '⏳ Accumulating'} "
                f"({ml_stats['n_samples']}/{ML_MIN_SAMPLES})\n\n"
                f"<b>Risk:</b> {risk_msg}\n\n"
                f"<b>Tomorrow:</b>\n"
                f"London KZ: 08:00–11:00 WAT\n"
                f"NY KZ: 13:00–16:00 WAT\n\n"
                f"<i>Trade the plan. Protect the capital.</i>"
            )

            active = SUB_MANAGER.all_active()
            for chat_id, _ in active:
                await send_msg(app.bot, chat_id, msg,
                               parse_mode="HTML")
                await asyncio.sleep(0.4)

        except Exception as e:
            log.error(f"Daily summary: {e}")
            await asyncio.sleep(3600)

# ════════════════════════════════════════════════════════════════
#  KEYBOARDS
# ════════════════════════════════════════════════════════════════

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎯 Get Signal",     callback_data="predict_menu"),
         InlineKeyboardButton("📊 Analyze",        callback_data="analyze_menu")],
        [InlineKeyboardButton("🧠 Market Summary", callback_data="summary_menu")],
        [InlineKeyboardButton("🔔 Auto Alerts",    callback_data="alerts_menu"),
         InlineKeyboardButton("💱 Strength",       callback_data="strength")],
        [InlineKeyboardButton("🌍 Overview",       callback_data="overview"),
         InlineKeyboardButton("🕐 Sessions",       callback_data="sessions")],
        [InlineKeyboardButton("📊 Performance",    callback_data="performance"),
         InlineKeyboardButton("📚 Guide",          callback_data="guide")],
        [InlineKeyboardButton("🥇 Gold",           callback_data="predict_pair_XAU_USD"),
         InlineKeyboardButton("🥈 Silver",         callback_data="predict_pair_XAG_USD"),
         InlineKeyboardButton("📈 NASDAQ",         callback_data="predict_pair_NAS100_USD")],
    ])


def kb_pairs(mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []
    kb.append([InlineKeyboardButton("── Majors ──", callback_data="none")])
    row: List = []
    for p in ASSET_CATEGORIES["forex_major"]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3:
            kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton("── Crosses ──", callback_data="none")])
    row = []
    for p in ASSET_CATEGORIES["forex_cross"][:12]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3:
            kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton(
        "More Crosses ▼", callback_data=f"{mode}_more")])
    kb.append([InlineKeyboardButton("── Metals/Index ──", callback_data="none")])
    kb.append([
        InlineKeyboardButton("🥇 XAU/USD", callback_data=f"{mode}_pair_XAU_USD"),
        InlineKeyboardButton("🥈 XAG/USD", callback_data=f"{mode}_pair_XAG_USD"),
        InlineKeyboardButton("📈 NAS100",  callback_data=f"{mode}_pair_NAS100_USD"),
    ])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(kb)


def kb_more_crosses(mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []; row: List = []
    for p in ASSET_CATEGORIES["forex_cross"][12:]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{mode}_pair_{p}"))
        if len(row) == 3:
            kb.append(row); row = []
    if row: kb.append(row)
    kb.append([InlineKeyboardButton(
        "◀️ Back", callback_data=f"{mode}_menu")])
    return InlineKeyboardMarkup(kb)


def kb_tf(pair: str, mode: str) -> InlineKeyboardMarkup:
    kb: List[List] = []
    for tf, info in TIMEFRAMES.items():
        kb.append([InlineKeyboardButton(
            f"{info['emoji']} {info['label']}",
            callback_data=f"{mode}_pair_{pair}_{tf}")])
    kb.append([InlineKeyboardButton(
        "◀️ Back", callback_data=f"{mode}_menu")])
    return InlineKeyboardMarkup(kb)


def kb_alerts(chat_id: int) -> InlineKeyboardMarkup:
    is_sub = SUB_MANAGER.is_active(chat_id)
    kb: List[List] = []
    if is_sub:
        sub   = SUB_MANAGER.get(chat_id) or {}
        all_p = sub.get("all_pairs", False)
        kb.append([InlineKeyboardButton(
            "✅ ALL Pairs ON" if all_p else "📋 All Pairs: OFF",
            callback_data="al_toggle_all")])
        kb.append([
            InlineKeyboardButton("➕ Add Pair",    callback_data="al_add"),
            InlineKeyboardButton("➖ Remove Pair", callback_data="al_rem"),
        ])
        kb.append([
            InlineKeyboardButton("⚡ Timeframe",   callback_data="al_tf"),
            InlineKeyboardButton("⭐ Min Quality", callback_data="al_quality"),
        ])
        kb.append([InlineKeyboardButton(
            "📊 My Stats", callback_data="al_stats")])
        kb.append([InlineKeyboardButton(
            "🔕 Unsubscribe", callback_data="al_unsub")])
    else:
        kb.append([InlineKeyboardButton(
            "🔔 Subscribe to Auto Alerts",
            callback_data="al_sub")])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(kb)


def kb_alert_pairs(action: str) -> InlineKeyboardMarkup:
    kb: List[List] = []; row: List = []
    for p in ASSET_CATEGORIES["forex_major"]:
        row.append(InlineKeyboardButton(
            pair_display(p), callback_data=f"{action}_{p}"))
        if len(row) == 3:
            kb.append(row); row = []
    if row: kb.append(row)
    kb.append([
        InlineKeyboardButton("🥇 Gold",   callback_data=f"{action}_XAU_USD"),
        InlineKeyboardButton("📈 NAS100", callback_data=f"{action}_NAS100_USD"),
    ])
    kb.append([InlineKeyboardButton("◀️ Back", callback_data="alerts_menu")])
    return InlineKeyboardMarkup(kb)


def kb_quality() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 A+ Only",         callback_data="al_setq_A+")],
        [InlineKeyboardButton("✅ A and above",      callback_data="al_setq_A")],
        [InlineKeyboardButton("⚡ B and above",      callback_data="al_setq_B")],
        [InlineKeyboardButton("⚠️ All (C and above)",callback_data="al_setq_C")],
        [InlineKeyboardButton("◀️ Back", callback_data="alerts_menu")],
    ])


def parse_pair_tf_cb(data:   str,
                     prefix: str
                     ) -> Tuple[Optional[str], Optional[str]]:
    remainder = data[len(prefix):]
    for pair in sorted(ALL_PAIRS, key=len, reverse=True):
        if remainder.startswith(pair + "_"):
            tf = remainder[len(pair)+1:]
            if tf in TIMEFRAMES:
                return pair, tf
            return None, None
        elif remainder == pair:
            return pair, None
    return None, None

# ════════════════════════════════════════════════════════════════
#  TELEGRAM COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    name    = esc(user.first_name) if user else "Trader"
    ml_s    = ML_ENGINE.get_stats()
    sess    = SESSION_ENGINE.get_session()
    wat     = sess.wat_time

    ml_st = (
        f"✅ Active ({ml_s['n_samples']} samples, "
        f"OOS:{ml_s['oos_accuracy']:.0%})"
        if ml_s["is_trained"] else
        f"⏳ Learning ({ml_s['n_samples']}/{ML_MIN_SAMPLES} resolved, "
        f"{ml_s['n_total']} tracked)")

    text = (
        f"🎯 <b>Welcome, {name}!</b>\n\n"
        f"<b>FOREX QUANT v8.1</b> — The Beast\n\n"
        f"{'─'*30}\n\n"
        f"<b>Now:</b> {wat}\n"
        f"<b>Session:</b> {sess.session_name}"
        f"{' | 🎯 ' + sess.kill_zone_name if sess.is_kill_zone else ''}\n"
        f"<b>AMD Phase:</b> {sess.amd_phase}\n\n"
        f"{'─'*30}\n\n"
        f"<b>v8.1 Upgrades:</b>\n"
        f"✅ R/R validated — shown correctly as reward:risk\n"
        f"✅ Unique Trade IDs (FQ-EURUSD-BUY-...)\n"
        f"✅ Breakeven alerts (50% to target)\n"
        f"✅ Trailing stop alerts (75% to target)\n"
        f"✅ Outcome alerts ONLY for sent signals\n"
        f"✅ ML persists across restarts\n"
        f"✅ HTF bias enforcement\n"
        f"✅ Min target pips per instrument\n"
        f"✅ Signal geometry validation\n\n"
        f"<b>🧠 ML:</b> {ml_st}\n"
        f"<b>📊 Strategies:</b> {len(list(StrategyType))}\n"
        f"<b>💱 Instruments:</b> {len(ALL_PAIRS)}\n\n"
        f"<i>Signals auto-sent. Use /alerts to subscribe.</i>"
    )
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=kb_main())


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎯 <b>SELECT PAIR FOR SIGNAL</b>",
        parse_mode="HTML", reply_markup=kb_pairs("predict"))


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 <b>SELECT PAIR TO ANALYZE</b>",
        parse_mode="HTML", reply_markup=kb_pairs("analyze"))


async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 <b>SELECT PAIR FOR SUMMARY</b>",
        parse_mode="HTML", reply_markup=kb_pairs("summary"))


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    is_sub  = SUB_MANAGER.is_active(chat_id)
    if is_sub:
        sub   = SUB_MANAGER.get(chat_id) or {}
        pairs = sub.get("pairs", [])
        all_p = sub.get("all_pairs", False)
        text  = (
            f"🔔 <b>AUTO ALERTS — ACTIVE</b>\n\n"
            f"Watching: "
            f"{'ALL ' + str(len(ALL_PAIRS)) + ' pairs' if all_p else ', '.join(pair_display(p) for p in pairs) or 'None'}\n"
            f"TF: {sub.get('timeframe','H1')} | "
            f"Min Quality: {sub.get('min_quality','B')}\n"
            f"Alerts received: {sub.get('alert_count',0)}\n\n"
            f"<b>Alert System:</b>\n"
            f"• Session-adjusted confidence\n"
            f"• R/R ≥ 1:{MIN_RR_RATIO}\n"
            f"• Geometry validated before sending\n"
            f"• Breakeven alert at 50% of target\n"
            f"• Trailing alert at 75% of target\n"
            f"• TP/SL only for YOUR signals"
        )
    else:
        text = (
            f"🔔 <b>AUTO ALERTS</b>\n\n"
            f"Get notified when institutional setups appear.\n\n"
            f"<b>What you get:</b>\n"
            f"• Signal with unique Trade ID\n"
            f"• Entry, Target, Stop (validated geometry)\n"
            f"• Breakeven alert (50% to target)\n"
            f"• Trailing stop alert (75% to target)\n"
            f"• TP/SL notifications for YOUR trades only\n"
            f"• Daily performance summary\n\n"
            f"<i>Tap Subscribe.</i>"
        )
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=kb_alerts(chat_id))


async def cmd_performance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg_l = await update.message.reply_text(
        "⏳ Generating...", parse_mode="HTML")
    report = PERF_ENGINE.format_report()
    bt     = HISTORICAL_BT.format_report(QB.history)
    ch     = chart_performance(QB.history)
    await del_msg(msg_l)
    if ch:
        await send_photo(
            ctx.bot, update.effective_chat.id, ch,
            caption="📊 <b>Performance Analytics</b>",
            parse_mode="HTML")
    await update.message.reply_text(
        report, parse_mode="HTML", reply_markup=kb_main())
    await update.message.reply_text(
        bt, parse_mode="HTML", reply_markup=kb_main())


async def cmd_sessions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        SA.schedule(), parse_mode="HTML", reply_markup=kb_main())


async def cmd_strength(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg_l = await update.message.reply_text(
        "⏳ Calculating...", parse_mode="HTML")
    try:
        async with aiohttp.ClientSession() as session:
            st  = await STRENGTH_CACHE.get(session)
        ch  = chart_strength(st)
        sc  = sorted(st.values(),
                     key=lambda x: x.strength, reverse=True)
        cap = (
            f"💪 <b>Currency Strength (24H)</b>\n\n"
            f"🟢 Strongest: {sc[0].currency} "
            f"({sc[0].strength:+.2f}%)\n"
            f"🔴 Weakest: {sc[-1].currency} "
            f"({sc[-1].strength:+.2f}%)\n\n"
            f"WAT: {SESSION_ENGINE.wat_time()}")
        await del_msg(msg_l)
        await send_photo(ctx.bot, update.effective_chat.id, ch,
                         caption=cap, parse_mode="HTML",
                         reply_markup=kb_main())
    except Exception as e:
        await msg_l.edit_text(f"❌ Error: {str(e)[:100]}")


async def cmd_overview(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg_l = await update.message.reply_text("⏳ Loading...")
    try:
        async with aiohttp.ClientSession() as http_session:
            st   = await STRENGTH_CACHE.get(http_session)
        ch   = chart_strength(st)
        sc   = sorted(st.values(),
                      key=lambda x: x.strength, reverse=True)
        sess = SESSION_ENGINE.get_session()
        cap  = (
            f"🌍 <b>MARKET OVERVIEW</b>\n\n"
            f"🕐 {sess.wat_time}\n"
            f"Session: {sess.session_name}"
            f"{' | 🎯' if sess.is_kill_zone else ''}\n"
            f"AMD: {sess.amd_phase}\n\n"
            f"🟢 {sc[0].currency} ({sc[0].strength:+.2f}%)\n"
            f"🔴 {sc[-1].currency} ({sc[-1].strength:+.2f}%)\n"
            f"💡 Best pair: {sc[0].currency}/{sc[-1].currency}"
        )
        await del_msg(msg_l)
        await send_photo(ctx.bot, update.effective_chat.id, ch,
                         caption=cap, parse_mode="HTML",
                         reply_markup=kb_main())
    except Exception as e:
        await msg_l.edit_text(f"❌ {str(e)[:100]}")


async def cmd_guide(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ml_s = ML_ENGINE.get_stats()
    text = (
        f"📚 <b>FOREX QUANT v8.1 GUIDE</b>\n\n"
        f"<b>🔧 Key Fixes in v8.1:</b>\n"
        f"• R/R = Reward÷Risk (e.g. 1:2 means risking 1 to make 2)\n"
        f"• Geometry validation before every signal\n"
        f"• Min target: Forex {MIN_TARGET_PIPS_FOREX}p, "
        f"Gold {MIN_TARGET_PIPS_GOLD}p\n"
        f"• Min R/R: 1:{MIN_RR_RATIO}\n"
        f"• TP/SL alerts only for signals sent to YOU\n"
        f"• Trade IDs for easy reference\n\n"
        f"<b>🏛️ ICT:</b> BOS·CHOCH·OB·FVG·Sweep·AMD·OTE\n"
        f"<b>📐 Math:</b> Hurst·Entropy·Z-Score·ADR·Fibonacci\n"
        f"<b>🤖 ML:</b> Meta-labeling · Purged WF · Calibrated\n"
        f"Status: {'✅ Active' if ml_s['is_trained'] else '⏳ Learning'} "
        f"({ml_s['n_samples']}/{ML_MIN_SAMPLES})\n\n"
        f"<b>📌 Trade Management:</b>\n"
        f"• Breakeven alert: 50% of target reached\n"
        f"• Trailing stop alert: 75% of target reached\n"
        f"• TP hit: Celebrate + note MAE/MFE\n"
        f"• SL hit: Capital protected. Move on.\n\n"
        f"<b>⚠️ Risk:</b> 1-2% max per trade. Always."
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
        # ── Navigation ───────────────────────────────────────────
        if data == "main_menu":
            await del_msg(query.message)
            sess = SESSION_ENGINE.get_session()
            await send_msg(
                ctx.bot, chat_id,
                f"🎯 <b>FOREX QUANT v8.1</b>\n"
                f"🕐 {sess.wat_time} | {sess.session_name}",
                parse_mode="HTML", reply_markup=kb_main())
            return

        if data in ("predict_menu","analyze_menu","summary_menu"):
            mode = data.replace("_menu","")
            ic   = ("🎯" if mode=="predict" else
                    "📊" if mode=="analyze" else "🧠")
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           f"{ic} <b>SELECT PAIR</b>",
                           parse_mode="HTML",
                           reply_markup=kb_pairs(mode))
            return

        if data.endswith("_more"):
            mode = data.replace("_more","")
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "<b>More Cross Pairs:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_more_crosses(mode))
            return

        if data == "sessions":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id, SA.schedule(),
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data == "performance":
            loading = await send_msg(
                ctx.bot, chat_id, "⏳ Generating...",
                parse_mode="HTML")
            report = PERF_ENGINE.format_report()
            ch     = chart_performance(QB.history)
            if loading: await del_msg(loading)
            if ch:
                await send_photo(ctx.bot, chat_id, ch,
                                 caption="📊 <b>Performance</b>",
                                 parse_mode="HTML")
            await send_msg(ctx.bot, chat_id, report,
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data == "overview":
            loading = await send_msg(
                ctx.bot, chat_id, "⏳ Loading...",
                parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as http_session:
                    st  = await STRENGTH_CACHE.get(http_session)
                ch  = chart_strength(st)
                sc  = sorted(st.values(),
                             key=lambda x: x.strength, reverse=True)
                sess = SESSION_ENGINE.get_session()
                cap  = (
                    f"🌍 <b>OVERVIEW</b>\n"
                    f"🕐 {sess.wat_time} | {sess.session_name}\n"
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
            loading = await send_msg(
                ctx.bot, chat_id, "⏳ Calculating...",
                parse_mode="HTML")
            try:
                async with aiohttp.ClientSession() as http_session:
                    st  = await STRENGTH_CACHE.get(http_session)
                ch  = chart_strength(st)
                sc  = sorted(st.values(),
                             key=lambda x: x.strength, reverse=True)
                cap = (
                    f"💪 <b>Currency Strength</b>\n"
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
                f"📚 <b>Quant v8.1</b>\n\n"
                f"R/R fixed ✅ | Trade IDs ✅ | "
                f"Breakeven alerts ✅\n"
                f"ML: {'Active' if ml_s['is_trained'] else 'Learning'} "
                f"({ml_s['n_samples']}/{ML_MIN_SAMPLES})\n\n"
                f"Use /guide for full details.",
                parse_mode="HTML", reply_markup=kb_main())
            return

        # ── Alert management ─────────────────────────────────────
        if data == "alerts_menu":
            await del_msg(query.message)
            is_sub = SUB_MANAGER.is_active(chat_id)
            sub    = SUB_MANAGER.get(chat_id) or {}
            if is_sub:
                pairs = sub.get("pairs", [])
                all_p = sub.get("all_pairs", False)
                text  = (
                    f"🔔 <b>ALERTS ACTIVE</b>\n\n"
                    f"Watching: "
                    f"{'ALL pairs' if all_p else ', '.join(pair_display(p) for p in pairs) or 'None'}\n"
                    f"TF: {sub.get('timeframe','H1')} | "
                    f"Min Q: {sub.get('min_quality','B')}\n"
                    f"Sent: {sub.get('alert_count',0)}")
            else:
                text = ("🔔 <b>AUTO ALERTS</b>\n\n"
                        "<i>Subscribe to receive signals.</i>")
            await send_msg(ctx.bot, chat_id, text,
                           parse_mode="HTML",
                           reply_markup=kb_alerts(chat_id))
            return

        if data == "al_sub":
            SUB_MANAGER.subscribe(chat_id, [], "H1")
            await send_msg(
                ctx.bot, chat_id,
                "✅ <b>Subscribed!</b>\n\n"
                "Toggle 'ALL Pairs' to receive every signal, "
                "or add specific pairs.",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data == "al_unsub":
            SUB_MANAGER.unsubscribe(chat_id)
            await send_msg(ctx.bot, chat_id,
                           "🔕 <b>Unsubscribed.</b>",
                           parse_mode="HTML", reply_markup=kb_main())
            return

        if data == "al_toggle_all":
            sub  = SUB_MANAGER.get(chat_id) or {}
            curr = sub.get("all_pairs", False)
            SUB_MANAGER.set_all_pairs(chat_id, not curr)
            await send_msg(
                ctx.bot, chat_id,
                f"{'✅ ALL pairs ON — every signal delivered.' if not curr else '📋 Specific pairs mode.'}",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data == "al_add":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "➕ <b>SELECT PAIR TO ADD:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_alert_pairs("aladd"))
            return

        if data == "al_rem":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "➖ <b>SELECT PAIR TO REMOVE:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_alert_pairs("alrem"))
            return

        if data == "al_tf":
            await del_msg(query.message)
            await send_msg(
                ctx.bot, chat_id,
                "⚡ <b>SELECT ALERT TIMEFRAME:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton(
                        "⚡ M15", callback_data="al_settf_M15"),
                     InlineKeyboardButton(
                        "📊 H1",  callback_data="al_settf_H1")],
                    [InlineKeyboardButton(
                        "📈 H4",  callback_data="al_settf_H4")],
                    [InlineKeyboardButton(
                        "◀️ Back", callback_data="alerts_menu")],
                ]))
            return

        if data == "al_quality":
            await del_msg(query.message)
            await send_msg(ctx.bot, chat_id,
                           "⭐ <b>MINIMUM SIGNAL QUALITY:</b>",
                           parse_mode="HTML",
                           reply_markup=kb_quality())
            return

        if data == "al_stats":
            sub = SUB_MANAGER.get(chat_id) or {}
            await send_msg(
                ctx.bot, chat_id,
                f"📊 <b>Your Alert Stats</b>\n\n"
                f"Total alerts: {sub.get('alert_count',0)}\n"
                f"Last alert: "
                f"{(sub.get('last_alert','Never') or 'Never')[:19]}\n\n"
                f"<i>TP/SL/Breakeven only for signals sent to you.</i>",
                parse_mode="HTML", reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("aladd_"):
            pair = data[6:]
            if pair in ALL_PAIRS:
                SUB_MANAGER.add_pair(chat_id, pair)
                sub   = SUB_MANAGER.get(chat_id) or {}
                pairs = sub.get("pairs",[])
                await send_msg(
                    ctx.bot, chat_id,
                    f"✅ Added {pair_display(pair)}\n\n"
                    f"Watching: "
                    f"{', '.join(pair_display(p) for p in pairs)}",
                    parse_mode="HTML",
                    reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("alrem_"):
            pair = data[6:]
            if pair in ALL_PAIRS:
                SUB_MANAGER.remove_pair(chat_id, pair)
                sub   = SUB_MANAGER.get(chat_id) or {}
                pairs = sub.get("pairs",[])
                await send_msg(
                    ctx.bot, chat_id,
                    f"✅ Removed {pair_display(pair)}\n\n"
                    f"Watching: "
                    f"{', '.join(pair_display(p) for p in pairs) or 'None'}",
                    parse_mode="HTML",
                    reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("al_settf_"):
            tf = data[9:]
            if tf in TIMEFRAMES:
                SUB_MANAGER.set_tf(chat_id, tf)
                await send_msg(ctx.bot, chat_id,
                               f"✅ Alert TF: <b>{tf}</b>",
                               parse_mode="HTML",
                               reply_markup=kb_alerts(chat_id))
            return

        if data.startswith("al_setq_"):
            q = data[8:]
            SUB_MANAGER.set_min_quality(chat_id, q)
            await send_msg(ctx.bot, chat_id,
                           f"✅ Min quality: <b>{q}</b>",
                           parse_mode="HTML",
                           reply_markup=kb_alerts(chat_id))
            return

        # ── Pair → TF → Analysis ─────────────────────────────────
        for mode in ("predict","analyze","summary"):
            prefix = f"{mode}_pair_"
            if not data.startswith(prefix):
                continue
            pair, tf = parse_pair_tf_cb(data, prefix)

            if pair and not tf:
                # Show TF selector
                await del_msg(query.message)
                await send_msg(
                    ctx.bot, chat_id,
                    f"{asset_emoji(pair)} <b>{pair_display(pair)}</b>"
                    f"\n\n<i>Select timeframe:</i>",
                    parse_mode="HTML",
                    reply_markup=kb_tf(pair, mode))
                return

            if pair and tf:
                await del_msg(query.message)
                loading = await send_msg(
                    ctx.bot, chat_id,
                    f"⏳ <b>Analyzing {pair_display(pair)} {tf}...</b>",
                    parse_mode="HTML")
                try:
                    async with aiohttp.ClientSession() as http_session:
                        dd = await full_analysis(http_session, pair, tf)

                    of      = dd["of"]
                    af      = dd["af"]
                    session = SESSION_ENGINE.get_session()
                    regime  = REGIME_ENGINE.detect(
                        dd["candles"], dd["daily_candles"], pair)
                    ict     = ICT_ENGINE.full_analysis(
                        dd["candles"], dd["daily_candles"],
                        session, pair)

                    if mode == "predict":
                        result = await QB.generate(
                            dd["candles"], of, af, dd["vp"],
                            dd["ob"], dd["pb"], dd["cs"], pair,
                            dd["daily_candles"],
                            dd["m15_candles"], dd["h4_candles"])

                        if loading: await del_msg(loading)

                        if result.get("has_setup"):
                            signal = result["signal"]
                            sim    = PATTERN_MEMORY.find_similar(
                                pair, signal.direction,
                                signal.strategy.value, af, of,
                                result["regime"], session)

                            # Generate trade ID
                            trade_id = generate_trade_id(
                                pair, signal.direction)
                            result["trade_id"] = trade_id

                            msg = fmt_signal_message(
                                pair, tf, result, of, af,
                                session, result["regime"], sim)

                            # Chart
                            ch = None
                            try:
                                ch = chart_signal(
                                    dd["candles"], pair, result,
                                    of, af, result["regime"], ict)
                            except Exception:
                                pass

                            if ch:
                                await send_photo(
                                    ctx.bot, chat_id, ch,
                                    caption=(
                                        f"{asset_emoji(pair)} "
                                        f"<b>{pair_display(pair)}</b> | "
                                        f"{'🟢' if signal.direction=='LONG' else '🔴'} "
                                        f"{signal.direction} | "
                                        f"🆔 {trade_id}"),
                                    parse_mode="HTML")

                            await send_msg(ctx.bot, chat_id, msg,
                                           parse_mode="HTML",
                                           reply_markup=kb_main())
                            await send_msg(ctx.bot, chat_id,
                                           SA.advice(pair, session),
                                           parse_mode="HTML")

                            # Track prediction
                            now_dt  = datetime.now(timezone.utc)
                            pid     = str(uuid.uuid4())
                            aligned = sum(
                                1 for f in signal.factors
                                if f.direction == (
                                    "BULLISH" if signal.direction=="LONG"
                                    else "BEARISH"))

                            pobj = QuantPrediction(
                                trade_id           = trade_id,
                                prediction_id      = pid,
                                pair               = pair,
                                timeframe          = tf,
                                timestamp          = now_dt.isoformat(),
                                current_price      = of.price,
                                direction          = signal.direction,
                                target_price       = signal.target,
                                invalidation_price = signal.stop,
                                breakeven_price    = signal.breakeven_price,
                                trailing_price     = signal.trailing_price,
                                confidence         = result["confidence"],
                                calibrated_prob    = result.get(
                                    "calibrated_prob", 0.5),
                                quality            = signal.quality.value,
                                strategy           = signal.strategy.value,
                                reasons            = signal.reasons,
                                key_levels         = result.get(
                                    "key_levels",[]),
                                factors_aligned    = aligned,
                                features           = signal.features,
                                rr_ratio           = signal.rr_ratio,
                                reward_pips        = signal.reward_pips,
                                risk_pips          = signal.risk_pips,
                                status             = "ACTIVE",
                                chat_ids           = [chat_id],
                                ml_confidence      = result.get("ml_prob",0.0),
                                ml_used            = ML_ENGINE.is_trained,
                                regime_at_signal   = result["regime"].regime.value,
                                session_at_signal  = session.session_name,
                                was_sent_to_users  = True,
                                sent_quality       = signal.quality.value,
                            )
                            QB.add_to_history(pobj)
                            PT.add(pobj)
                            PATTERN_MEMORY.record(
                                pobj, of, af, regime, session)

                            # ML data store
                            ML_DATA_STORE.add({
                                "prediction_id": pid,
                                "features":      signal.features,
                                "outcome":       "",
                                "pips_gained":   0.0,
                                "strategy":      signal.strategy.value,
                                "session":       session.session_name,
                                "regime":        regime.regime.value,
                            })

                        else:
                            factors = []
                            if result.get("signal"):
                                factors = result["signal"].factors
                            msg = fmt_no_setup_message(
                                pair, tf, of,
                                result.get("regime", regime),
                                session, factors)
                            ch = None
                            try:
                                ch = chart_no_setup(dd["candles"], pair)
                            except Exception:
                                pass
                            if ch:
                                await send_photo(
                                    ctx.bot, chat_id, ch,
                                    caption=(
                                        f"{asset_emoji(pair)} "
                                        f"<b>{pair_display(pair)}</b>"
                                        f" | ⚪ NO SETUP"),
                                    parse_mode="HTML")
                            await send_msg(ctx.bot, chat_id, msg,
                                           parse_mode="HTML",
                                           reply_markup=kb_main())

                    elif mode == "analyze":
                        msgs = fmt_analysis_messages(
                            pair, tf, of, af, dd["vp"],
                            dd["ob"], dd["pb"], regime,
                            session, ict)
                        if loading: await del_msg(loading)
                        for m in msgs:
                            if m.strip():
                                await send_msg(ctx.bot, chat_id, m,
                                               parse_mode="HTML")
                                await asyncio.sleep(0.4)
                        await send_msg(ctx.bot, chat_id,
                                       SA.advice(pair, session),
                                       parse_mode="HTML",
                                       reply_markup=kb_main())

                    elif mode == "summary":
                        result = await QB.generate(
                            dd["candles"], of, af, dd["vp"],
                            dd["ob"], dd["pb"], dd["cs"], pair,
                            dd["daily_candles"],
                            dd["m15_candles"], dd["h4_candles"])
                        summ = fmt_market_summary(
                            pair, of, af, dd["vp"], dd["ob"],
                            dd["pb"], regime, session, ict,
                            result if result.get("has_setup") else None)
                        if loading: await del_msg(loading)
                        # Split long messages
                        sections = summ.split("\n\n")
                        chunk    = ""
                        for sec in sections:
                            if len(chunk) + len(sec) > 3800:
                                if chunk.strip():
                                    await send_msg(
                                        ctx.bot, chat_id,
                                        chunk.strip(),
                                        parse_mode="HTML")
                                    await asyncio.sleep(0.4)
                                chunk = sec + "\n\n"
                            else:
                                chunk += sec + "\n\n"
                        if chunk.strip():
                            await send_msg(ctx.bot, chat_id,
                                           chunk.strip(),
                                           parse_mode="HTML")
                        await send_msg(ctx.bot, chat_id,
                                       SA.advice(pair, session),
                                       parse_mode="HTML",
                                       reply_markup=kb_main())

                except Exception as e:
                    log.error(
                        f"Callback {mode} {pair}: {e}", exc_info=True)
                    if loading: await del_msg(loading)
                    await send_msg(
                        ctx.bot, chat_id,
                        f"❌ Analysis error: {str(e)[:100]}",
                        parse_mode="HTML", reply_markup=kb_main())
                return

    except Exception as e:
        log.error(f"Callback error: {e}", exc_info=True)
        await send_msg(ctx.bot, chat_id,
                       "❌ An error occurred. Please try again.",
                       reply_markup=kb_main())


async def on_error(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error(f"Bot error: {ctx.error}", exc_info=True)
    try:
        if update and update.effective_chat:
            await send_msg(ctx.bot, update.effective_chat.id,
                           "❌ Error occurred. Please try again.",
                           reply_markup=kb_main())
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════
#  END OF PART 3
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  FOREX QUANT v8.1 — PART 4 OF 4
#  Background Task Manager, Additional Commands,
#  Startup/Shutdown Hooks, Main Entry Point
#  Complete Production Assembly
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  BACKGROUND TASK MANAGER
# ════════════════════════════════════════════════════════════════

class BackgroundTaskManager:
    """
    Manages all background tasks with automatic restart,
    exponential backoff on errors, and health monitoring.
    """

    def __init__(self):
        self.tasks:        Dict[str, asyncio.Task] = {}
        self.start_times:  Dict[str, datetime]     = {}
        self.error_counts: Dict[str, int]          = {}
        self.running       = False

    async def start_all(self, app):
        """Start all background tasks."""
        self.running = True
        task_defs = [
            ("monitor",        self._run_monitor,       app),
            ("alert_scanner",  self._run_alert_scanner, app),
            ("ml_retrain",     self._run_ml_retrain,    None),
            ("initial_bt",     self._run_initial_bt,    None),
            ("daily_summary",  self._run_daily_summary, app),
            ("regime_cache",   self._run_regime_cache,  None),
            ("health_check",   self._run_health_check,  None),
        ]
        for name, coro_fn, arg in task_defs:
            task = asyncio.create_task(
                self._protected(name, coro_fn, arg),
                name=name)
            self.tasks[name]        = task
            self.start_times[name]  = datetime.now(timezone.utc)
            self.error_counts[name] = 0
            log.info(f"Task started: {name}")

        log.info(f"All {len(task_defs)} background tasks running")

    async def _protected(self, name: str, coro_fn, arg):
        """Restart task on failure with exponential backoff."""
        while self.running:
            try:
                if arg is not None:
                    await coro_fn(arg)
                else:
                    await coro_fn()
                log.info(f"Task '{name}' ended normally — restarting...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                log.info(f"Task '{name}' cancelled.")
                break
            except Exception as e:
                self.error_counts[name] = (
                    self.error_counts.get(name, 0) + 1)
                count = self.error_counts[name]
                wait  = min(300, 10 * (2 ** min(count - 1, 5)))
                log.error(
                    f"Task '{name}' error #{count}: {e}. "
                    f"Restarting in {wait}s...")
                await asyncio.sleep(wait)

    # ── Monitor: check predictions, send outcome alerts ──────────
    async def _run_monitor(self, app):
        log.info("Monitor task active")
        while True:
            try:
                notes = await PT.check(app.bot)
                for n in notes:
                    pred = n["pred"]
                    # CRITICAL: Only notify if signal was sent to users
                    if not pred.was_sent_to_users:
                        log.debug(
                            f"Skipping notification for unsent pred "
                            f"[{pred.trade_id}]")
                        continue

                    msg = fmt_outcome_notification(
                        pred, n["type"], n["cp"])

                    # Send to ALL chat_ids that received the original signal
                    for chat_id in pred.chat_ids:
                        await send_msg(
                            app.bot, chat_id, msg,
                            parse_mode="HTML")
                        await asyncio.sleep(0.3)

                    # Record in performance engine for resolved trades
                    if n["type"] in ("TARGET_HIT", "STOP_HIT"):
                        PERF_ENGINE.record(pred)
                        log.info(
                            f"[{pred.trade_id}] {n['type']} | "
                            f"{pred.pair} | "
                            f"{pred.pips_gained:+.1f}p | "
                            f"MAE:{pred.mae_pips:.1f}p | "
                            f"MFE:{pred.mfe_pips:.1f}p")

                await asyncio.sleep(MONITORING_INTERVAL)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Monitor task error: {e}")
                await asyncio.sleep(MONITORING_INTERVAL)

    # ── Alert scanner: scan all pairs ────────────────────────────
    async def _run_alert_scanner(self, app):
        log.info("Alert scanner task active")
        # Initial delay — let bot fully initialize
        await asyncio.sleep(20)
        while True:
            try:
                session  = SESSION_ENGINE.get_session()
                interval = SESSION_ENGINE.get_scan_interval(session)
                n        = await AE.scan_all(app)
                if n > 0:
                    log.info(
                        f"Scanner: {n} alerts | "
                        f"{session.session_name} | "
                        f"Next in {interval}s")
                else:
                    log.debug(
                        f"Scanner: no setups | "
                        f"{session.session_name} | "
                        f"Next in {interval}s")
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Alert scanner error: {e}")
                await asyncio.sleep(60)

    # ── ML retrain: every 4 hours ─────────────────────────────────
    async def _run_ml_retrain(self):
        log.info("ML retrain task active")
        # First run after 30 minutes
        await asyncio.sleep(1800)
        while True:
            try:
                n_resolved = ML_DATA_STORE.n_samples
                n_total    = ML_DATA_STORE.n_total

                log.info(
                    f"ML retrain check: "
                    f"{n_resolved} resolved / "
                    f"{n_total} total / "
                    f"need {ML_MIN_SAMPLES}")

                if n_resolved >= ML_MIN_SAMPLES:
                    result = ML_ENGINE.train()
                    log.info(f"ML retrain: {result}")
                else:
                    log.info(
                        f"ML waiting: "
                        f"{n_resolved}/{ML_MIN_SAMPLES} "
                        f"resolved trades needed")

                # Also retrain from QB history as backup
                if not ML_ENGINE.is_trained:
                    hist_resolved = [
                        p for p in QB.history
                        if p.get("outcome") in ("WIN","LOSS")
                        and len(p.get("features",[])) == 20
                    ]
                    if len(hist_resolved) >= ML_MIN_SAMPLES:
                        # Feed into ML data store
                        for h in hist_resolved:
                            ML_DATA_STORE.add({
                                "prediction_id": h.get(
                                    "prediction_id",""),
                                "features":      h.get("features",[]),
                                "outcome":       h.get("outcome",""),
                                "pips_gained":   h.get("pips_gained",0),
                                "strategy":      h.get("strategy",""),
                                "session":       h.get("session",""),
                                "regime":        h.get("regime",""),
                            })
                        result = ML_ENGINE.train()
                        log.info(f"ML trained from QB history: {result}")

                # Retrain every 4 hours
                await asyncio.sleep(4 * 3600)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"ML retrain error: {e}")
                await asyncio.sleep(3600)

    # ── Initial backtest: runs on startup ────────────────────────
    async def _run_initial_bt(self):
        log.info("Backtest task active")
        # Wait 45 seconds for everything to initialize
        await asyncio.sleep(45)

        try:
            if HISTORICAL_BT.stale():
                log.info("Running startup backtest...")

                # Phase 1: Major pairs
                await HISTORICAL_BT.run(
                    pairs=ASSET_CATEGORIES["forex_major"],
                    tf="H1")
                await asyncio.sleep(30)

                # Phase 2: Selected crosses
                await HISTORICAL_BT.run(
                    pairs=ASSET_CATEGORIES["forex_cross"][:7],
                    tf="H1")
                await asyncio.sleep(30)

                # Phase 3: Metals + Indices
                await HISTORICAL_BT.run(
                    pairs=(ASSET_CATEGORIES["metals"] +
                           ASSET_CATEGORIES["indices"]),
                    tf="H1")

                log.info("Startup backtest complete")
            else:
                log.info("Backtest data fresh — skipping startup run")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Initial backtest error: {e}")

        # Daily refresh at 02:00 UTC
        while True:
            try:
                now    = datetime.now(timezone.utc)
                target = now.replace(
                    hour=2, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                wait = (target - now).total_seconds()
                log.info(f"Next backtest refresh in {wait/3600:.1f}h")
                await asyncio.sleep(wait)

                log.info("Running daily backtest refresh...")
                await HISTORICAL_BT.run(
                    pairs=ASSET_CATEGORIES["forex_major"],
                    tf="H1")
                await asyncio.sleep(60)
                await HISTORICAL_BT.run(
                    pairs=ASSET_CATEGORIES["forex_cross"][:7],
                    tf="H1")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Daily backtest error: {e}")
                await asyncio.sleep(3600)

    # ── Daily summary: sends at 21:00 UTC (22:00 WAT) ────────────
    async def _run_daily_summary(self, app):
        log.info("Daily summary task active")
        while True:
            try:
                now    = datetime.now(timezone.utc)
                target = now.replace(
                    hour=21, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                wait = (target - now).total_seconds()
                await asyncio.sleep(wait)

                # Build summary
                exp      = PERF_ENGINE.get_expectancy()
                ml_s     = ML_ENGINE.get_stats()
                strat_b  = PERF_ENGINE.get_strategy_breakdown()
                sess_b   = PERF_ENGINE.get_session_breakdown()
                mode, risk_pct, risk_msg = PERF_ENGINE.get_risk_mode()
                session  = SESSION_ENGINE.get_session()
                wat      = SESSION_ENGINE.wat_time()

                best_strat = (
                    max(strat_b.items(),
                        key=lambda x: x[1]["expectancy"],
                        default=(None, {}))[0]
                    if strat_b else None)

                best_sess = (
                    max(sess_b.items(),
                        key=lambda x: x[1]["expectancy"],
                        default=(None, {}))[0]
                    if sess_b else None)

                msg = (
                    f"{'='*35}\n"
                    f"📊 <b>DAILY PERFORMANCE SUMMARY</b>\n"
                    f"{'='*35}\n\n"
                    f"🕐 {wat} | "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"

                    f"<b>Performance (All Time):</b>\n"
                    f"├ Resolved:   <b>{exp['total']}</b> trades\n"
                    f"├ Win Rate:   <b>{exp['win_rate']:.1%}</b>\n"
                    f"├ Avg Win:    <b>+{exp['avg_win']:.1f} pips</b>\n"
                    f"├ Avg Loss:   <b>-{exp['avg_loss']:.1f} pips</b>\n"
                    f"├ Expectancy: <b>{exp['expectancy']:+.1f} pips/trade</b>\n"
                    f"├ Avg MAE:    {exp['avg_mae']:.1f}p\n"
                    f"└ Max DD:     "
                    f"{PERF_ENGINE.data.get('max_drawdown',0):.1f}p\n\n"
                )

                if best_strat:
                    e = strat_b[best_strat]
                    msg += (
                        f"🏆 Best Strategy: "
                        f"<b>{best_strat[:25]}</b>\n"
                        f"   {e['win_rate']:.0%} WR | "
                        f"{e['expectancy']:+.1f}p exp\n\n")

                if best_sess:
                    e = sess_b[best_sess]
                    msg += (
                        f"⏰ Best Session: <b>{best_sess}</b>\n"
                        f"   {e['win_rate']:.0%} WR\n\n")

                msg += (
                    f"<b>🧠 ML Engine:</b>\n"
                    f"{'✅ Active' if ml_s['is_trained'] else '⏳ Accumulating'} "
                    f"| {ml_s['n_samples']}/{ML_MIN_SAMPLES} resolved")

                if ml_s.get("oos_accuracy", 0) > 0:
                    msg += f" | OOS: {ml_s['oos_accuracy']:.0%}"

                msg += (
                    f"\n\n<b>⚠️ Risk Mode:</b> {risk_msg}\n\n"
                    f"<b>Tomorrow's Kill Zones (WAT):</b>\n"
                    f"🎯 London: 08:00–11:00 WAT\n"
                    f"🎯 NY:     13:00–16:00 WAT\n\n"
                    f"<i>Trade the plan. Protect the capital.\n"
                    f"Scan #{AE._scan_count} total scans today.</i>"
                )

                # Send to all active subscribers
                active = SUB_MANAGER.all_active()
                sent   = 0
                for chat_id, _ in active:
                    ok = await send_msg(
                        app.bot, chat_id, msg, parse_mode="HTML")
                    if ok:
                        sent += 1
                    await asyncio.sleep(0.4)

                log.info(f"Daily summary sent to {sent} users")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Daily summary error: {e}")
                await asyncio.sleep(3600)

    # ── Regime cache: pre-compute for major pairs ─────────────────
    async def _run_regime_cache(self):
        log.info("Regime cache task active")
        await asyncio.sleep(25)
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    for pair in ASSET_CATEGORIES["forex_major"]:
                        try:
                            candles = await fetch_candles(
                                session, pair, "H1", 168)
                            daily   = await fetch_candles(
                                session, pair, "D",  60)
                            if len(candles) >= 50:
                                regime = REGIME_ENGINE.detect(
                                    candles, daily, pair)
                                log.debug(
                                    f"Regime {pair}: "
                                    f"{regime.regime.value} | "
                                    f"H:{regime.hurst_exponent:.2f} | "
                                    f"Contradict:{regime.contradictory}")
                            await asyncio.sleep(2)
                        except Exception as e:
                            log.debug(f"Regime cache {pair}: {e}")

                # Refresh every 30 minutes
                await asyncio.sleep(1800)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Regime cache error: {e}")
                await asyncio.sleep(600)

    # ── Health check: logs system status every 15 min ────────────
    async def _run_health_check(self):
        log.info("Health check task active")
        while True:
            try:
                await asyncio.sleep(900)   # every 15 minutes

                now      = datetime.now(timezone.utc)
                session  = SESSION_ENGINE.get_session(now)
                ml_s     = ML_ENGINE.get_stats()
                active   = len(PT.active)
                subs     = len(SUB_MANAGER.all_active())
                history  = len(QB.history)
                resolved = sum(1 for p in QB.history
                               if p.get("outcome") in ("WIN","LOSS"))
                pending  = sum(1 for p in QB.history
                               if p.get("outcome") == "PENDING")

                tasks_ok = sum(
                    1 for t in self.tasks.values()
                    if not t.done())

                log.info(
                    f"HEALTH | "
                    f"{session.session_name} | "
                    f"KZ:{session.is_kill_zone} | "
                    f"WAT:{session.wat_time} | "
                    f"Tasks:{tasks_ok}/{len(self.tasks)} | "
                    f"ActivePreds:{active} | "
                    f"Subs:{subs} | "
                    f"History:{history} | "
                    f"Resolved:{resolved} | "
                    f"Pending:{pending} | "
                    f"ML:{'✅' if ml_s['is_trained'] else '⏳'} "
                    f"({ml_s['n_samples']}/{ML_MIN_SAMPLES}) | "
                    f"Scans:{AE._scan_count}"
                )

                # Warn if tasks are dying
                if tasks_ok < len(self.tasks):
                    dead = [
                        name for name, t in self.tasks.items()
                        if t.done()]
                    log.warning(
                        f"Dead tasks detected: {dead}. "
                        f"They will auto-restart.")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Health check error: {e}")
                await asyncio.sleep(300)

    def stop_all(self):
        """Cancel all running tasks gracefully."""
        self.running = False
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                log.info(f"Task cancelled: {name}")

    def get_status(self) -> Dict:
        return {
            name: {
                "running":    not task.done(),
                "errors":     self.error_counts.get(name, 0),
                "started_at": (self.start_times[name].isoformat()
                               if name in self.start_times else None),
            }
            for name, task in self.tasks.items()
        }


BG_MANAGER = BackgroundTaskManager()

# ════════════════════════════════════════════════════════════════
#  ADDITIONAL COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """System health status command."""
    session  = SESSION_ENGINE.get_session()
    ml_s     = ML_ENGINE.get_stats()
    exp      = PERF_ENGINE.get_expectancy()
    task_st  = BG_MANAGER.get_status()
    wat      = session.wat_time
    utc_t    = SESSION_ENGINE.utc_time()

    tasks_ok  = sum(1 for t in task_st.values() if t["running"])
    tasks_tot = len(task_st)

    resolved  = sum(1 for p in QB.history
                    if p.get("outcome") in ("WIN","LOSS"))
    pending   = sum(1 for p in QB.history
                    if p.get("outcome") == "PENDING")

    msg = (
        f"{'='*35}\n"
        f"⚙️ <b>SYSTEM STATUS</b>\n"
        f"{'='*35}\n\n"
        f"🕐 {wat} ({utc_t})\n"
        f"📍 {session.session_name}"
        f"{' | 🎯 ' + session.kill_zone_name if session.is_kill_zone else ''}\n"
        f"AMD Phase: {session.amd_phase}\n"
        f"DST: "
        f"{'London ' if session.is_dst_london else ''}"
        f"{'NY' if session.is_dst_ny else ''}"
        f"{'None' if not session.is_dst_london and not session.is_dst_ny else ''}\n\n"

        f"{'─'*35}\n"
        f"<b>Background Tasks ({tasks_ok}/{tasks_tot} running):</b>\n"
    )

    for name, st in task_st.items():
        icon = "✅" if st["running"] else "❌"
        err  = f" ({st['errors']}err)" if st["errors"] > 0 else ""
        msg += f"{icon} {name}{err}\n"

    msg += (
        f"\n{'─'*35}\n"
        f"<b>Data:</b>\n"
        f"├ History:    {len(QB.history)} signals\n"
        f"├ Resolved:   {resolved}\n"
        f"├ Pending:    {pending}\n"
        f"├ Active:     {len(PT.active)} predictions\n"
        f"├ Subscribers:{len(SUB_MANAGER.all_active())}\n"
        f"├ Patterns:   {len(PATTERN_MEMORY.patterns)}\n"
        f"├ BT results: {len(HISTORICAL_BT.results)}\n"
        f"└ ML store:   {ML_DATA_STORE.n_total} total / "
        f"{ML_DATA_STORE.n_samples} resolved\n\n"

        f"{'─'*35}\n"
        f"<b>ML Engine:</b>\n"
        f"├ Status:    "
        f"{'✅ Trained' if ml_s['is_trained'] else '⏳ Accumulating'}\n"
        f"├ Resolved:  {ml_s['n_samples']}/{ML_MIN_SAMPLES}\n"
        f"├ Total:     {ml_s['n_total']} tracked\n"
        f"├ Need:      {ml_s['n_needed']} more resolved\n"
        f"├ OOS Acc:   {ml_s['oos_accuracy']:.1%}\n"
        f"├ Brier:     {ml_s['brier_score']:.3f}\n"
        f"└ Calibrated:{'✅' if ml_s['is_calibrated'] else '❌'}\n\n"

        f"{'─'*35}\n"
        f"<b>Performance:</b>\n"
        f"├ Trades:    {exp['total']}\n"
        f"├ Win Rate:  {exp['win_rate']:.1%}\n"
        f"├ Expectancy:{exp['expectancy']:+.1f} pips/trade\n"
        f"└ Max DD:    "
        f"{PERF_ENGINE.data.get('max_drawdown',0):.1f} pips\n\n"

        f"<b>Scanner:</b>\n"
        f"└ Total scans: {AE._scan_count}\n"
    )

    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show backtest results."""
    msg_l  = await update.message.reply_text(
        "⏳ Loading backtest results...", parse_mode="HTML")
    report = HISTORICAL_BT.format_report(QB.history)
    ch     = chart_performance(QB.history)
    await del_msg(msg_l)
    if ch:
        await send_photo(
            ctx.bot, update.effective_chat.id, ch,
            caption="📊 <b>Backtest + Performance</b>",
            parse_mode="HTML")
    await update.message.reply_text(
        report, parse_mode="HTML", reply_markup=kb_main())


async def cmd_scan_now(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Force an immediate scan of all pairs."""
    session = SESSION_ENGINE.get_session()
    msg_l   = await update.message.reply_text(
        f"🔍 Scanning all {len(ALL_PAIRS)} pairs...\n"
        f"Session: {session.session_name} | {session.wat_time}",
        parse_mode="HTML")
    try:
        n = await AE.scan_all(ctx.application)
        await msg_l.edit_text(
            f"✅ Scan complete. {n} signal(s) sent.\n"
            f"Total scans run: {AE._scan_count}",
            parse_mode="HTML")
    except Exception as e:
        await msg_l.edit_text(f"❌ Scan error: {str(e)[:100]}")


async def cmd_pairs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List all tracked instruments."""
    session = SESSION_ENGINE.get_session()
    best    = SA.BEST_PAIRS.get(session.session_name, [])

    msg = (
        f"💱 <b>ALL TRACKED INSTRUMENTS</b>\n\n"
        f"🕐 {session.wat_time} | {session.session_name}\n\n"
                f"<b>📊 Majors ({len(ASSET_CATEGORIES['forex_major'])}):</b>\n"
        f"{', '.join(pair_display(p) for p in ASSET_CATEGORIES['forex_major'])}\n\n"
        f"<b>💱 Crosses ({len(ASSET_CATEGORIES['forex_cross'])}):</b>\n"
        f"{', '.join(pair_display(p) for p in ASSET_CATEGORIES['forex_cross'])}\n\n"
        f"<b>🥇 Metals ({len(ASSET_CATEGORIES['metals'])}):</b>\n"
        f"{', '.join(pair_display(p) for p in ASSET_CATEGORIES['metals'])}\n\n"
        f"<b>📈 Indices ({len(ASSET_CATEGORIES['indices'])}):</b>\n"
        f"{', '.join(pair_display(p) for p in ASSET_CATEGORIES['indices'])}\n\n"
        f"<b>Total: {len(ALL_PAIRS)} instruments</b>\n\n"
    )

    if best:
        msg += (
            f"<b>✅ Best for {session.session_name} now:</b>\n"
            f"{', '.join(pair_display(p) for p in best)}\n\n")

    msg += (
        f"<i>All {len(ALL_PAIRS)} pairs scanned automatically.\n"
        f"Use /alerts to subscribe to signals.</i>"
    )

    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_math(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Explain the mathematical models used."""
    msg = (
        f"📐 <b>MARKET MATHEMATICS GUIDE</b>\n\n"

        f"<b>Hurst Exponent (H):</b>\n"
        f"• H > 0.58: Trending (momentum strategies work)\n"
        f"• H = 0.50: Random walk (no statistical edge)\n"
        f"• H < 0.42: Mean-reverting (fade extremes)\n"
        f"• Calculated via Rescaled Range (R/S) analysis\n\n"

        f"<b>Fractal Dimension:</b>\n"
        f"• Near 1.0: Clean smooth trend\n"
        f"• Near 2.0: Chaotic space-filling\n"
        f"• Independent confirmation of Hurst\n\n"

        f"<b>Shannon Entropy:</b>\n"
        f"• Low (<0.5): Orderly, predictable\n"
        f"• High (>0.85): Chaotic, avoid trading\n"
        f"• Measures information disorder in returns\n\n"

        f"<b>Z-Score Normalization:</b>\n"
        f"• Signals scored vs own 200-period history\n"
        f"• Normal imbalance: z = 0.5 (ignored)\n"
        f"• Unusual imbalance: z > 2.0 ✅ (actionable)\n"
        f"• Prevents false confluence from normal noise\n\n"

        f"<b>Variance Ratio Test:</b>\n"
        f"• > 1.0: Momentum (prices persist)\n"
        f"• = 1.0: Random walk\n"
        f"• < 1.0: Mean reversion\n\n"

        f"<b>Market Efficiency Ratio:</b>\n"
        f"• Direct move / total path = 0 to 1\n"
        f"• Near 1.0: Pure trend (efficient move)\n"
        f"• Near 0.0: Choppy (inefficient)\n\n"

        f"<b>ADR (Avg Daily Range):</b>\n"
        f"• Signals rejected if >80% of ADR consumed\n"
        f"• Prevents chasing exhausted moves\n"
        f"• Calculated over last 20 trading days\n\n"

        f"<b>R/R Calculation (FIXED in v8.1):</b>\n"
        f"• R/R = Reward ÷ Risk\n"
        f"• 1:2 means risk 1 pip to make 2 pips\n"
        f"• Minimum: 1:{MIN_RR_RATIO}\n"
        f"• Validated BEFORE any signal is sent\n\n"

        f"<b>Kelly Criterion:</b>\n"
        f"• Optimal bet = f = (bp-q)/b\n"
        f"• We use Half-Kelly for safety\n"
        f"• Capped at 5% of account\n\n"

        f"<b>Fibonacci OTE Zone:</b>\n"
        f"• 61.8%–78.6% retracement\n"
        f"• Optimal Trade Entry (ICT)\n"
        f"• Highest probability when OB/FVG aligns\n\n"

        f"<b>Regime Contradiction Detection:</b>\n"
        f"• Hurst says trending BUT ER says choppy\n"
        f"• Signal confidence penalized by 15%\n"
        f"• Displayed with ⚠️ in signals\n"
    )
    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_ict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Explain ICT concepts implemented."""
    msg = (
        f"🏛️ <b>ICT CONCEPTS IN v8.1</b>\n\n"

        f"<b>Market Structure:</b>\n"
        f"• BOS: Break of Structure (trend continuation)\n"
        f"• CHOCH: Change of Character (reversal signal)\n"
        f"• HH+HL = Bullish | LH+LL = Bearish\n"
        f"• Internal vs External structure tracked\n\n"

        f"<b>Liquidity Levels:</b>\n"
        f"• Equal Highs/Lows: stop-loss clusters\n"
        f"• PDH/PDL: Previous Day High/Low\n"
        f"• ASH/ASL: Asian Session High/Low\n"
        f"• Sweep = wick through + close back inside\n\n"

        f"<b>Fair Value Gap (FVG):</b>\n"
        f"• 3-candle pattern leaving a price gap\n"
        f"• Unfilled = magnet for future price\n"
        f"• Consequent encroachment (50% fill)\n"
        f"• Inversion: once filled becomes S/R\n\n"

        f"<b>Order Block (STRICT validation):</b>\n"
        f"✅ Last opposing candle before displacement\n"
        f"✅ Displacement must break structure\n"
        f"✅ Must leave a Fair Value Gap\n"
        f"❌ Missing any = OB rejected entirely\n\n"

        f"<b>Displacement Candle:</b>\n"
        f"• Body ≥ 65% of total range\n"
        f"• Body ≥ 1.5× recent average body\n"
        f"• Institutional order execution signature\n\n"

        f"<b>AMD (Power of Three):</b>\n"
        f"• Accumulation: Asian session range\n"
        f"• Manipulation: Fake move at session open\n"
        f"• Distribution: True directional move\n"
        f"• Only trades DISTRIBUTION phase\n\n"

        f"<b>OTE Zone (Optimal Trade Entry):</b>\n"
        f"• 61.8%–78.6% Fibonacci retracement\n"
        f"• Where institutions re-enter after impulse\n"
        f"• Strongest when OB or FVG aligns here\n\n"

        f"<b>Kill Zones (WAT times):</b>\n"
        f"🎯 London KZ: 08:00–11:00 WAT\n"
        f"🎯 NY KZ:     13:00–16:00 WAT\n"
        f"• AMD manipulation phase happens here\n"
        f"• Tighter scan interval (90s vs 240s)\n"
        f"• Lower confidence threshold for alerts\n\n"

        f"<b>HTF Bias Enforcement (NEW v8.1):</b>\n"
        f"• Never LONG against H4 bearish structure\n"
        f"• Never SHORT against H4 bullish structure\n"
        f"• NEUTRAL HTF allowed but penalized\n"
    )
    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_trade(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Look up a specific trade by Trade ID."""
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "Usage: /trade FQ-EURUSD-BUY-20240115-A3K9\n\n"
            "Find a specific trade by its ID.",
            reply_markup=kb_main())
        return

    trade_id = args[0].upper()

    # Search active predictions
    active_match = next(
        (p for p in PT.active if p.trade_id == trade_id), None)

    # Search history
    history_match = next(
        (h for h in QB.history
         if h.get("trade_id") == trade_id), None)

    if not active_match and not history_match:
        await update.message.reply_text(
            f"❌ Trade ID <code>{trade_id}</code> not found.\n\n"
            f"<i>Trade IDs are shown in signal messages.\n"
            f"Format: FQ-EURUSD-BUY-20240115-A3K9</i>",
            parse_mode="HTML", reply_markup=kb_main())
        return

    if active_match:
        pred = active_match
        pip  = pip_value(pred.pair)
        # Try to get current price
        cp   = None
        try:
            async with aiohttp.ClientSession() as session:
                cp = await fetch_current_price(session, pred.pair)
        except Exception:
            pass

        pnl = 0.0
        if cp:
            if pred.direction == "LONG":
                pnl = (cp - pred.current_price) / pip
            else:
                pnl = (pred.current_price - cp) / pip

        msg = (
            f"{'='*35}\n"
            f"📋 <b>TRADE LOOKUP</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{trade_id}</code>\n"
            f"Status: <b>🟡 ACTIVE</b>\n\n"
            f"{asset_emoji(pred.pair)} <b>{pair_display(pred.pair)}</b> | "
            f"{'🟢' if pred.direction=='LONG' else '🔴'} "
            f"{pred.direction}\n"
            f"📐 Strategy: {pred.strategy}\n"
            f"⭐ Quality:  {pred.quality}\n\n"
            f"<b>Levels:</b>\n"
            f"├ Entry:     <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ 🎯 Target: <code>{fmt_price(pred.target_price,  pred.pair)}</code>\n"
            f"├ ⛔ Stop:   <code>{fmt_price(pred.invalidation_price, pred.pair)}</code>\n"
            f"├ 📌 BE:     <code>{fmt_price(pred.breakeven_price, pred.pair)}</code>\n"
            f"└ 🔄 Trail:  <code>{fmt_price(pred.trailing_price,  pred.pair)}</code>\n\n"
            f"<b>Stats:</b>\n"
            f"├ R/R:      1:{pred.rr_ratio:.2f}\n"
            f"├ Reward:   +{pred.reward_pips:.1f} pips\n"
            f"├ Risk:     -{pred.risk_pips:.1f} pips\n"
            f"├ Conf:     {pred.confidence:.0f}%\n"
            f"├ MFE so far: +{pred.mfe_pips:.1f}p\n"
            f"├ MAE so far: -{pred.mae_pips:.1f}p\n"
        )
        if cp:
            msg += (
                f"├ Current:  <code>{fmt_price(cp, pred.pair)}</code>\n"
                f"└ Float P/L: <b>{pnl:+.1f} pips</b>\n\n"
            )
        msg += (
            f"<b>Context:</b>\n"
            f"├ Session: {pred.session_at_signal}\n"
            f"├ Regime:  {pred.regime_at_signal}\n"
            f"├ ML:      {pred.ml_confidence:.0%}\n"
            f"└ Opened:  {pred.timestamp[:19]}\n\n"
            f"<b>BE notified:</b> "
            f"{'✅' if pred.breakeven_notified else '⏳'} | "
            f"<b>Trail notified:</b> "
            f"{'✅' if pred.trailing_notified else '⏳'}\n"
        )

    else:
        h   = history_match
        msg = (
            f"{'='*35}\n"
            f"📋 <b>TRADE LOOKUP</b>\n"
            f"{'='*35}\n\n"
            f"🆔 <code>{trade_id}</code>\n"
        )
        outcome = h.get("outcome", "PENDING")
        if outcome == "WIN":
            status_icon = "✅ WIN"
        elif outcome == "LOSS":
            status_icon = "❌ LOSS"
        elif outcome == "EXPIRED":
            status_icon = "⏰ EXPIRED"
        else:
            status_icon = "🟡 PENDING"

        msg += (
            f"Status: <b>{status_icon}</b>\n\n"
            f"{asset_emoji(h.get('pair',''))} "
            f"<b>{pair_display(h.get('pair','?'))}</b> | "
            f"{'🟢' if h.get('direction')=='LONG' else '🔴'} "
            f"{h.get('direction','?')}\n"
            f"📐 Strategy: {h.get('strategy','?')}\n"
            f"⭐ Quality:  {h.get('quality','?')}\n\n"
            f"<b>Levels:</b>\n"
            f"├ Entry:  <code>{fmt_price(h.get('current_price',0), h.get('pair','EUR_USD'))}</code>\n"
            f"├ Target: <code>{fmt_price(h.get('target_price',0),  h.get('pair','EUR_USD'))}</code>\n"
            f"└ Stop:   <code>{fmt_price(h.get('invalidation_price',0), h.get('pair','EUR_USD'))}</code>\n\n"
            f"<b>Result:</b>\n"
            f"├ P/L:   <b>{h.get('pips_gained',0):+.1f} pips</b>\n"
            f"├ MAE:   {h.get('mae_pips',0):.1f}p\n"
            f"├ MFE:   {h.get('mfe_pips',0):.1f}p\n"
            f"├ R/R:   1:{h.get('rr_ratio',0):.2f}\n"
            f"├ Conf:  {h.get('confidence',0):.0f}%\n\n"
            f"<b>Context:</b>\n"
            f"├ Session: {h.get('session','?')}\n"
            f"├ Regime:  {h.get('regime','?')}\n"
            f"└ Opened:  {str(h.get('timestamp','?'))[:19]}\n"
        )

    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_active(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show all currently active predictions."""
    if not PT.active:
        await update.message.reply_text(
            "📋 <b>No active predictions right now.</b>\n\n"
            "<i>Signals are tracked automatically when generated.</i>",
            parse_mode="HTML", reply_markup=kb_main())
        return

    msg = (
        f"📋 <b>ACTIVE PREDICTIONS ({len(PT.active)})</b>\n\n"
        f"{'─'*35}\n"
    )

    for pred in PT.active[:10]:   # show max 10
        pip  = pip_value(pred.pair)
        de   = "🟢" if pred.direction == "LONG" else "🔴"
        age  = ""
        try:
            pt  = datetime.fromisoformat(
                pred.timestamp.replace("Z","+00:00"))
            hrs = (datetime.now(timezone.utc) - pt
                   ).total_seconds() / 3600
            age = f"{hrs:.1f}h"
        except Exception:
            pass

        msg += (
            f"🆔 <code>{pred.trade_id}</code>\n"
            f"{asset_emoji(pred.pair)} "
            f"<b>{pair_display(pred.pair)}</b> "
            f"{de} {pred.direction} | {pred.quality}\n"
            f"├ Entry:   <code>{fmt_price(pred.current_price, pred.pair)}</code>\n"
            f"├ Target:  <code>{fmt_price(pred.target_price,  pred.pair)}</code>\n"
            f"├ Stop:    <code>{fmt_price(pred.invalidation_price, pred.pair)}</code>\n"
            f"├ R/R:     1:{pred.rr_ratio:.2f} | "
            f"Conf: {pred.confidence:.0f}%\n"
            f"├ BE: {'✅' if pred.breakeven_notified else '⏳'} | "
            f"Trail: {'✅' if pred.trailing_notified else '⏳'}\n"
            f"└ Age: {age} | MFE:{pred.mfe_pips:.1f}p MAE:{pred.mae_pips:.1f}p\n\n"
        )

    if len(PT.active) > 10:
        msg += f"<i>...and {len(PT.active)-10} more</i>\n"

    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_ml(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show detailed ML engine status."""
    ml_s = ML_ENGINE.get_stats()
    msg  = (
        f"{'='*35}\n"
        f"🧠 <b>ML ENGINE STATUS</b>\n"
        f"{'='*35}\n\n"

        f"<b>Status:</b> "
        f"{'✅ ACTIVE & LEARNING' if ml_s['is_trained'] else '⏳ ACCUMULATING DATA'}\n\n"

        f"<b>Data:</b>\n"
        f"├ Total tracked:  {ml_s['n_total']}\n"
        f"├ Resolved:       {ml_s['n_samples']}\n"
        f"├ Need to train:  {ml_s['n_needed']} more\n"
        f"└ Min threshold:  {ML_MIN_SAMPLES}\n\n"
    )

    if ml_s["is_trained"]:
        msg += (
            f"<b>Performance:</b>\n"
            f"├ OOS Accuracy:  <b>{ml_s['oos_accuracy']:.1%}</b>\n"
            f"├ Brier Score:   {ml_s['brier_score']:.4f} "
            f"(lower=better)\n"
            f"├ Calibrated:    "
            f"{'✅ Yes' if ml_s['is_calibrated'] else '❌ No'}\n"
            f"└ Ensemble:      "
            f"RF {ml_s['weights'][0]:.0%} + "
            f"GB {ml_s['weights'][1]:.0%} + "
            f"LR {ml_s['weights'][2]:.0%}\n\n"

            f"<b>Top Predictive Features:</b>\n"
        )
        for feat, imp in ml_s.get("top_features", [])[:8]:
            bar = progress_bar(imp * 1000, 8)
            msg += f"• {feat[:20]}: {bar} {imp:.4f}\n"

        if ml_s["accuracy_history"]:
            msg += (
                f"\n<b>Accuracy History:</b>\n"
                f"{' → '.join(f'{v:.0%}' for v in ml_s['accuracy_history'][-5:])}\n"
            )
    else:
        msg += (
            f"<b>What happens at {ML_MIN_SAMPLES} samples:</b>\n"
            f"• Purged walk-forward training begins\n"
            f"• Isotonic probability calibration\n"
            f"• Meta-labeling: predicts signal SUCCESS\n"
            f"• NOT direction — the rule system does that\n"
            f"• Signals with ML prob < 38% are penalized\n\n"
            f"<b>Data accumulates from:</b>\n"
            f"• Every auto-alert signal generated\n"
            f"• Every manual /signal request\n"
            f"• Historical backtest simulation\n"
            f"• Data persists across bot restarts ✅\n"
        )

    await update.message.reply_text(
        msg, parse_mode="HTML", reply_markup=kb_main())


async def cmd_validate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Test signal geometry validator — useful for debugging."""
    args = ctx.args
    if len(args) < 4:
        await update.message.reply_text(
            "Usage: /validate DIRECTION ENTRY TARGET STOP PAIR\n\n"
            "Example: /validate LONG 1.0820 1.0870 1.0800 EUR_USD\n\n"
            "Tests if a signal would pass geometry validation.",
            reply_markup=kb_main())
        return

    try:
        direction = args[0].upper()
        entry     = float(args[1])
        target    = float(args[2])
        stop      = float(args[3])
        pair      = args[4].upper() if len(args) > 4 else "EUR_USD"

        if pair not in ALL_PAIRS:
            pair = "EUR_USD"

        val = validate_signal_geometry(
            direction, entry, target, stop, pair)

        if val.is_valid:
            msg = (
                f"✅ <b>SIGNAL VALID</b>\n\n"
                f"Pair:       {pair_display(pair)}\n"
                f"Direction:  {direction}\n"
                f"Entry:      <code>{fmt_price(entry,  pair)}</code>\n"
                f"Target:     <code>{fmt_price(target, pair)}</code>\n"
                f"Stop:       <code>{fmt_price(stop,   pair)}</code>\n\n"
                f"Reward:     +{val.reward_pips:.1f} pips\n"
                f"Risk:       -{val.risk_pips:.1f} pips\n"
                f"R/R:        1:{val.rr_ratio:.2f} ✅\n\n"
                f"BE Level:   <code>{fmt_price(calculate_breakeven_price(direction, entry, target), pair)}</code>\n"
                f"Trail Level:<code>{fmt_price(calculate_trailing_price(direction, entry, target), pair)}</code>"
            )
        else:
            msg = (
                f"❌ <b>SIGNAL INVALID</b>\n\n"
                f"Pair:      {pair_display(pair)}\n"
                f"Direction: {direction}\n"
                f"Entry:     <code>{fmt_price(entry,  pair)}</code>\n"
                f"Target:    <code>{fmt_price(target, pair)}</code>\n"
                f"Stop:      <code>{fmt_price(stop,   pair)}</code>\n\n"
                f"<b>Reason: {val.reason}</b>\n\n"
                f"Reward pips: {val.reward_pips:.1f}\n"
                f"Risk pips:   {val.risk_pips:.1f}\n"
                f"R/R:         {val.rr_ratio:.2f}"
            )

        await update.message.reply_text(
            msg, parse_mode="HTML", reply_markup=kb_main())

    except (ValueError, IndexError) as e:
        await update.message.reply_text(
            f"❌ Parse error: {e}\n\n"
            f"Usage: /validate LONG 1.0820 1.0870 1.0800 EUR_USD",
            reply_markup=kb_main())


# ════════════════════════════════════════════════════════════════
#  STARTUP AND SHUTDOWN HOOKS
# ════════════════════════════════════════════════════════════════

async def on_startup(app):
    """Called after bot is initialized. Starts all background tasks."""
    log.info("Bot startup sequence initiated...")

    session = SESSION_ENGINE.get_session()
    log.info(
        f"Session: {session.session_name} | "
        f"WAT: {session.wat_time} | "
        f"KZ: {session.is_kill_zone} | "
        f"AMD: {session.amd_phase}")

    # Attempt to load any QB history into ML data store
    # This ensures ML accumulates even if predictions were made
    # before ML_DATA_STORE existed
    existing_resolved = [
        p for p in QB.history
        if p.get("outcome") in ("WIN","LOSS")
        and len(p.get("features",[])) == 20
    ]
    if existing_resolved:
        log.info(
            f"Found {len(existing_resolved)} resolved predictions "
            f"in QB history — syncing to ML data store...")
        for h in existing_resolved:
            # Only add if not already in store
            existing_ids = {
                s.get("prediction_id","")
                for s in ML_DATA_STORE.data}
            if h.get("prediction_id","") not in existing_ids:
                ML_DATA_STORE.add({
                    "prediction_id": h.get("prediction_id",""),
                    "features":      h.get("features",[]),
                    "outcome":       h.get("outcome",""),
                    "pips_gained":   h.get("pips_gained",0),
                    "strategy":      h.get("strategy",""),
                    "session":       h.get("session",""),
                    "regime":        h.get("regime",""),
                })
        log.info(
            f"ML data store now has "
            f"{ML_DATA_STORE.n_samples} resolved samples")

        # If we have enough, train immediately
        if (ML_DATA_STORE.n_samples >= ML_MIN_SAMPLES
                and not ML_ENGINE.is_trained):
            log.info("Training ML immediately from existing data...")
            result = ML_ENGINE.train()
            log.info(f"Startup ML train: {result}")

    # Start all background tasks
    await BG_MANAGER.start_all(app)

    # Notify existing subscribers of restart
    active = SUB_MANAGER.all_active()
    if active:
        session = SESSION_ENGINE.get_session()
        restart_msg = (
            f"🔄 <b>FOREX QUANT v8.1 ONLINE</b>\n\n"
            f"🕐 {session.wat_time}\n"
            f"📍 {session.session_name}"
            f"{' | 🎯 ' + session.kill_zone_name if session.is_kill_zone else ''}\n"
            f"AMD: {session.amd_phase}\n\n"
            f"<b>All systems active:</b>\n"
            f"✅ Scanning {len(ALL_PAIRS)} instruments\n"
            f"✅ Trade IDs for reference\n"
            f"✅ Breakeven + trailing alerts\n"
            f"✅ R/R validated (reward÷risk)\n"
            f"✅ TP/SL only for YOUR signals\n"
            f"✅ ML persists across restarts\n\n"
            f"<b>ML:</b> "
            f"{'✅ Active' if ML_ENGINE.is_trained else '⏳ Accumulating'} "
            f"({ML_DATA_STORE.n_samples}/{ML_MIN_SAMPLES})\n\n"
            f"<i>Signals fire automatically. "
            f"Use /alerts to manage your subscription.</i>"
        )
        for chat_id, _ in active:
            try:
                await send_msg(app.bot, chat_id, restart_msg,
                               parse_mode="HTML")
                await asyncio.sleep(0.4)
            except Exception:
                pass

    log.info("Startup complete. ForexQuant v8.1 is LIVE.")


async def on_shutdown(app):
    """Graceful shutdown — save all state."""
    log.info("Shutdown initiated...")
    BG_MANAGER.stop_all()

    # Save all state files
    try:
        QB._save()
        log.info("QB history saved")
    except Exception as e:
        log.error(f"QB save error: {e}")

    try:
        PT._save()
        log.info("Active predictions saved")
    except Exception as e:
        log.error(f"PT save error: {e}")

    try:
        PATTERN_MEMORY._save()
        log.info("Pattern memory saved")
    except Exception as e:
        log.error(f"Pattern memory save error: {e}")

    try:
        PERF_ENGINE._save()
        log.info("Performance data saved")
    except Exception as e:
        log.error(f"Performance save error: {e}")

    try:
        HISTORICAL_BT._save()
        log.info("Backtest results saved")
    except Exception as e:
        log.error(f"Backtest save error: {e}")

    try:
        ML_ENGINE._save()
        log.info("ML model saved")
    except Exception as e:
        log.error(f"ML save error: {e}")

    try:
        ML_DATA_STORE._save()
        log.info("ML data store saved")
    except Exception as e:
        log.error(f"ML data store save error: {e}")

    log.info("All state saved. Shutdown complete.")

# ════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():

    # ── Platform config ──────────────────────────────────────────
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())

    # ── Health server for Render/Railway keep-alive ───────────────
    Thread(target=start_health_server, daemon=True).start()

    # ── Startup banner ────────────────────────────────────────────
    session  = SESSION_ENGINE.get_session()
    ml_s     = ML_ENGINE.get_stats()

    log.info("=" * 65)
    log.info("  FOREX QUANT v8.1 — THE BEAST — ELITE TRADING INTELLIGENCE")
    log.info("=" * 65)
    log.info(f"  WAT Time:    {session.wat_time}")
    log.info(f"  Session:     {session.session_name}")
    log.info(f"  Kill Zone:   {session.is_kill_zone} "
             f"({session.kill_zone_name if session.is_kill_zone else 'None'})")
    log.info(f"  AMD Phase:   {session.amd_phase}")
    log.info(f"  DST London:  {session.is_dst_london} | "
             f"DST NY: {session.is_dst_ny}")
    log.info("─" * 65)
    log.info("  ICT:  BOS · CHOCH · OB · FVG · Sweep · AMD · OTE")
    log.info("  MATH: Hurst · Entropy · Z-Score · ADR · Kelly · Fib")
    log.info("  ML:   Meta-label · Purged WF · Isotonic Cal · Persistent")
    log.info("  SIG:  Geometry validated · HTF enforced · R/R corrected")
    log.info("  TRACK: Trade IDs · Breakeven · Trailing · MAE/MFE")
    log.info("─" * 65)
    log.info(f"  Instruments:  {len(ALL_PAIRS)} pairs")
    log.info(f"  Strategies:   {len(list(StrategyType))}")
    log.info(f"  ML Status:    "
             f"{'Active' if ml_s['is_trained'] else 'Accumulating'} "
             f"({ml_s['n_samples']}/{ML_MIN_SAMPLES} resolved, "
             f"{ml_s['n_total']} total)")
    log.info(f"  Active Preds: {len(PT.active)}")
    log.info(f"  History:      {len(QB.history)} signals")
    log.info(f"  Subscribers:  {len(SUB_MANAGER.all_active())}")
    log.info(f"  BT Results:   {len(HISTORICAL_BT.results)} pairs")
    log.info(f"  ML Store:     {ML_DATA_STORE.n_total} total / "
             f"{ML_DATA_STORE.n_samples} resolved")
    log.info(f"  Patterns:     {len(PATTERN_MEMORY.patterns)}")
    log.info("─" * 65)
    log.info(f"  Min R/R:      1:{MIN_RR_RATIO}")
    log.info(f"  Min Target:   Forex {MIN_TARGET_PIPS_FOREX}p | "
             f"Gold {MIN_TARGET_PIPS_GOLD}p")
    log.info(f"  Breakeven:    At {BREAKEVEN_TRIGGER_PCT:.0f}% of target")
    log.info(f"  Trailing:     At {TRAILING_TRIGGER_PCT:.0f}% of target")
    log.info(f"  Scan Normal:  {SCAN_INTERVAL_NORMAL}s | "
             f"KZ: {SCAN_INTERVAL_KILL_ZONE}s | "
             f"Off: {SCAN_INTERVAL_OFF_HOURS}s")
    log.info("=" * 65)

    # ── Build application ─────────────────────────────────────────
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(45)
        .write_timeout(45)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # ── Register ALL command handlers ─────────────────────────────

    # Core trading commands
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("signal",   cmd_signal))
    app.add_handler(CommandHandler("predict",  cmd_signal))      # alias
    app.add_handler(CommandHandler("analyze",  cmd_analyze))
    app.add_handler(CommandHandler("summary",  cmd_summary))

    # Alert and subscription
    app.add_handler(CommandHandler("alerts",   cmd_alerts))

    # Market data
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(CommandHandler("strength", cmd_strength))
    app.add_handler(CommandHandler("overview", cmd_overview))
    app.add_handler(CommandHandler("pairs",    cmd_pairs))

    # Performance and tracking
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("backtest",    cmd_backtest))
    app.add_handler(CommandHandler("active",      cmd_active))
    app.add_handler(CommandHandler("trade",       cmd_trade))

    # ML status
    app.add_handler(CommandHandler("ml",       cmd_ml))

    # Educational
    app.add_handler(CommandHandler("guide",    cmd_guide))
    app.add_handler(CommandHandler("math",     cmd_math))
    app.add_handler(CommandHandler("ict",      cmd_ict))

    # Admin/debug
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("scan",     cmd_scan_now))
    app.add_handler(CommandHandler("validate", cmd_validate))

    # Callback handler (all inline buttons)
    app.add_handler(CallbackQueryHandler(on_callback))

    # Error handler
    app.add_error_handler(on_error)

    # ── Lifecycle hooks ───────────────────────────────────────────
    app.post_init     = on_startup
    app.post_shutdown = on_shutdown

    # ── Launch ───────────────────────────────────────────────────
    log.info("🚀 FOREX QUANT v8.1 launching — THE BEAST IS LIVE!")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
        poll_interval=1.0,
        timeout=30,
    )


# ════════════════════════════════════════════════════════════════
#  PRODUCTION ASSEMBLY INSTRUCTIONS
# ════════════════════════════════════════════════════════════════
"""
TO DEPLOY AS A SINGLE FILE:

1. Create main.py
2. Paste Part 1 entirely (no changes)
3. Paste Part 2 after Part 1 — REMOVE the line:
      from part1 import *
4. Paste Part 3 after Part 2 — REMOVE the lines:
      from part1 import *
      from part2 import *
5. Paste Part 4 after Part 3 — REMOVE the lines:
      from part1 import *
      from part2 import *
      from part3 import *
6. Keep ONLY the if __name__ == "__main__": at the very bottom
   (from Part 4). Remove any others.

COMMANDS AVAILABLE:
  /start       — Welcome + current session status
  /signal      — Get signal for a pair (/predict is alias)
  /analyze     — Deep multi-part analysis
  /summary     — Full market intelligence brief
  /alerts      — Manage auto-alert subscription
  /sessions    — Trading session guide (WAT times)
  /strength    — Currency strength chart
  /overview    — Market overview chart
  /pairs       — List all tracked instruments
  /performance — Full performance report + chart
  /backtest    — Historical backtest results
  /active      — Show all active predictions
  /trade ID    — Look up specific trade by ID
  /ml          — ML engine detailed status
  /guide       — System guide
  /math        — Market mathematics explained
  /ict         — ICT concepts explained
  /status      — System health (admin)
  /scan        — Force immediate scan (admin)
  /validate    — Test signal geometry validator

KEY FIXES IN v8.1:
  ✅ R/R = Reward÷Risk (was inverted showing 1:3 for bad trades)
  ✅ Signal geometry validation before any signal sent
  ✅ Minimum pip distances per instrument enforced
  ✅ HTF bias conflict rejection (no LONG against H4 bearish)
  ✅ Regime contradiction detection + confidence penalty
  ✅ Trade IDs (FQ-EURUSD-BUY-20240115-A3K9)
  ✅ Breakeven alert at 50% of target
  ✅ Trailing stop alert at 75% of target
  ✅ TP/SL/BE/Trail ONLY for signals sent to that user
  ✅ ML persists across restarts (never resets)
  ✅ ML activates at 50 resolved trades (was 300)
  ✅ Quality filter per subscriber respected
  ✅ All pairs scanned (was only subscribed pairs for some)
  ✅ /trade command for trade ID lookup
  ✅ /active shows all live predictions
  ✅ /ml shows detailed ML status
  ✅ /validate tests signal geometry

FILES CREATED AT RUNTIME:
  predictions_v81.json
  active_predictions_v81.json
  ml_model_v81.pkl
  ml_training_data_v81.json
  historical_backtest_v81.json
  pattern_memory_v81.json
  user_subscriptions_v81.json
  performance_v81.json
"""

REQUIREMENTS_TXT = """
python-telegram-bot==20.7
aiohttp==3.9.1
numpy==1.26.2
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
"""

RENDER_YAML = """
services:
  - type: web
    name: forex-quant-v81
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: OANDA_API_KEY
        sync: false
      - key: OANDA_ACCOUNT_ID
        sync: false
      - key: TELEGRAM_TOKEN
        sync: false
      - key: OANDA_ENV
        value: practice
      - key: PORT
        value: 10000
"""

# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
