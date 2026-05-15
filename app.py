import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
Path(".cache").mkdir(exist_ok=True)

st.set_page_config(
    page_title="Professional Investments Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# TRANSLATIONS
# ═══════════════════════════════════════════════════════════════

T = {
    "header_sub": {
        "EN": "Institutional-grade market intelligence for stocks, crypto and ETFs",
        "EL": "Επαγγελματική ανάλυση αγορών για μετοχές, crypto και ETFs",
        "DE": "Professionelle Marktanalyse für Aktien, Krypto und ETFs",
    },
    "hero_equities": {"EN": "Equities", "EL": "Μετοχές", "DE": "Aktien"},
    "hero_crypto": {"EN": "Crypto", "EL": "Crypto", "DE": "Krypto"},
    "hero_etfs": {"EN": "ETFs", "EL": "ETFs", "DE": "ETFs"},
    "hero_technical": {"EN": "Technical Signals", "EL": "Τεχνικά Σήματα", "DE": "Technische Signale"},
    "hero_financials": {"EN": "Financial Statements", "EL": "Οικονομικές Καταστάσεις", "DE": "Finanzberichte"},
    "hero_comparison": {"EN": "Comparative Analysis", "EL": "Συγκριτική Ανάλυση", "DE": "Vergleichsanalyse"},
    "search_placeholder": {
        "EN": "Search ticker or company name — e.g. AAPL, Bitcoin, NVDA…",
        "EL": "Αναζήτηση ticker ή ονόματος — π.χ. AAPL, Bitcoin, NVDA…",
        "DE": "Ticker oder Firmenname suchen — z. B. AAPL, Bitcoin, NVDA…",
    },
    "btn_analyze": {"EN": "▶  Analyze", "EL": "▶  Ανάλυση", "DE": "▶  Analysieren"},
    "quick_picks": {"EN": "Quick Picks", "EL": "Γρήγορες Επιλογές", "DE": "Schnellauswahl"},
    "period_label": {"EN": "Time Period", "EL": "Χρονική Περίοδος", "DE": "Zeitraum"},
    "periods": {
        "EN": ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
        "EL": ["1 Μήνας", "3 Μήνες", "6 Μήνες", "1 Έτος", "2 Έτη", "5 Έτη"],
        "DE": ["1 Monat", "3 Monate", "6 Monate", "1 Jahr", "2 Jahre", "5 Jahre"],
    },
    "fetching": {"EN": "Fetching data…", "EL": "Λήψη δεδομένων…", "DE": "Daten werden geladen…"},
    "analyzing": {"EN": "Analyzing {}…", "EL": "Ανάλυση {}…", "DE": "{} wird analysiert…"},
    "done": {"EN": "✓ Done {}", "EL": "✓ Ολοκλήρωση {}", "DE": "✓ {} fertig"},
    "no_symbol_error": {
        "EN": "Please enter at least one symbol.",
        "EL": "Παρακαλώ εισάγετε τουλάχιστον ένα σύμβολο.",
        "DE": "Bitte mindestens ein Symbol eingeben.",
    },
    "no_data_warning": {
        "EN": "No valid data found for {}.",
        "EL": "Δεν βρέθηκαν έγκυρα δεδομένα για {}.",
        "DE": "Keine gültigen Daten für {} gefunden.",
    },
    "no_data_error": {
        "EN": "No valid data retrieved.",
        "EL": "Δεν ανακτήθηκαν έγκυρα δεδομένα.",
        "DE": "Keine gültigen Daten abgerufen.",
    },
    "return_comparison": {"EN": "Return Comparison", "EL": "Σύγκριση Αποδόσεων", "DE": "Renditevergleich"},
    "price": {"EN": "Price", "EL": "Τιμή", "DE": "Kurs"},
    "return": {"EN": "Return", "EL": "Απόδοση", "DE": "Rendite"},
    "market_cap": {"EN": "Market Cap", "EL": "Κεφαλαιοποίηση", "DE": "Marktkapitalisierung"},
    "risk_beta": {"EN": "Risk (Beta)", "EL": "Κίνδυνος (Beta)", "DE": "Risiko (Beta)"},
    "analyst_target": {"EN": "Analyst Target", "EL": "Στόχος Αναλυτών", "DE": "Analystenziel"},
    "analyst_consensus": {"EN": "Analyst Consensus", "EL": "Consensus Αναλυτών", "DE": "Analystenkonsens"},
    "analysts": {"EN": "analysts", "EL": "αναλυτές", "DE": "Analysten"},
    "upside": {"EN": "Upside", "EL": "Upside", "DE": "Upside"},
    "technical_analysis": {"EN": "Technical Analysis", "EL": "Τεχνική Ανάλυση", "DE": "Technische Analyse"},
    "technical_signals": {"EN": "Technical Signals", "EL": "Τεχνικά Σήματα", "DE": "Technische Signale"},
    "fundamentals_title": {"EN": "Fundamental Indicators", "EL": "Θεμελιώδεις Δείκτες", "DE": "Fundamentale Kennzahlen"},
    "score_radar": {"EN": "Score Radar", "EL": "Radar Score", "DE": "Radar-Score"},
    "category_breakdown": {"EN": "Category Breakdown", "EL": "Ανάλυση Κατηγοριών", "DE": "Kategorien"},
    "overall_score": {"EN": "Overall Score", "EL": "Συνολικό Score", "DE": "Gesamtscore"},
    "business_desc": {"EN": "Business Description", "EL": "Περιγραφή Επιχείρησης", "DE": "Unternehmensbeschreibung"},
    "investment_assessment": {"EN": "Investment Assessment", "EL": "Επενδυτική Αξιολόγηση", "DE": "Investmentbewertung"},
    "download_csv": {"EN": "⬇  Download CSV", "EL": "⬇  Λήψη CSV", "DE": "⬇  CSV herunterladen"},
    "annual": {"EN": "Annual", "EL": "Ετήσια", "DE": "Jährlich"},
    "quarterly": {"EN": "Quarterly", "EL": "Τριμηνιαία", "DE": "Quartalsweise"},
    "income_statement": {"EN": "Income Statement", "EL": "Κατάσταση Αποτελεσμάτων", "DE": "Gewinn- und Verlustrechnung"},
    "balance_sheet": {"EN": "Balance Sheet", "EL": "Ισολογισμός", "DE": "Bilanz"},
    "cash_flow": {"EN": "Cash Flow", "EL": "Ταμειακές Ροές", "DE": "Cashflow"},
    "trend_charts": {"EN": "Trend Charts", "EL": "Διαγράμματα Τάσης", "DE": "Trend-Charts"},
    "no_financials": {
        "EN": "No historical financial statements available for this asset.",
        "EL": "Δεν υπάρχουν διαθέσιμα ιστορικά οικονομικά στοιχεία για αυτό το asset.",
        "DE": "Keine historischen Finanzdaten für dieses Asset verfügbar.",
    },
    "footer": {
        "EN": "Data provided by Yahoo Finance · For informational purposes only · Not financial advice",
        "EL": "Δεδομένα από Yahoo Finance · Μόνο για ενημέρωση · Δεν αποτελεί επενδυτική συμβουλή",
        "DE": "Daten von Yahoo Finance · Nur zu Informationszwecken · Keine Anlageberatung",
    },
    "valuation": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    "quality": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    "growth": {"EN": "Growth", "EL": "Ανάπτυξη", "DE": "Wachstum"},
    "risk": {"EN": "Risk", "EL": "Κίνδυνος", "DE": "Risiko"},
    "technical": {"EN": "Technical", "EL": "Τεχνικά", "DE": "Technisch"},
    "indicator_col": {"EN": "Indicator", "EL": "Δείκτης", "DE": "Indikator"},
    "value_col": {"EN": "Current Value", "EL": "Τρέχουσα Τιμή", "DE": "Aktueller Wert"},
    "range_col": {"EN": "Healthy Range", "EL": "Υγιές Εύρος", "DE": "Gesunder Bereich"},
    "signal_col": {"EN": "Signal", "EL": "Σήμα", "DE": "Signal"},
    "category_col": {"EN": "Category", "EL": "Κατηγορία", "DE": "Kategorie"},
    "what_col": {"EN": "What it measures", "EL": "Τι μετρά", "DE": "Was es misst"},
    "how_col": {"EN": "How to read it", "EL": "Πώς διαβάζεται", "DE": "Wie man es liest"},
    "oversold": {"EN": "Oversold", "EL": "Υπερπουλημένο", "DE": "Überverkauft"},
    "overbought": {"EN": "Overbought", "EL": "Υπεραγορασμένο", "DE": "Überkauft"},
    "neutral": {"EN": "Neutral", "EL": "Ουδέτερο", "DE": "Neutral"},
    "bullish": {"EN": "Bullish", "EL": "Ανοδικό", "DE": "Bullisch"},
    "bearish": {"EN": "Bearish", "EL": "Καθοδικό", "DE": "Bärisch"},
    "yes": {"EN": "Yes", "EL": "Ναι", "DE": "Ja"},
    "no": {"EN": "No", "EL": "Όχι", "DE": "Nein"},
    "above": {"EN": "Above SMA200", "EL": "Πάνω SMA200", "DE": "Über SMA200"},
    "below": {"EN": "Below SMA200", "EL": "Κάτω SMA200", "DE": "Unter SMA200"},
    "rising": {"EN": "Rising", "EL": "Ανοδικό", "DE": "Steigend"},
    "falling": {"EN": "Falling", "EL": "Καθοδικό", "DE": "Fallend"},
    "add_to_comparison": {"EN": "Add to comparison", "EL": "Προσθήκη για σύγκριση", "DE": "Zum Vergleich hinzufügen"},
    "search_any_ticker": {"EN": "Search any ticker (e.g. APLD, MSTR, IBIT…)", "EL": "Αναζήτηση οποιουδήποτε ticker", "DE": "Beliebigen Ticker suchen"},
}

def _lang():
    return st.session_state.get("LANG", "EN")

def t(key: str) -> str:
    return T.get(key, {}).get(_lang(), T.get(key, {}).get("EN", key))

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

DEFAULTS = {
    "LANG": "EN",
    "search_val": "",
    "period_code": "1y",
    "analysis_results": {},
    "analysis_symbols": [],
    "analysis_has_run": False,
    "last_period_used": "1y",
    "live_search": "",
    "ac_selection": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# SYMBOLS DATABASE
# ═══════════════════════════════════════════════════════════════

SYMBOL_DB = [
    ("AAPL", "Apple Inc.", "Stock"), ("MSFT", "Microsoft Corp.", "Stock"), ("NVDA", "NVIDIA Corp.", "Stock"),
    ("GOOGL", "Alphabet Inc.", "Stock"), ("AMZN", "Amazon.com Inc.", "Stock"), ("META", "Meta Platforms", "Stock"),
    ("TSLA", "Tesla Inc.", "Stock"), ("BRK-B", "Berkshire Hathaway", "Stock"), ("JPM", "JPMorgan Chase", "Stock"),
    ("V", "Visa Inc.", "Stock"), ("UNH", "UnitedHealth Group", "Stock"), ("JNJ", "Johnson & Johnson", "Stock"),
    ("XOM", "Exxon Mobil", "Stock"), ("WMT", "Walmart Inc.", "Stock"), ("MA", "Mastercard Inc.", "Stock"),
    ("PG", "Procter & Gamble", "Stock"), ("HD", "Home Depot", "Stock"), ("CVX", "Chevron Corp.", "Stock"),
    ("MRK", "Merck & Co.", "Stock"), ("ABBV", "AbbVie Inc.", "Stock"), ("KO", "Coca-Cola Co.", "Stock"),
    ("PEP", "PepsiCo Inc.", "Stock"), ("AVGO", "Broadcom Inc.", "Stock"), ("COST", "Costco Wholesale", "Stock"),
    ("NFLX", "Netflix Inc.", "Stock"), ("CRM", "Salesforce Inc.", "Stock"), ("AMD", "Advanced Micro Devices", "Stock"),
    ("ORCL", "Oracle Corp.", "Stock"), ("INTC", "Intel Corp.", "Stock"), ("QCOM", "Qualcomm Inc.", "Stock"),
    ("DIS", "Walt Disney Co.", "Stock"), ("BA", "Boeing Co.", "Stock"), ("CAT", "Caterpillar Inc.", "Stock"),
    ("GS", "Goldman Sachs", "Stock"), ("MS", "Morgan Stanley", "Stock"), ("BAC", "Bank of America", "Stock"),
    ("WFC", "Wells Fargo", "Stock"), ("C", "Citigroup Inc.", "Stock"), ("UBER", "Uber Technologies", "Stock"),
    ("PYPL", "PayPal Holdings", "Stock"), ("COIN", "Coinbase Global", "Stock"), ("PLTR", "Palantir Technologies", "Stock"),
    ("ARM", "Arm Holdings", "Stock"), ("SMCI", "Super Micro Computer", "Stock"), ("APLD", "Applied Digital Corp.", "Stock"),
    ("MSTR", "MicroStrategy Inc.", "Stock"), ("IBIT", "iShares Bitcoin Trust", "ETF"), ("FBTC", "Fidelity Bitcoin ETF", "ETF"),
    ("RIVN", "Rivian Automotive", "Stock"), ("LCID", "Lucid Group", "Stock"), ("NIO", "NIO Inc.", "Stock"),
    ("BABA", "Alibaba Group", "Stock"), ("TSM", "Taiwan Semiconductor", "Stock"), ("ASML", "ASML Holding", "Stock"),
    ("SAP", "SAP SE", "Stock"), ("SNOW", "Snowflake Inc.", "Stock"), ("DDOG", "Datadog Inc.", "Stock"),
    ("NET", "Cloudflare Inc.", "Stock"), ("CRWD", "CrowdStrike Holdings", "Stock"), ("ZS", "Zscaler Inc.", "Stock"),
    ("PANW", "Palo Alto Networks", "Stock"), ("MDB", "MongoDB Inc.", "Stock"), ("TTD", "The Trade Desk", "Stock"),
    ("U", "Unity Software", "Stock"), ("RBLX", "Roblox Corp.", "Stock"), ("ABNB", "Airbnb Inc.", "Stock"),
    ("BTC-USD", "Bitcoin", "Crypto"), ("ETH-USD", "Ethereum", "Crypto"), ("SOL-USD", "Solana", "Crypto"),
    ("BNB-USD", "BNB", "Crypto"), ("XRP-USD", "XRP", "Crypto"), ("ADA-USD", "Cardano", "Crypto"),
    ("AVAX-USD", "Avalanche", "Crypto"), ("DOGE-USD", "Dogecoin", "Crypto"), ("DOT-USD", "Polkadot", "Crypto"),
    ("LINK-USD", "Chainlink", "Crypto"), ("MATIC-USD", "Polygon", "Crypto"), ("LTC-USD", "Litecoin", "Crypto"),
    ("SPY", "SPDR S&P 500 ETF", "ETF"), ("QQQ", "Invesco QQQ Trust", "ETF"),
    ("IWM", "iShares Russell 2000", "ETF"), ("GLD", "SPDR Gold Shares", "ETF"), ("TLT", "iShares 20Y+ Treasury", "ETF"),
    ("VTI", "Vanguard Total Stock Market", "ETF"), ("VOO", "Vanguard S&P 500", "ETF"),
    ("XLK", "Technology Select Sector SPDR", "ETF"), ("XLF", "Financial Select Sector SPDR", "ETF"),
    ("XLE", "Energy Select Sector SPDR", "ETF"), ("ARKK", "ARK Innovation ETF", "ETF"),
    ("HTO.AT", "Hellenic Telecom", "Stock"), ("ETE.AT", "National Bank of Greece", "Stock"),
    ("OPAP.AT", "OPAP SA", "Stock"), ("EUROB.AT", "Eurobank Ergasias", "Stock"),
    ("ALPHA.AT", "Alpha Services & Holdings", "Stock"), ("PPC.AT", "Public Power Corp.", "Stock"),
]

TICKER_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "GLD", "TLT", "^VIX"]
PERIOD_CODES = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

def search_symbols(query: str, max_results: int = 8) -> list:
    if not query or len(query.strip()) < 1:
        return []
    q = query.upper().strip()
    q_lower = query.lower().strip()
    results = []
    exact_sym = []
    prefix_sym = []
    name_match = []
    for sym, name, typ in SYMBOL_DB:
        if sym.upper() == q:
            exact_sym.append((sym, name, typ))
        elif sym.upper().startswith(q):
            prefix_sym.append((sym, name, typ))
        elif q_lower in name.lower():
            name_match.append((sym, name, typ))
    results = exact_sym + prefix_sym + name_match
    seen = set()
    unique = []
    for item in results:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique[:max_results]

def get_period_maps():
    labels = t("periods")
    return labels, PERIOD_CODES, dict(zip(labels, PERIOD_CODES))

def period_label_from_code(code: str):
    labels, codes, _ = get_period_maps()
    for lbl, c in zip(labels, codes):
        if c == code:
            return lbl
    return labels[3]

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def safe_last(series):
    try:
        if series is None or len(series) == 0:
            return np.nan
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan

def safe_float(value, mult=1):
    try:
        v = float(value)
        return v * mult if not np.isnan(v) else np.nan
    except Exception:
        return np.nan

def fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_pct(v):
    try:
        return f"{float(v):+.1f}%" if not np.isnan(float(v)) else "N/A"
    except Exception:
        return "N/A"

def fmt_val(v, d=2):
    try:
        f = float(v)
        return f"{f:.{d}f}" if not np.isnan(f) else "N/A"
    except Exception:
        return "N/A"

def fmt_financial_value(v):
    if pd.isna(v):
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f}K"
    return f"{v:,.0f}"

def risk_level(beta):
    if np.isnan(beta):
        return "Unknown", "gray"
    if beta < 0.8:
        return "Low Risk", "green"
    if beta < 1.3:
        return "Mid Risk", "yellow"
    return "High Risk", "red"

def pct_from_52w(price, high, low):
    pct_hi = (price - high) / high * 100 if high else np.nan
    pct_lo = (price - low) / low * 100 if low else np.nan
    pos = (price - low) / (high - low) * 100 if (high and low and high != low) else np.nan
    return pct_hi, pct_lo, pos

def sig_card(label, value, ok):
    color = "#00d09c" if ok else "#ff4f6a"
    icon = "▲" if ok else "▼"
    return (
        f'<div class="sig-card">'
        f'<div class="sig-label">{label}</div>'
        f'<div class="sig-value" style="color:{color}">{icon} {value}</div>'
        f'</div>'
    )

def color_signal_df(val):
    if val in ("Optimal", "Strong", "Strong Buy Zone", "Attractive"):
        return "color:#00d09c;font-weight:600"
    if val in ("Caution", "Weak"):
        return "color:#ff4f6a;font-weight:600"
    if val == "Neutral":
        return "color:#f0b429"
    return "color:#8892a4"

# ═══════════════════════════════════════════════════════════════
# CSS — Bloomberg-style dark institutional UI
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* ── App background ── */
.stApp {
    background: #080c14;
    color: #c9d1de;
}
.stApp > header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 1320px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1420; }
::-webkit-scrollbar-thumb { background: #243044; border-radius: 3px; }

/* ══════════════════════════════════════
   TICKER BAR
══════════════════════════════════════ */
.ticker-bar {
    background: #0a0e1a;
    border-bottom: 1px solid #1a2335;
    padding: 7px 0;
    overflow: hidden;
    margin-bottom: 0;
    width: 100%;
}
.ticker-track {
    display: flex;
    gap: 0;
    animation: scroll-left 55s linear infinite;
    white-space: nowrap;
}
.ticker-track:hover { animation-play-state: paused; }
@keyframes scroll-left { from { transform: translateX(0) } to { transform: translateX(-50%) } }
.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: #8892a4;
    padding: 0 18px;
    border-right: 1px solid #1a2335;
}
.ticker-sym { color: #e8edf5; font-weight: 600; letter-spacing: .03em; }
.ticker-up { color: #00d09c; }
.ticker-down { color: #ff4f6a; }

/* ══════════════════════════════════════
   HERO — Advanced Trading Background
══════════════════════════════════════ */
.hero-wrap {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, #080c14 0%, #0a0f1e 100%);
    border-bottom: 1px solid #1a2335;
    padding: 48px 32px 36px;
    text-align: center;
    margin-bottom: 0;
}
.hero-canvas {
    position: absolute;
    inset: 0;
    pointer-events: none;
    opacity: 1;
}
.hero-inner {
    position: relative;
    z-index: 2;
    max-width: 820px;
    margin: 0 auto;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f4fa;
    letter-spacing: -.02em;
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero-title span { color: #3b82f6; }
.hero-sub {
    color: #8892a4;
    font-size: .88rem;
    font-weight: 400;
    max-width: 620px;
    margin: 0 auto 24px;
    line-height: 1.65;
}
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-pill {
    border: 1px solid #1e2d42;
    background: rgba(15,23,42,.65);
    color: #6b7a8f;
    border-radius: 4px;
    padding: 5px 11px;
    font-size: .70rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: .05em;
}

/* ══════════════════════════════════════
   SEARCH SECTION
══════════════════════════════════════ */
.search-section {
    background: #0a0e1a;
    border-bottom: 1px solid #1a2335;
    padding: 28px 32px 20px;
}
.search-inner {
    max-width: 820px;
    margin: 0 auto;
}
.search-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #3b5070;
    margin-bottom: 10px;
}

/* Streamlit input overrides */
.search-section .stTextInput > div > div {
    background: #0d1520 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 6px !important;
    transition: border-color .2s;
}
.search-section .stTextInput > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
}
.search-section .stTextInput input {
    color: #e8edf5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .92rem !important;
}
.search-section .stTextInput input::placeholder {
    color: #3b5070 !important;
}
.search-section .stSelectbox > div > div {
    background: #0d1520 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 6px !important;
}
.search-section .stSelectbox div[data-baseweb="select"] > div {
    color: #e8edf5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82rem !important;
}

/* ── Analyze button ── */
.search-section .stButton > button {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    letter-spacing: .04em !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: background .2s !important;
}
.search-section .stButton > button:hover {
    background: #2563eb !important;
}

/* ── Autocomplete ── */
.ac-box {
    background: #0d1520;
    border: 1px solid #1e2d42;
    border-top: none;
    border-radius: 0 0 6px 6px;
    overflow: hidden;
    margin-top: -2px;
    box-shadow: 0 12px 32px rgba(0,0,0,.5);
}
.ac-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    text-decoration: none !important;
    color: #c9d1de !important;
    border-bottom: 1px solid #121c2c;
    cursor: pointer;
    transition: background .12s;
}
.ac-item:last-child { border-bottom: none; }
.ac-item:hover { background: #111d2f; }
.ac-sym {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .84rem;
    font-weight: 600;
    color: #e8edf5;
}
.ac-name { font-size: .76rem; color: #6b7a8f; margin-top: 1px; }
.ac-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem;
    padding: 3px 8px;
    border-radius: 3px;
    border: 1px solid;
    letter-spacing: .04em;
}
.ac-badge-stock { color: #3b82f6; border-color: #1d4ed8; background: rgba(59,130,246,.08); }
.ac-badge-crypto { color: #f59e0b; border-color: #b45309; background: rgba(245,158,11,.08); }
.ac-badge-etf { color: #10b981; border-color: #065f46; background: rgba(16,185,129,.08); }

/* ── Quick picks ── */
.qp-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
}
.qp-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #3b5070;
    margin-bottom: 8px;
}
.qp-pill {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 4px;
    border: 1px solid #1e2d42;
    background: #0d1520;
    color: #8892a4 !important;
    text-decoration: none !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .70rem;
    letter-spacing: .03em;
    transition: all .12s;
}
.qp-pill:hover {
    background: #111d2f;
    border-color: #2d4a6a;
    color: #c9d1de !important;
}

/* ══════════════════════════════════════
   RESULTS AREA
══════════════════════════════════════ */
.results-wrap {
    padding: 24px 0;
}

/* ── Section divider ── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px;
}
.section-hdr-line {
    flex: 1;
    height: 1px;
    background: #1a2335;
}
.section-hdr-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .66rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #3b5070;
    white-space: nowrap;
}

/* ── Asset header ── */
.asset-header {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.asset-sym {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: #f0f4fa;
    font-weight: 600;
    letter-spacing: .02em;
}
.asset-name { color: #6b7a8f; font-size: .88rem; margin-top: 3px; font-weight: 400; }
.asset-meta {
    color: #3b5070;
    font-size: .74rem;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: .03em;
}

/* ── Verdict badge ── */
.verdict-card {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: right;
    height: 100%;
}
.verdict-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: .04em;
}
.verdict-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem;
    color: #6b7a8f;
    margin-top: 4px;
}

/* ── Metric cards ── */
.metric-card {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: .64rem;
    color: #3b5070;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #e8edf5;
    margin-top: 4px;
}
.metric-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: #6b7a8f;
    margin-top: 3px;
}

/* ── Signal cards ── */
.sig-card {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-radius: 6px;
    padding: 11px 13px;
    margin-bottom: 8px;
}
.sig-label {
    font-size: .63rem;
    color: #3b5070;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-family: 'IBM Plex Mono', monospace;
}
.sig-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .9rem;
    font-weight: 600;
    margin-top: 3px;
}

/* ── 52W range bar ── */
.range-card {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.range-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .63rem;
    color: #3b5070;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 8px;
}
.range-bar-bg {
    background: #111d2f;
    border-radius: 2px;
    height: 5px;
    width: 100%;
    margin: 8px 0 4px;
}
.range-bar-fill {
    background: linear-gradient(90deg, #1d4ed8, #00d09c);
    border-radius: 2px;
    height: 5px;
}

/* ── Business description ── */
.biz-desc {
    background: #0a0f1e;
    border: 1px solid #1a2335;
    border-left: 3px solid #1d4ed8;
    border-radius: 6px;
    padding: 16px 18px;
    color: #8892a4;
    font-size: .84rem;
    line-height: 1.75;
    margin-bottom: 14px;
}

/* ── Summary / assessment ── */
.summary-box {
    border-radius: 6px;
    padding: 20px 22px;
    line-height: 1.8;
    font-size: .88rem;
}
.summary-box h2, .summary-box h3 {
    color: #c9d1de;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: .02em;
}

/* ── Streamlit overrides global ── */
.stButton > button {
    background: #0d1520 !important;
    color: #8892a4 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important;
}
.stButton > button:hover {
    background: #111d2f !important;
    color: #c9d1de !important;
    border-color: #2d4a6a !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 2px;
    border-bottom: 1px solid #1a2335;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7a8f;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .76rem;
    letter-spacing: .06em;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1a2335;
    border-radius: 6px;
    overflow: hidden;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6) !important;
    border-radius: 2px !important;
}
.stProgress > div > div {
    background: #111d2f !important;
    border-radius: 2px !important;
}
div[data-testid="stExpander"] {
    background: #0a0f1e;
    border: 1px solid #1a2335 !important;
    border-radius: 6px !important;
}

/* ── Footer ── */
.footer-bar {
    border-top: 1px solid #1a2335;
    padding: 16px 0;
    text-align: center;
    color: #3b5070;
    font-size: .72rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: .04em;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TICKER BAR
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ticker_prices(symbols: tuple) -> dict:
    results = {}
    for sym in symbols:
        try:
            tkr = yf.Ticker(sym)
            hist = tkr.history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]
                curr = hist["Close"].iloc[-1]
                chg = ((curr - prev) / prev) * 100
                results[sym] = {"price": float(curr), "chg": float(chg)}
            elif len(hist) == 1:
                results[sym] = {"price": float(hist["Close"].iloc[-1]), "chg": 0.0}
        except Exception:
            pass
    return results

def build_ticker_html(prices: dict) -> str:
    items = []
    label_map = {"^VIX": "VIX", "SPY": "S&P500", "QQQ": "NASDAQ"}
    for sym, data in prices.items():
        lbl = label_map.get(sym, sym.replace("-USD", "").replace(".AT", ""))
        price = data["price"]
        chg = data["chg"]
        cls = "ticker-up" if chg >= 0 else "ticker-down"
        arrow = "▲" if chg >= 0 else "▼"
        price_str = f"${price:,.2f}" if price >= 1 else f"${price:.4f}"
        items.append(
            f'<span class="ticker-item">'
            f'<span class="ticker-sym">{lbl}</span>'
            f'<span>{price_str}</span>'
            f'<span class="{cls}">{arrow} {abs(chg):.2f}%</span>'
            f'</span>'
        )
    content = "".join(items) * 2
    return f'<div class="ticker-bar"><div class="ticker-track">{content}</div></div>'

# ═══════════════════════════════════════════════════════════════
# TECHNICALS
# ═══════════════════════════════════════════════════════════════

def compute_rsi(close, w=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    l_safe = l.copy()
    l_safe[l_safe == 0] = np.nan
    rs = g / l_safe
    return 100 - (100 / (1 + rs))

def compute_macd(close):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

def compute_bollinger(close, w=20):
    sma = close.rolling(w).mean()
    std = close.rolling(w).std()
    up = sma + 2 * std
    lo = sma - 2 * std
    return sma, up, lo, (close - lo) / (up - lo)

def compute_atr(hist, w=14):
    hl = hist["High"] - hist["Low"]
    hc = (hist["High"] - hist["Close"].shift()).abs()
    lc = (hist["Low"] - hist["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(w).mean()

def compute_stochastic(hist, k=14, d=3):
    lo = hist["Low"].rolling(k).min()
    hi = hist["High"].rolling(k).max()
    denom = (hi - lo).copy()
    denom[denom == 0] = np.nan
    sk = 100 * (hist["Close"] - lo) / denom
    return sk, sk.rolling(d).mean()

def compute_obv(hist):
    direction = np.sign(hist["Close"].diff()).copy()
    direction.iat[0] = 0
    return (direction * hist["Volume"]).cumsum()

def technical_signals(hist):
    close = hist["Close"]
    volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(0, index=hist.index)
    rsi = compute_rsi(close)
    macd, sig, mhist = compute_macd(close)
    sma20, bb_up, bb_lo, pct_b = compute_bollinger(close)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    stk, _ = compute_stochastic(hist)
    atr = compute_atr(hist)
    obv = compute_obv(hist)
    vol_ma = volume.rolling(20).mean()
    c = float(close.iloc[-1]) if len(close) > 0 and pd.notna(close.iloc[-1]) else np.nan

    atr_last = safe_last(atr)
    atr_pct = (atr_last / c * 100) if (c and not np.isnan(c) and not np.isnan(atr_last)) else np.nan

    vol_last = safe_last(volume)
    vol_ma_last = safe_last(vol_ma)
    vol_ratio = (vol_last / vol_ma_last) if (vol_ma_last and not np.isnan(vol_ma_last) and vol_ma_last != 0) else np.nan

    obv_trend = "Rising" if (len(obv) > 20 and pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-20]) and obv.iloc[-1] > obv.iloc[-20]) else "Falling"

    macd_last = safe_last(macd)
    sig_last = safe_last(sig)
    macd_bull = (not np.isnan(macd_last) and not np.isnan(sig_last) and macd_last > sig_last)

    sma50_last = safe_last(sma50)
    sma200_last = safe_last(sma200)
    golden = (not np.isnan(sma50_last) and not np.isnan(sma200_last) and sma50_last > sma200_last)

    sma20_last = safe_last(sma20)
    price_vs_20 = (not np.isnan(c) and not np.isnan(sma20_last) and c > sma20_last)
    price_vs_50 = (not np.isnan(c) and not np.isnan(sma50_last) and c > sma50_last)
    price_vs_200 = (not np.isnan(c) and not np.isnan(sma200_last) and c > sma200_last)

    high_52 = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
    low_52 = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()

    return {
        "_close": close, "_volume": volume, "_rsi": rsi, "_macd": macd, "_macd_sig": sig,
        "_macd_hist": mhist, "_sma20": sma20, "_sma50": sma50, "_sma200": sma200,
        "_bb_up": bb_up, "_bb_lo": bb_lo, "_stk": stk, "_atr": atr, "_obv": obv, "_vol_ma": vol_ma,
        "RSI": safe_last(rsi),
        "MACD": macd_last,
        "MACD_Signal": sig_last,
        "MACD_Bullish": macd_bull,
        "Stochastic_K": safe_last(stk),
        "BB_PctB": safe_last(pct_b),
        "ATR_Pct": atr_pct,
        "OBV_Trend": obv_trend,
        "Price_vs_SMA20": price_vs_20,
        "Price_vs_SMA50": price_vs_50,
        "Price_vs_SMA200": price_vs_200,
        "Golden_Cross": golden,
        "Volume_vs_MA": vol_ratio,
        "Price_52w_High": float(high_52) if pd.notna(high_52) else np.nan,
        "Price_52w_Low": float(low_52) if pd.notna(low_52) else np.nan,
    }

# ═══════════════════════════════════════════════════════════════
# INDICATOR META
# ═══════════════════════════════════════════════════════════════

INDICATOR_META = {
    "RSI": {
        "desc": {
            "EN": "Relative Strength Index — measures short-term price momentum",
            "EL": "Relative Strength Index — μετρά τη βραχυπρόθεσμη ορμή τιμής",
            "DE": "Relative Strength Index — misst kurzfristiges Kursmomentum",
        },
        "optimal": "40 – 60",
        "assessment": {
            "EN": "Below 30 = oversold / potential bounce. Above 70 = overbought / potential pullback.",
            "EL": "Κάτω από 30 = υπερπουλημένο / πιθανή ανάκαμψη. Πάνω από 70 = υπεραγορασμένο / πιθανή διόρθωση.",
            "DE": "Unter 30 = überverkauft / mögliche Erholung. Über 70 = überkauft / möglicher Rückgang.",
        },
        "numeric_optimal": (40, 60),
        "buy_max": 30, "sell_min": 70, "higher_better": None,
        "category": {"EN": "Technical", "EL": "Τεχνικά", "DE": "Technisch"},
    },
    "P/E Ratio": {
        "desc": {
            "EN": "Price-to-Earnings — how many times earnings investors pay for the stock",
            "EL": "Price-to-Earnings — πόσες φορές τα κέρδη πληρώνει η αγορά",
            "DE": "Kurs-Gewinn-Verhältnis — wie oft der Gewinn im Preis enthalten ist",
        },
        "optimal": "15 – 25",
        "assessment": {
            "EN": "Very low may mean undervaluation or business stress. Very high requires strong growth to justify.",
            "EL": "Πολύ χαμηλό ίσως σημαίνει υποτίμηση ή προβλήματα. Πολύ υψηλό απαιτεί ισχυρή ανάπτυξη.",
            "DE": "Sehr niedrig = mögliche Unterbewertung. Sehr hoch = starkes Wachstum erforderlich.",
        },
        "numeric_optimal": (15, 25), "buy_max": 15, "sell_min": 35, "higher_better": False,
        "category": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    },
    "Forward P/E": {
        "desc": {
            "EN": "Forward P/E — uses expected future earnings instead of trailing",
            "EL": "Forward P/E — χρησιμοποιεί αναμενόμενα μελλοντικά κέρδη",
            "DE": "Forward-KGV — nutzt erwartete zukünftige Gewinne",
        },
        "optimal": "12 – 22",
        "assessment": {
            "EN": "If below trailing P/E, market expects earnings growth. High forward P/E = rich expectations.",
            "EL": "Αν χαμηλότερο από trailing P/E, η αγορά αναμένει αύξηση κερδών.",
            "DE": "Liegt es unter dem aktuellen KGV, erwartet der Markt Gewinnwachstum.",
        },
        "numeric_optimal": (12, 22), "buy_max": 12, "sell_min": 30, "higher_better": False,
        "category": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    },
    "Price/Book": {
        "desc": {
            "EN": "Price-to-Book — compares market price to book value per share",
            "EL": "Price-to-Book — συγκρίνει τιμή με λογιστική αξία",
            "DE": "Kurs-Buchwert-Verhältnis — Marktpreis vs. Buchwert je Aktie",
        },
        "optimal": "1 – 3",
        "assessment": {
            "EN": "Below 1 may signal undervaluation. Very high implies premium market pricing.",
            "EL": "Κάτω από 1 ίσως δείχνει υποτίμηση. Πολύ υψηλό σημαίνει premium αποτίμηση.",
            "DE": "Unter 1 = mögliche Unterbewertung. Sehr hoch = Premium-Bewertung.",
        },
        "numeric_optimal": (1, 3), "buy_max": 1, "sell_min": 6, "higher_better": False,
        "category": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    },
    "Price/Sales": {
        "desc": {
            "EN": "Price-to-Sales — market cap divided by annual revenue",
            "EL": "Price-to-Sales — κεφαλαιοποίηση διά ετήσιων εσόδων",
            "DE": "Kurs-Umsatz-Verhältnis — Marktkapitalisierung geteilt durch Umsatz",
        },
        "optimal": "1 – 4",
        "assessment": {
            "EN": "Useful when profitability is weak. Lower values indicate more reasonable revenue-based valuation.",
            "EL": "Χρήσιμο όταν η κερδοφορία είναι αδύναμη. Χαμηλότερες τιμές = πιο λογική αποτίμηση.",
            "DE": "Nützlich bei schwacher Profitabilität. Niedrigere Werte = vernünftigere Bewertung.",
        },
        "numeric_optimal": (1, 4), "buy_max": 1, "sell_min": 8, "higher_better": False,
        "category": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    },
    "EV/EBITDA": {
        "desc": {
            "EN": "Enterprise Value / EBITDA — total enterprise value vs operating earnings",
            "EL": "EV/EBITDA — αξία επιχείρησης προς λειτουργικά κέρδη",
            "DE": "EV/EBITDA — Unternehmenswert vs. operativer Gewinn",
        },
        "optimal": "6 – 14",
        "assessment": {
            "EN": "Below 10 is often attractive. Above 20 is expensive unless growth is exceptional.",
            "EL": "Κάτω από 10 είναι συχνά ελκυστικό. Πάνω από 20 είναι ακριβό εκτός από εξαιρετική ανάπτυξη.",
            "DE": "Unter 10 ist oft attraktiv. Über 20 ist teuer ohne außergewöhnliches Wachstum.",
        },
        "numeric_optimal": (6, 14), "buy_max": 6, "sell_min": 20, "higher_better": False,
        "category": {"EN": "Valuation", "EL": "Αποτίμηση", "DE": "Bewertung"},
    },
    "ROE %": {
        "desc": {
            "EN": "Return on Equity — profit generated per unit of shareholders' capital",
            "EL": "Return on Equity — κέρδος ανά μονάδα ιδίων κεφαλαίων",
            "DE": "Eigenkapitalrendite — Gewinn pro Einheit Eigenkapital",
        },
        "optimal": "15 – 35%",
        "assessment": {
            "EN": "Above 15% is strong. Very high ROE with high debt may be misleading — check leverage.",
            "EL": "Πάνω από 15% είναι ισχυρό. Πολύ υψηλό ROE με υψηλό χρέος μπορεί να παραπλανά.",
            "DE": "Über 15% ist stark. Sehr hohes ROE mit hohen Schulden kann irreführend sein.",
        },
        "numeric_optimal": (15, 35), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "ROA %": {
        "desc": {
            "EN": "Return on Assets — profit generated relative to total asset base",
            "EL": "Return on Assets — κέρδος σε σχέση με το σύνολο ενεργητικού",
            "DE": "Gesamtkapitalrendite — Gewinn im Verhältnis zu den Gesamtvermögen",
        },
        "optimal": "5 – 20%",
        "assessment": {
            "EN": "Above 5% is healthy for most sectors. Capital-intensive sectors typically have lower ROA.",
            "EL": "Πάνω από 5% είναι υγιές για τους περισσότερους κλάδους.",
            "DE": "Über 5% ist für die meisten Sektoren gesund.",
        },
        "numeric_optimal": (5, 20), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Debt/Equity": {
        "desc": {
            "EN": "Debt-to-Equity — total debt divided by shareholders' equity",
            "EL": "Debt-to-Equity — συνολικό χρέος διά ιδίων κεφαλαίων",
            "DE": "Verschuldungsgrad — Gesamtschulden geteilt durch Eigenkapital",
        },
        "optimal": "0 – 1",
        "assessment": {
            "EN": "Below 1 is conservative. Above 2 significantly increases financial risk.",
            "EL": "Κάτω από 1 είναι συντηρητικό. Πάνω από 2 αυξάνει σημαντικά τον χρηματοοικονομικό κίνδυνο.",
            "DE": "Unter 1 ist konservativ. Über 2 erhöht das Finanzrisiko erheblich.",
        },
        "numeric_optimal": (0, 1), "buy_max": 0.5, "sell_min": 2, "higher_better": False,
        "category": {"EN": "Risk", "EL": "Κίνδυνος", "DE": "Risiko"},
    },
    "Current Ratio": {
        "desc": {
            "EN": "Current Ratio — current assets divided by current liabilities",
            "EL": "Current Ratio — κυκλοφορούν ενεργητικό διά τρεχουσών υποχρεώσεων",
            "DE": "Liquiditätsgrad — kurzfristige Aktiva geteilt durch kurzfristige Passiva",
        },
        "optimal": "1.5 – 3",
        "assessment": {
            "EN": "Below 1 signals potential liquidity pressure. Above 3 may indicate inefficient capital deployment.",
            "EL": "Κάτω από 1 δείχνει πιθανή πίεση ρευστότητας. Πάνω από 3 ίσως είναι αναποτελεσματικό.",
            "DE": "Unter 1 = Liquiditätsdruck. Über 3 = möglicherweise ineffiziente Kapitalnutzung.",
        },
        "numeric_optimal": (1.5, 3), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Gross Margin %": {
        "desc": {
            "EN": "Gross Margin — revenue minus cost of goods sold, as % of revenue",
            "EL": "Μεικτό Περιθώριο — έσοδα μείον κόστος πωλήσεων ως % εσόδων",
            "DE": "Bruttomarge — Umsatz minus Herstellungskosten als % des Umsatzes",
        },
        "optimal": "40 – 80%",
        "assessment": {
            "EN": "Higher gross margin usually signals stronger pricing power. Sector context matters greatly.",
            "EL": "Υψηλότερο gross margin σημαίνει ισχυρότερο pricing power. Ο κλάδος παίζει ρόλο.",
            "DE": "Höhere Bruttomarge bedeutet stärkere Preissetzungsmacht. Sektorkontext ist entscheidend.",
        },
        "numeric_optimal": (40, 80), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Operating Margin %": {
        "desc": {
            "EN": "Operating Margin — operating income as % of revenue",
            "EL": "Λειτουργικό Περιθώριο — λειτουργικά κέρδη ως % εσόδων",
            "DE": "Operative Marge — operatives Ergebnis als % des Umsatzes",
        },
        "optimal": "15 – 40%",
        "assessment": {
            "EN": "Above 15% is typically healthy. Low or negative values indicate structural cost pressure.",
            "EL": "Πάνω από 15% είναι συνήθως υγιές. Χαμηλές ή αρνητικές τιμές δείχνουν πίεση κόστους.",
            "DE": "Über 15% gilt als gesund. Niedrig oder negativ = struktureller Kostendruck.",
        },
        "numeric_optimal": (15, 40), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Net Margin %": {
        "desc": {
            "EN": "Net Margin — net profit as % of revenue (after all costs, taxes, interest)",
            "EL": "Καθαρό Περιθώριο — καθαρό κέρδος ως % εσόδων (μετά από όλα τα έξοδα)",
            "DE": "Nettomarge — Nettogewinn als % des Umsatzes (nach allen Kosten)",
        },
        "optimal": "10 – 30%",
        "assessment": {
            "EN": "Above 10% is healthy. Negative margin = company is loss-making.",
            "EL": "Πάνω από 10% είναι υγιές. Αρνητικό περιθώριο = ζημιογόνα εταιρεία.",
            "DE": "Über 10% ist gesund. Negative Marge = Verlustunternehmen.",
        },
        "numeric_optimal": (10, 30), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Beta": {
        "desc": {
            "EN": "Beta — price volatility relative to the broad market (market = 1.0)",
            "EL": "Beta — μεταβλητότητα σε σχέση με τη συνολική αγορά (αγορά = 1.0)",
            "DE": "Beta — Kursvolatilität relativ zum Gesamtmarkt (Markt = 1,0)",
        },
        "optimal": "0.8 – 1.3",
        "assessment": {
            "EN": "Below 0.8 = defensive. Above 1.3 = aggressive and volatile. Negative beta = inverse to market.",
            "EL": "Κάτω από 0.8 = αμυντικό. Πάνω από 1.3 = επιθετικό και volatile. Αρνητική beta = αντίστροφο.",
            "DE": "Unter 0,8 = defensiv. Über 1,3 = aggressiv und volatil. Negative Beta = gegenläufig.",
        },
        "numeric_optimal": (0.8, 1.3), "buy_max": None, "sell_min": None, "higher_better": None,
        "category": {"EN": "Risk", "EL": "Κίνδυνος", "DE": "Risiko"},
    },
    "Dividend Yield %": {
        "desc": {
            "EN": "Dividend Yield — annual dividend payout as % of current stock price",
            "EL": "Dividend Yield — ετήσιο μέρισμα ως % τρέχουσας τιμής",
            "DE": "Dividendenrendite — jährliche Dividende als % des aktuellen Kurses",
        },
        "optimal": "2 – 6%",
        "assessment": {
            "EN": "Attractive only if earnings and payout are sustainable. Very high yield can signal distress.",
            "EL": "Ελκυστικό μόνο αν τα κέρδη και το payout είναι βιώσιμα. Πολύ υψηλό yield μπορεί να σημαίνει κίνδυνο.",
            "DE": "Attraktiv nur bei nachhaltigen Gewinnen. Sehr hohe Rendite kann auf Stress hinweisen.",
        },
        "numeric_optimal": (2, 6), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Payout Ratio %": {
        "desc": {
            "EN": "Payout Ratio — percentage of earnings paid out as dividends",
            "EL": "Payout Ratio — ποσοστό κερδών που διανέμεται ως μέρισμα",
            "DE": "Ausschüttungsquote — Anteil des Gewinns der als Dividende gezahlt wird",
        },
        "optimal": "20 – 60%",
        "assessment": {
            "EN": "Above 100% is typically unsustainable. Below 20% suggests room for dividend growth.",
            "EL": "Πάνω από 100% είναι συνήθως μη βιώσιμο. Κάτω από 20% σημαίνει χώρο για αύξηση μερίσματος.",
            "DE": "Über 100% = langfristig nicht nachhaltig. Unter 20% = Spielraum für Dividendenwachstum.",
        },
        "numeric_optimal": (20, 60), "buy_max": None, "sell_min": 100, "higher_better": False,
        "category": {"EN": "Risk", "EL": "Κίνδυνος", "DE": "Risiko"},
    },
    "Revenue Growth %": {
        "desc": {
            "EN": "Revenue Growth — year-over-year change in total revenue",
            "EL": "Revenue Growth — ετήσια μεταβολή συνολικών εσόδων",
            "DE": "Umsatzwachstum — Jahresveränderung des Gesamtumsatzes",
        },
        "optimal": "10 – 40%",
        "assessment": {
            "EN": "Strong positive growth is constructive. Negative growth is a red flag for business momentum.",
            "EL": "Ισχυρή θετική ανάπτυξη είναι θετική. Αρνητική ανάπτυξη είναι red flag.",
            "DE": "Starkes positives Wachstum ist konstruktiv. Negatives Wachstum ist ein Warnsignal.",
        },
        "numeric_optimal": (10, 40), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Growth", "EL": "Ανάπτυξη", "DE": "Wachstum"},
    },
    "EPS Growth %": {
        "desc": {
            "EN": "EPS Growth — year-over-year change in earnings per share",
            "EL": "EPS Growth — ετήσια μεταβολή κερδών ανά μετοχή",
            "DE": "EPS-Wachstum — Jahresveränderung des Gewinns je Aktie",
        },
        "optimal": "15 – 50%",
        "assessment": {
            "EN": "Strong EPS growth supports higher valuation multiples. Deteriorating EPS is a warning sign.",
            "EL": "Ισχυρή αύξηση EPS υποστηρίζει υψηλότερα multiples. Επιδείνωση EPS είναι προειδοποιητικό.",
            "DE": "Starkes EPS-Wachstum rechtfertigt höhere Multiplikatoren. Rückgang = Warnsignal.",
        },
        "numeric_optimal": (15, 50), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Growth", "EL": "Ανάπτυξη", "DE": "Wachstum"},
    },
    "Free Cash Flow Yield %": {
        "desc": {
            "EN": "FCF Yield — free cash flow generated relative to market capitalisation",
            "EL": "FCF Yield — ελεύθερες ταμειακές ροές ως % κεφαλαιοποίησης",
            "DE": "FCF-Rendite — freier Cashflow im Verhältnis zur Marktkapitalisierung",
        },
        "optimal": "4 – 10%",
        "assessment": {
            "EN": "Above 4% is often attractive if cash generation is stable and recurring.",
            "EL": "Πάνω από 4% είναι συχνά ελκυστικό αν η παραγωγή ταμειακών ροών είναι σταθερή.",
            "DE": "Über 4% ist oft attraktiv bei stabiler Cashflow-Generierung.",
        },
        "numeric_optimal": (4, 10), "buy_max": None, "sell_min": None, "higher_better": True,
        "category": {"EN": "Quality", "EL": "Ποιότητα", "DE": "Qualität"},
    },
    "Short Interest %": {
        "desc": {
            "EN": "Short Interest — percentage of float shares sold short",
            "EL": "Short Interest — ποσοστό μετοχών διασποράς που είναι short",
            "DE": "Leerverkaufsquote — Anteil des Streubesitzes mit Leerverkäufen",
        },
        "optimal": "0 – 10%",
        "assessment": {
            "EN": "Above 20% signals significant bearish conviction. Can also trigger short squeeze if reversed.",
            "EL": "Πάνω από 20% σημαίνει έντονο bearish positioning. Μπορεί να προκαλέσει short squeeze.",
            "DE": "Über 20% = starke bärische Überzeugung. Kann auch Short Squeeze auslösen.",
        },
        "numeric_optimal": (0, 10), "buy_max": None, "sell_min": 20, "higher_better": False,
        "category": {"EN": "Risk", "EL": "Κίνδυνος", "DE": "Risiko"},
    },
}

# ═══════════════════════════════════════════════════════════════
# FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════

def extract_fundamentals(info):
    g = lambda k: info.get(k)

    pe = safe_float(g("trailingPE"))
    fpe = safe_float(g("forwardPE"))
    pb = safe_float(g("priceToBook"))
    ps = safe_float(g("priceToSalesTrailing12Months"))
    ev = safe_float(g("enterpriseToEbitda"))
    roe = safe_float(g("returnOnEquity"), 100)
    roa = safe_float(g("returnOnAssets"), 100)

    de = safe_float(g("debtToEquity"))
    de = de / 100 if (not np.isnan(de) and de > 10) else de

    cr = safe_float(g("currentRatio"))
    gm = safe_float(g("grossMargins"), 100)
    om = safe_float(g("operatingMargins"), 100)
    nm = safe_float(g("profitMargins"), 100)
    beta = safe_float(g("beta"))
    dy = safe_float(g("dividendYield"), 100)
    pr = safe_float(g("payoutRatio"), 100)
    rg = safe_float(g("revenueGrowth"), 100)
    eg = safe_float(g("earningsGrowth"), 100)

    fcf = safe_float(g("freeCashflow"))
    mcap = safe_float(g("marketCap"))
    fcf_y = (fcf / mcap * 100) if (not np.isnan(fcf) and not np.isnan(mcap) and mcap > 0) else np.nan

    shares_short = safe_float(g("sharesShort"))
    float_shares = safe_float(g("floatShares"))
    si = (shares_short / float_shares * 100) if (not np.isnan(shares_short) and not np.isnan(float_shares) and float_shares > 0) else np.nan

    tp = safe_float(g("targetMeanPrice"))
    cp = safe_float(g("currentPrice"))
    if np.isnan(cp):
        cp = safe_float(g("regularMarketPrice"))
    upside = ((tp - cp) / cp * 100) if (not np.isnan(tp) and not np.isnan(cp) and cp > 0) else np.nan

    rk = (g("recommendationKey") or "").lower()
    rating_map = {"strongbuy": "Strong Buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "underperform": "Underperform"}

    return {
        "P/E Ratio": pe, "Forward P/E": fpe, "Price/Book": pb, "Price/Sales": ps, "EV/EBITDA": ev,
        "ROE %": roe, "ROA %": roa, "Debt/Equity": de, "Current Ratio": cr,
        "Gross Margin %": gm, "Operating Margin %": om, "Net Margin %": nm,
        "Beta": beta, "Dividend Yield %": dy, "Payout Ratio %": pr,
        "Revenue Growth %": rg, "EPS Growth %": eg,
        "Free Cash Flow Yield %": fcf_y, "Short Interest %": si,
        "_analyst_rating": rating_map.get(rk, rk.title() if rk else "N/A"),
        "_analyst_count": safe_float(g("numberOfAnalystOpinions")),
        "_target_price": tp, "_current_price": cp, "_upside": upside,
        "_market_cap": mcap, "_enterprise_value": safe_float(g("enterpriseValue")),
        "_revenue": safe_float(g("totalRevenue")), "_ebitda": safe_float(g("ebitda")),
        "_net_income": safe_float(g("netIncomeToCommon")),
        "_cash": safe_float(g("totalCash")), "_total_debt": safe_float(g("totalDebt")),
        "_sector": g("sector") or "N/A", "_industry": g("industry") or "N/A",
        "_country": g("country") or "N/A",
        "_name": g("longName") or g("shortName") or "N/A",
        "_summary": (g("longBusinessSummary") or "")[:900],
    }

def score_indicator(name, value):
    try:
        value = float(value)
    except Exception:
        return 1, "N/A"
    if np.isnan(value):
        return 1, "N/A"
    meta = INDICATOR_META.get(name, {})
    opt = meta.get("numeric_optimal")
    bm = meta.get("buy_max")
    sm = meta.get("sell_min")
    hb = meta.get("higher_better")
    if opt and isinstance(opt, tuple):
        lo, hi = opt
        if lo <= value <= hi:
            return 2, "Optimal"
    if bm is not None and value <= bm:
        return 2, "Strong Buy Zone"
    if sm is not None and value >= sm:
        return 0, "Caution"
    if hb is True and opt and value > opt[1]:
        return 2, "Strong"
    if hb is False and opt and value < opt[0]:
        return 2, "Attractive"
    if hb is True and opt and value < opt[0]:
        return 0, "Weak"
    return 1, "Neutral"

def _is_valid(v):
    """Return True if v is a finite number (not nan, not inf, not bool/str)."""
    try:
        f = float(v)
        return not (np.isnan(f) or np.isinf(f))
    except Exception:
        return False

def composite_score(fund, tech):
    cats = {"Valuation": [], "Quality": [], "Growth": [], "Risk": [], "Technical": []}
    for k in ["P/E Ratio", "Forward P/E", "Price/Book", "Price/Sales", "EV/EBITDA"]:
        v = fund.get(k, np.nan)
        if _is_valid(v):
            cats["Valuation"].append(score_indicator(k, v)[0])
    for k in ["ROE %", "ROA %", "Gross Margin %", "Operating Margin %", "Net Margin %", "Current Ratio", "Free Cash Flow Yield %", "Dividend Yield %"]:
        v = fund.get(k, np.nan)
        if _is_valid(v):
            cats["Quality"].append(score_indicator(k, v)[0])
    for k in ["Revenue Growth %", "EPS Growth %"]:
        v = fund.get(k, np.nan)
        if _is_valid(v):
            cats["Growth"].append(score_indicator(k, v)[0])
    for k in ["Debt/Equity", "Beta", "Short Interest %", "Payout Ratio %"]:
        v = fund.get(k, np.nan)
        if _is_valid(v):
            cats["Risk"].append(score_indicator(k, v)[0])
    ts, tc = 0, 0
    rsi = tech.get("RSI", np.nan)
    if _is_valid(rsi):
        ts += score_indicator("RSI", rsi)[0]; tc += 1
    if tech.get("MACD_Bullish") is True: ts += 2; tc += 1
    if tech.get("Golden_Cross") is True: ts += 2; tc += 1
    if tech.get("Price_vs_SMA200") is True: ts += 2; tc += 1
    if tech.get("OBV_Trend") == "Rising": ts += 2; tc += 1
    if tc:
        cats["Technical"].append(ts / tc)
    cat_scores = {c: float(np.mean(v)) * 50 if v else np.nan for c, v in cats.items()}
    valid_cats = [v for v in cat_scores.values() if _is_valid(v)]
    overall = float(np.mean(valid_cats)) if valid_cats else 50.0
    if overall >= 70:
        verdict = ("BUY", "#00d09c")
    elif overall >= 50:
        verdict = ("HOLD", "#f0b429")
    else:
        verdict = ("SELL / AVOID", "#ff4f6a")
    return {"overall": overall, "categories": cat_scores, "verdict": verdict}

# ═══════════════════════════════════════════════════════════════
# FINANCIAL STATEMENTS
# ═══════════════════════════════════════════════════════════════

def get_ticker_attr(ticker, attr_names):
    for name in attr_names:
        try:
            value = getattr(ticker, name)
            if isinstance(value, pd.DataFrame) and not value.empty:
                df = value.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financial_statements(symbol: str):
    try:
        tk = yf.Ticker(symbol)
        income_annual = get_ticker_attr(tk, ["income_stmt", "financials"])
        balance_annual = get_ticker_attr(tk, ["balance_sheet", "balancesheet"])
        cashflow_annual = get_ticker_attr(tk, ["cashflow", "cash_flow"])
        income_quarterly = get_ticker_attr(tk, ["quarterly_income_stmt", "quarterly_financials"])
        balance_quarterly = get_ticker_attr(tk, ["quarterly_balance_sheet", "quarterly_balancesheet"])
        cashflow_quarterly = get_ticker_attr(tk, ["quarterly_cashflow", "quarterly_cash_flow"])
        return {
            "annual": {"income": income_annual, "balance": balance_annual, "cashflow": cashflow_annual},
            "quarterly": {"income": income_quarterly, "balance": balance_quarterly, "cashflow": cashflow_quarterly},
        }
    except Exception:
        return {
            "annual": {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()},
            "quarterly": {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()},
        }

def normalize_statement(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        cols = sorted(list(out.columns))
        out = out[cols]
    except Exception:
        pass
    return out

def find_first_existing(df, candidates):
    if df.empty:
        return pd.Series(dtype=float)
    for cand in candidates:
        if cand in df.index:
            return df.loc[cand]
    return pd.Series(dtype=float)

def build_statement_table(df, metric_map):
    df = normalize_statement(df)
    if df.empty:
        return pd.DataFrame()
    rows = {}
    for label, aliases in metric_map.items():
        ser = find_first_existing(df, aliases)
        if not ser.empty:
            rows[label] = ser
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).T
    # Normalize column names to date strings
    new_cols = []
    for c in table.columns:
        try:
            new_cols.append(pd.to_datetime(c).strftime("%Y-%m-%d"))
        except Exception:
            new_cols.append(str(c))
    table.columns = new_cols
    table = table.iloc[:, ::-1]
    return table

def format_statement_table(table):
    if table.empty:
        return table
    return table.map(fmt_financial_value)

def trend_chart_from_table(table, title):
    if table.empty or table.shape[1] < 2:
        return None
    fig = go.Figure()
    colors = ["#3b82f6", "#00d09c", "#f0b429", "#ff4f6a", "#a78bfa"]
    for i, idx in enumerate(table.index):
        vals = pd.to_numeric(table.loc[idx], errors="coerce")
        if vals.notna().sum() >= 2:
            fig.add_trace(go.Scatter(
                x=list(table.columns), y=list(vals.values),
                mode="lines+markers", name=idx,
                line=dict(color=colors[i % len(colors)], width=1.8),
                marker=dict(size=5)
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=11, color="#6b7a8f")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", family="IBM Plex Mono"),
        height=300, margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(gridcolor="#111d2f", linecolor="#1a2335"),
        yaxis=dict(gridcolor="#111d2f", linecolor="#1a2335"),
        legend=dict(orientation="h", font=dict(size=10))
    )
    return fig

INCOME_METRICS = {
    "Revenue": ["Total Revenue", "Operating Revenue"],
    "Gross Profit": ["Gross Profit"],
    "Operating Income": ["Operating Income", "EBIT"],
    "EBITDA": ["EBITDA", "Normalized EBITDA"],
    "Net Income": ["Net Income", "Net Income Common Stockholders"],
    "Diluted EPS": ["Diluted EPS", "Basic EPS"],
}
BALANCE_METRICS = {
    "Total Assets": ["Total Assets"],
    "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "Stockholders Equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "Cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Equivalents"],
    "Total Debt": ["Total Debt", "Current Debt And Capital Lease Obligation"],
    "Current Assets": ["Current Assets"],
    "Current Liabilities": ["Current Liabilities"],
}
CASHFLOW_METRICS = {
    "Operating Cash Flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "Capital Expenditure": ["Capital Expenditure"],
    "Free Cash Flow": ["Free Cash Flow"],
    "Investing Cash Flow": ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"],
    "Financing Cash Flow": ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"],
}

# ═══════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, period: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        info = ticker.info
        # Flatten MultiIndex columns if present (yfinance >=0.2.x)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        # Drop timezone from index for plotly compatibility
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist, info
    except Exception:
        return pd.DataFrame(), {}

# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

_AX = dict(gridcolor="#111d2f", zerolinecolor="#1a2335", linecolor="#1a2335")
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#8892a4", size=10),
    xaxis=_AX, yaxis=_AX
)

def price_chart(hist, tech, symbol):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.14, 0.18, 0.16], vertical_spacing=0.018,
        subplot_titles=["", "", "RSI (14)", "MACD"]
    )
    close = tech["_close"]
    sma20 = tech["_sma20"]; sma50 = tech["_sma50"]; sma200 = tech["_sma200"]
    bb_up = tech["_bb_up"]; bb_lo = tech["_bb_lo"]
    volume = tech["_volume"]; vol_ma = tech["_vol_ma"]
    rsi = tech["_rsi"]; macd = tech["_macd"]; sig = tech["_macd_sig"]; mhist = tech["_macd_hist"]

    has_ohlc = all(c in hist.columns for c in ["Open", "High", "Low", "Close"])
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name="Price",
            increasing_fillcolor="#00d09c", decreasing_fillcolor="#ff4f6a",
            increasing_line_color="#00d09c", decreasing_line_color="#ff4f6a"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=close.index, y=close, name="Price", line=dict(width=1.5, color="#3b82f6")), row=1, col=1)

    fig.add_trace(go.Scatter(x=bb_up.index, y=bb_up, name="BB+", line=dict(width=1, dash="dot", color="#334155"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_lo.index, y=bb_lo, name="BB-", line=dict(width=1, dash="dot", color="#334155"), fill="tonexty", fillcolor='rgba(51,65,85,0.06)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma20.index, y=sma20, name="SMA20", line=dict(width=1.2, color="#f0b429")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50, name="SMA50", line=dict(width=1.2, color="#3b82f6")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma200.index, y=sma200, name="SMA200", line=dict(width=1.2, color="#a78bfa")), row=1, col=1)

    open_col = hist["Open"] if has_ohlc else hist["Close"].shift().fillna(hist["Close"])
    colors_v = ["#00d09c" if c >= o else "#ff4f6a" for c, o in zip(hist["Close"], open_col)]
    fig.add_trace(go.Bar(x=volume.index, y=volume, marker_color=colors_v, opacity=0.4, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=vol_ma.index, y=vol_ma, line=dict(color="#475569", width=1), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(width=1.4, color="#3b82f6"), showlegend=False), row=3, col=1)
    for lvl, col in [(70, "#ff4f6a"), (50, "#334155"), (30, "#00d09c")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=col, line_width=0.8, row=3, col=1)

    colors_m = ["#00d09c" if v >= 0 else "#ff4f6a" for v in mhist.fillna(0)]
    fig.add_trace(go.Bar(x=mhist.index, y=mhist, marker_color=colors_m, opacity=0.6, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(width=1.2, color="#3b82f6"), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig, name="Signal", line=dict(width=1.2, color="#f0b429"), showlegend=False), row=4, col=1)

    fig.update_layout(
        title=dict(text=f"{symbol}  ·  Technical Analysis", font=dict(family="IBM Plex Mono", size=12, color="#6b7a8f")),
        height=700,
        **CHART_THEME,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.02, font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def radar_chart(cat_scores, symbol):
    cats = list(cat_scores.keys())
    vals = [cat_scores[c] if _is_valid(cat_scores[c]) else 0 for c in cats]
    vals += [vals[0]]; cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(59,130,246,0.10)",
        line=dict(width=1.8, color="#3b82f6"),
        marker=dict(size=4, color="#3b82f6"),
        name=symbol
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(tickfont_size=10, tickfont_color="#8892a4", gridcolor="#1a2335", linecolor="#1a2335"),
            radialaxis=dict(visible=True, range=[0, 100], tickfont_size=8, tickfont_color="#3b5070", gridcolor="#111d2f", linecolor="#1a2335")
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        height=300, showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def return_comparison_chart(all_returns):
    syms = list(all_returns.keys())
    vals = list(all_returns.values())
    colors = ["#00d09c" if v >= 0 else "#ff4f6a" for v in vals]
    fig = go.Figure(go.Bar(
        x=syms, y=vals, marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#8892a4")
    ))
    fig.update_layout(
        title=dict(text=t("return_comparison"), font=dict(family="IBM Plex Mono", size=11, color="#6b7a8f")),
        height=320, **CHART_THEME,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# ANALYSIS TEXT
# ═══════════════════════════════════════════════════════════════

def metric_sentence(label, value):
    if np.isnan(value):
        return None
    lang = _lang()

    sentences = {
        "P/E Ratio": {
            "EN": (
                f"Trailing P/E is **{value:.1f}x** — " +
                ("a relatively modest valuation vs. earnings." if value < 15 else
                 "a broadly reasonable range for a mature quality business." if value <= 25 else
                 "the market is pricing the stock richly and expects strong growth.")
            ),
            "EL": (
                f"Το trailing P/E είναι **{value:.1f}x** — " +
                ("σχετικά χαμηλή αποτίμηση σε σχέση με τα κέρδη." if value < 15 else
                 "λογικό εύρος για μια ώριμη ποιοτική εταιρεία." if value <= 25 else
                 "η αγορά αποτιμά ακριβά και προεξοφλεί ισχυρή ανάπτυξη.")
            ),
            "DE": (
                f"Das KGV beträgt **{value:.1f}x** — " +
                ("eine relativ bescheidene Bewertung." if value < 15 else
                 "ein vernünftiger Bereich für ein reifes Qualitätsunternehmen." if value <= 25 else
                 "der Markt bewertet die Aktie teuer und erwartet starkes Wachstum.")
            ),
        },
        "ROE %": {
            "EN": f"ROE is **{value:.1f}%** — " + ("excellent capital efficiency." if value > 20 else "satisfactory but not outstanding." if value > 10 else "below average, warranting scrutiny."),
            "EL": f"Το ROE είναι **{value:.1f}%** — " + ("εξαιρετική αποδοτικότητα κεφαλαίων." if value > 20 else "ικανοποιητικό αλλά όχι εξαιρετικό." if value > 10 else "κάτω από μέσο επίπεδο."),
            "DE": f"ROE beträgt **{value:.1f}%** — " + ("hervorragende Kapitaleffizienz." if value > 20 else "befriedigend aber nicht herausragend." if value > 10 else "unterdurchschnittlich."),
        },
        "Net Margin %": {
            "EN": f"Net margin is **{value:.1f}%** — " + ("strong bottom-line profitability." if value > 15 else "healthy level." if value > 5 else "thin but positive." if value >= 0 else "company is loss-making."),
            "EL": f"Το net margin είναι **{value:.1f}%** — " + ("ισχυρή τελική κερδοφορία." if value > 15 else "υγιές επίπεδο." if value > 5 else "λεπτό αλλά θετικό." if value >= 0 else "η εταιρεία είναι ζημιογόνα."),
            "DE": f"Nettomarge beträgt **{value:.1f}%** — " + ("starke Profitabilität." if value > 15 else "gesundes Niveau." if value > 5 else "dünn aber positiv." if value >= 0 else "Unternehmen schreibt Verluste."),
        },
        "Revenue Growth %": {
            "EN": f"Revenue growth is **{value:.1f}%** — " + ("very strong commercial momentum." if value > 20 else "healthy and positive." if value > 8 else "mild but positive." if value >= 0 else "top line is contracting — red flag."),
            "EL": f"Η ανάπτυξη εσόδων είναι **{value:.1f}%** — " + ("πολύ ισχυρή εμπορική δυναμική." if value > 20 else "θετική και υγιής." if value > 8 else "ήπια αλλά θετική." if value >= 0 else "συρρίκνωση εσόδων — red flag."),
            "DE": f"Umsatzwachstum beträgt **{value:.1f}%** — " + ("sehr starke Dynamik." if value > 20 else "gesund und positiv." if value > 8 else "mild aber positiv." if value >= 0 else "Umsatzrückgang — Warnsignal."),
        },
        "EPS Growth %": {
            "EN": f"EPS growth is **{value:.1f}%** — " + ("strong profit expansion." if value > 15 else "positive but modest." if value >= 0 else "per-share earnings are deteriorating."),
            "EL": f"Το EPS growth είναι **{value:.1f}%** — " + ("ισχυρή επέκταση κερδών." if value > 15 else "θετικό αλλά μέτριο." if value >= 0 else "τα κέρδη ανά μετοχή επιδεινώνονται."),
            "DE": f"EPS-Wachstum beträgt **{value:.1f}%** — " + ("starke Gewinnexpansion." if value > 15 else "positiv aber bescheiden." if value >= 0 else "Gewinn je Aktie verschlechtert sich."),
        },
        "Debt/Equity": {
            "EN": f"Debt/Equity is **{value:.2f}x** — " + ("conservative leverage." if value < 0.5 else "moderate but manageable." if value < 1.5 else "elevated financial risk."),
            "EL": f"Το Debt/Equity είναι **{value:.2f}x** — " + ("συντηρητικός δανεισμός." if value < 0.5 else "μέτριο αλλά διαχειρίσιμο." if value < 1.5 else "αυξημένος χρηματοοικονομικός κίνδυνος."),
            "DE": f"Verschuldungsgrad beträgt **{value:.2f}x** — " + ("konservative Verschuldung." if value < 0.5 else "moderat aber handhabbar." if value < 1.5 else "erhöhtes Finanzrisiko."),
        },
        "Beta": {
            "EN": f"Beta is **{value:.2f}** — " + ("defensive profile vs. market." if value < 0.8 else "volatility close to market." if value <= 1.3 else "aggressive and volatile profile."),
            "EL": f"Η beta είναι **{value:.2f}** — " + ("αμυντικό προφίλ έναντι αγοράς." if value < 0.8 else "μεταβλητότητα κοντά στην αγορά." if value <= 1.3 else "επιθετικό και volatile προφίλ."),
            "DE": f"Beta beträgt **{value:.2f}** — " + ("defensives Profil vs. Markt." if value < 0.8 else "Volatilität nahe am Markt." if value <= 1.3 else "aggressives und volatiles Profil."),
        },
        "RSI": {
            "EN": f"RSI is **{value:.0f}** — " + ("oversold territory / possible bounce." if value < 30 else "overbought territory / possible pullback." if value > 70 else "neutral momentum zone."),
            "EL": f"Το RSI είναι **{value:.0f}** — " + ("υπερπουλημένη ζώνη / πιθανή ανάκαμψη." if value < 30 else "υπεραγορασμένη ζώνη / πιθανή διόρθωση." if value > 70 else "ουδέτερη ζώνη ορμής."),
            "DE": f"RSI beträgt **{value:.0f}** — " + ("überverkauft / mögliche Erholung." if value < 30 else "überkauft / möglicher Rückgang." if value > 70 else "neutrale Momentumzone."),
        },
    }
    entry = sentences.get(label, {})
    return entry.get(lang, entry.get("EN"))

def generate_investment_summary(sym, fund, tech, score, roi, period_lbl):
    verdict = score["verdict"][0]
    overall = score["overall"]
    name = fund.get("_name", sym)
    cat = score["categories"]
    lang = _lang()

    if lang == "EL":
        lines = [
            f"## Αναλυτική Αξιολόγηση — {name} ({sym})",
            f"Το asset εμφανίζει **συνολικό score {overall:.0f}/100** και η ένδειξη μοντέλου είναι **{verdict}**. "
            f"Στο διάστημα **{period_lbl}**, η απόδοση ήταν **{roi:+.1f}%**.",
            f"### Κατηγορίες Score\n"
            f"- **Αποτίμηση:** {cat.get('Valuation', np.nan):.0f}/100\n"
            f"- **Ποιότητα:** {cat.get('Quality', np.nan):.0f}/100\n"
            f"- **Ανάπτυξη:** {cat.get('Growth', np.nan):.0f}/100\n"
            f"- **Κίνδυνος:** {cat.get('Risk', np.nan):.0f}/100\n"
            f"- **Τεχνικά:** {cat.get('Technical', np.nan):.0f}/100",
        ]
    elif lang == "DE":
        lines = [
            f"## Detaillierte Bewertung — {name} ({sym})",
            f"Das Asset zeigt einen **Gesamtscore von {overall:.0f}/100** und das Modellurteil lautet **{verdict}**. "
            f"Im Zeitraum **{period_lbl}** betrug die Rendite **{roi:+.1f}%**.",
            f"### Kategorie-Scores\n"
            f"- **Bewertung:** {cat.get('Valuation', np.nan):.0f}/100\n"
            f"- **Qualität:** {cat.get('Quality', np.nan):.0f}/100\n"
            f"- **Wachstum:** {cat.get('Growth', np.nan):.0f}/100\n"
            f"- **Risiko:** {cat.get('Risk', np.nan):.0f}/100\n"
            f"- **Technisch:** {cat.get('Technical', np.nan):.0f}/100",
        ]
    else:
        lines = [
            f"## Detailed Assessment — {name} ({sym})",
            f"The asset shows an **overall score of {overall:.0f}/100** with model verdict **{verdict}**. "
            f"Over the **{period_lbl}** period, total return was **{roi:+.1f}%**.",
            f"### Category Scores\n"
            f"- **Valuation:** {cat.get('Valuation', np.nan):.0f}/100\n"
            f"- **Quality:** {cat.get('Quality', np.nan):.0f}/100\n"
            f"- **Growth:** {cat.get('Growth', np.nan):.0f}/100\n"
            f"- **Risk:** {cat.get('Risk', np.nan):.0f}/100\n"
            f"- **Technical:** {cat.get('Technical', np.nan):.0f}/100",
        ]

    key_metrics = ["P/E Ratio", "ROE %", "Net Margin %", "Revenue Growth %", "EPS Growth %", "Debt/Equity", "Beta"]
    points = []
    for m in key_metrics:
        s = metric_sentence(m, fund.get(m, np.nan))
        if s:
            points.append(f"- {s}")
    rsi_s = metric_sentence("RSI", tech.get("RSI", np.nan))
    if rsi_s:
        points.append(f"- {rsi_s}")

    hdr = {"EN": "### Key Metric Interpretations", "EL": "### Ερμηνεία Βασικών Δεικτών", "DE": "### Interpretation der Kennzahlen"}
    if points:
        lines.append(hdr.get(lang, hdr["EN"]) + "\n" + "\n".join(points))

    # Technical reading
    tech_hdr = {"EN": "### Technical Reading", "EL": "### Τεχνική Ανάγνωση", "DE": "### Technische Analyse"}
    if lang == "EL":
        tl = [
            "MACD σε bullish διάταξη — υποστηρίζει βραχυπρόθεσμη ορμή." if tech.get("MACD_Bullish") else "MACD όχι bullish — βραχυπρόθεσμη ορμή αδύναμη.",
            "Golden Cross παρόν — θετική μακροπρόθεσμη τεχνική εικόνα." if tech.get("Golden_Cross") else "Χωρίς Golden Cross — μακροπρόθεσμη τεχνική εικόνα πιο ουδέτερη.",
            "Τιμή πάνω από SMA200 — ανοδική μακροπρόθεσμη δομή." if tech.get("Price_vs_SMA200") else "Τιμή κάτω από SMA200 — μακροπρόθεσμη τάση δεν είναι ισχυρή.",
            "OBV ανοδικό — ο όγκος επιβεβαιώνει την κίνηση." if tech.get("OBV_Trend") == "Rising" else "OBV μη ανοδικό — αδύναμη επιβεβαίωση όγκου.",
        ]
    elif lang == "DE":
        tl = [
            "MACD in bullischer Konfiguration — unterstützt kurzfristiges Momentum." if tech.get("MACD_Bullish") else "MACD nicht bullisch — kurzfristiges Momentum schwach.",
            "Golden Cross vorhanden — positives langfristiges technisches Bild." if tech.get("Golden_Cross") else "Kein Golden Cross — langfristiges technisches Bild neutral.",
            "Kurs über SMA200 — aufwärts gerichtete Langfriststruktur." if tech.get("Price_vs_SMA200") else "Kurs unter SMA200 — Langfristtrend nicht stark.",
            "OBV steigend — Volumen bestätigt die Bewegung." if tech.get("OBV_Trend") == "Rising" else "OBV nicht steigend — schwache Volumenbestätigung.",
        ]
    else:
        tl = [
            "MACD in bullish configuration — short-term momentum is supportive." if tech.get("MACD_Bullish") else "MACD is not bullish — short-term momentum is weaker.",
            "Golden Cross is present — constructive longer-term technical backdrop." if tech.get("Golden_Cross") else "No Golden Cross — longer-term technical setup is neutral or weaker.",
            "Price above SMA200 — upward long-term structure." if tech.get("Price_vs_SMA200") else "Price below SMA200 — long-term trend is not especially strong.",
            "OBV is rising — volume confirms the directional move." if tech.get("OBV_Trend") == "Rising" else "OBV not rising — volume confirmation is weaker.",
        ]
    lines.append(tech_hdr.get(lang, tech_hdr["EN"]) + "\n" + "\n".join([f"- {x}" for x in tl]))

    # Analyst view
    ar = fund.get("_analyst_rating", "N/A")
    ac = fund.get("_analyst_count", np.nan)
    up = fund.get("_upside", np.nan)
    if ar != "N/A":
        if lang == "EL":
            txt = f"Consensus αναλυτών: **{ar}**"
            if _is_valid(ac): txt += f" ({int(ac)} αναλυτές)"
            if _is_valid(up): txt += f" — εκτιμώμενο upside **{up:+.1f}%**."
            lines.append("### Γνώμη Αναλυτών\n" + txt)
        elif lang == "DE":
            txt = f"Analystenkonsens: **{ar}**"
            if _is_valid(ac): txt += f" ({int(ac)} Analysten)"
            if _is_valid(up): txt += f" — impliziertes Upside **{up:+.1f}%**."
            lines.append("### Analystenmeinung\n" + txt)
        else:
            txt = f"Analyst consensus: **{ar}**"
            if _is_valid(ac): txt += f" ({int(ac)} analysts)"
            if _is_valid(up): txt += f" — implied upside **{up:+.1f}%**."
            lines.append("### Analyst View\n" + txt)

    # Final conclusion
    pos = sum(1 for v in cat.values() if _is_valid(v) and v >= 60)
    neg = sum(1 for v in cat.values() if _is_valid(v) and v < 40)

    if lang == "EL":
        if verdict == "BUY":
            conc = "Τα διαθέσιμα δεδομένα δείχνουν επαρκή στοιχεία που δικαιολογούν **αγορά**. Η ποιότητα, η αποτίμηση ή/και η τεχνική εικόνα προσφέρουν ελκυστικό προφίλ risk/reward."
        elif verdict == "HOLD":
            conc = "Η συνολική εικόνα οδηγεί σε **hold**. Υπάρχουν θετικά, αλλά όχι αρκετά για σαφή νέα αγορά — ή συνυπάρχουν αδυναμίες που περιορίζουν την ελκυστικότητα."
        else:
            conc = "Η ένδειξη είναι **sell / avoid**. Αυτή τη στιγμή η σχέση αποτίμησης, κινδύνου, ανάπτυξης και τεχνικής εικόνας δεν είναι αρκετά ελκυστική."
        lines.append("### Τελικό Συμπέρασμα\n" + conc)
        bal = f"**{pos} θετικοί** και **{neg} αρνητικοί** παράγοντες από τις 5 κατηγορίες."
        lines.append(bal)
    elif lang == "DE":
        if verdict == "BUY":
            conc = "Die verfügbaren Daten zeigen genug unterstützende Merkmale für eine **Kauf**-Empfehlung."
        elif verdict == "HOLD":
            conc = "Das Gesamtbild führt zu einem **Halten**. Es gibt konstruktive Merkmale, aber nicht genug für einen klaren Neukauf."
        else:
            conc = "Das Signal führt zu **Verkaufen / Meiden**. Die aktuelle Kombination aus Bewertung, Risiko und Technik ist nicht attraktiv genug."
        lines.append("### Fazit\n" + conc)
        bal = f"**{pos} positive** und **{neg} negative** Faktoren aus den 5 Kategorien."
        lines.append(bal)
    else:
        if verdict == "BUY":
            conc = "The data shows enough supportive characteristics to justify a **buy** view. Quality, valuation and/or technical structure create an attractive risk/reward profile."
        elif verdict == "HOLD":
            conc = "The overall picture leads to a **hold** view — constructive features exist but not enough for a fresh buy, or offsetting weaknesses limit overall attractiveness."
        else:
            conc = "The signal leads to **sell / avoid**. The current mix of valuation, risk, growth and technical positioning is not attractive enough for a new entry."
        lines.append("### Final Conclusion\n" + conc)
        bal = f"**{pos} positive** and **{neg} negative** factors across 5 scoring categories."
        lines.append(bal)

    return "\n\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# FETCH + ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def run_analysis_for_symbols(symbols_tuple, period):
    symbols = list(symbols_tuple)
    all_results = {}
    for sym in symbols:
        hist, info = fetch_data(sym, period)
        if hist.empty or len(hist) < 5:
            continue
        tech = technical_signals(hist)
        fund = extract_fundamentals(info)
        score = composite_score(fund, tech)
        roi = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
        statements = fetch_financial_statements(sym)
        all_results[sym] = {
            "hist": hist, "tech": tech, "fund": fund,
            "score": score, "roi": roi, "statements": statements,
        }
    return all_results

def quick_pick_analyze(symbol: str):
    st.session_state.search_val = symbol
    st.session_state.live_search = symbol
    st.session_state.analysis_symbols = [symbol]
    st.session_state.analysis_results = run_analysis_for_symbols((symbol,), st.session_state.period_code)
    st.session_state.analysis_has_run = True
    st.session_state.last_period_used = st.session_state.period_code

# ═══════════════════════════════════════════════════════════════
# HERO ADVANCED TRADING BACKGROUND
# ═══════════════════════════════════════════════════════════════

hero_bg_svg = """
<svg width="100%" height="100%" viewBox="0 0 1400 380" fill="none" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="cg1" x1="0" y1="0" x2="1400" y2="0" gradientUnits="userSpaceOnUse">
      <stop stop-color="#1d4ed8" stop-opacity="0.0"/>
      <stop offset="0.3" stop-color="#3b82f6" stop-opacity="0.55"/>
      <stop offset="0.7" stop-color="#00d09c" stop-opacity="0.35"/>
      <stop offset="1" stop-color="#1d4ed8" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="cg2" x1="0" y1="0" x2="1400" y2="0" gradientUnits="userSpaceOnUse">
      <stop stop-color="#a78bfa" stop-opacity="0.0"/>
      <stop offset="0.4" stop-color="#818cf8" stop-opacity="0.30"/>
      <stop offset="1" stop-color="#3b82f6" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="gridFade" x1="0" y1="0" x2="0" y2="380" gradientUnits="userSpaceOnUse">
      <stop stop-color="#1a2335" stop-opacity="0.0"/>
      <stop offset="0.5" stop-color="#1a2335" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#1a2335" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <!-- Grid -->
  <g opacity="0.18">
    <line x1="0" y1="60" x2="1400" y2="60" stroke="#1e293b"/>
    <line x1="0" y1="120" x2="1400" y2="120" stroke="#1e293b"/>
    <line x1="0" y1="180" x2="1400" y2="180" stroke="#1e293b"/>
    <line x1="0" y1="240" x2="1400" y2="240" stroke="#1e293b"/>
    <line x1="0" y1="300" x2="1400" y2="300" stroke="#1e293b"/>
    <line x1="140" y1="0" x2="140" y2="380" stroke="#1e293b"/>
    <line x1="280" y1="0" x2="280" y2="380" stroke="#1e293b"/>
    <line x1="420" y1="0" x2="420" y2="380" stroke="#1e293b"/>
    <line x1="560" y1="0" x2="560" y2="380" stroke="#1e293b"/>
    <line x1="700" y1="0" x2="700" y2="380" stroke="#1e293b"/>
    <line x1="840" y1="0" x2="840" y2="380" stroke="#1e293b"/>
    <line x1="980" y1="0" x2="980" y2="380" stroke="#1e293b"/>
    <line x1="1120" y1="0" x2="1120" y2="380" stroke="#1e293b"/>
    <line x1="1260" y1="0" x2="1260" y2="380" stroke="#1e293b"/>
  </g>

  <!-- Candlestick bars (decorative) -->
  <g opacity="0.10">
    <rect x="60" y="180" width="10" height="80" rx="2" fill="#00d09c"/>
    <rect x="85" y="160" width="10" height="100" rx="2" fill="#00d09c"/>
    <rect x="110" y="200" width="10" height="60" rx="2" fill="#ff4f6a"/>
    <rect x="135" y="170" width="10" height="90" rx="2" fill="#00d09c"/>
    <rect x="160" y="140" width="10" height="120" rx="2" fill="#00d09c"/>
    <rect x="185" y="155" width="10" height="105" rx="2" fill="#ff4f6a"/>
    <rect x="210" y="130" width="10" height="130" rx="2" fill="#00d09c"/>
    <rect x="235" y="120" width="10" height="140" rx="2" fill="#00d09c"/>
    <rect x="1150" y="190" width="10" height="70" rx="2" fill="#00d09c"/>
    <rect x="1175" y="170" width="10" height="90" rx="2" fill="#ff4f6a"/>
    <rect x="1200" y="150" width="10" height="110" rx="2" fill="#00d09c"/>
    <rect x="1225" y="140" width="10" height="120" rx="2" fill="#00d09c"/>
    <rect x="1250" y="160" width="10" height="100" rx="2" fill="#ff4f6a"/>
    <rect x="1275" y="130" width="10" height="130" rx="2" fill="#00d09c"/>
    <rect x="1300" y="120" width="10" height="140" rx="2" fill="#00d09c"/>
  </g>

  <!-- Main chart line 1 (price line) -->
  <path d="M0 300 C100 285 160 240 250 220 C330 202 380 230 460 210 C540 190 590 155 680 140 C760 125 820 160 900 145 C980 130 1040 100 1120 88 C1200 76 1280 95 1400 80"
        stroke="url(#cg1)" stroke-width="2.5" fill="none" stroke-linecap="round"/>

  <!-- Second line (moving average) -->
  <path d="M0 310 C80 300 160 265 260 250 C360 235 440 245 540 232 C640 219 720 195 820 182 C920 169 1020 175 1120 162 C1220 149 1300 140 1400 132"
        stroke="url(#cg2)" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-dasharray="6 3"/>

  <!-- Glows / accents -->
  <circle cx="700" cy="140" r="120" fill="#3b82f6" fill-opacity="0.04"/>
  <circle cx="1120" cy="88" r="80" fill="#00d09c" fill-opacity="0.06"/>
  <circle cx="250" cy="220" r="60" fill="#818cf8" fill-opacity="0.05"/>
</svg>
"""

# ═══════════════════════════════════════════════════════════════
# UI — TICKER BAR
# ═══════════════════════════════════════════════════════════════

ticker_prices = fetch_ticker_prices(tuple(TICKER_SYMBOLS))
if ticker_prices:
    st.markdown(build_ticker_html(ticker_prices), unsafe_allow_html=True)

# Language selector (top right)
lc1, lc2 = st.columns([11, 1])
with lc2:
    lang_pick = st.selectbox("lang", ["🇬🇧 EN", "🇬🇷 EL", "🇩🇪 DE"], index=0, label_visibility="collapsed", key="lang_sel")
    st.session_state.LANG = lang_pick.split()[1]

# ═══════════════════════════════════════════════════════════════
# UI — HERO
# ═══════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-canvas">{hero_bg_svg}</div>
        <div class="hero-inner">
            <div class="hero-eyebrow">MARKET INTELLIGENCE PLATFORM</div>
            <div class="hero-title">Professional <span>Investments</span> Analyzer</div>
            <div class="hero-sub">{t("header_sub")}</div>
            <div class="hero-pills">
                <span class="hero-pill">{t("hero_equities")}</span>
                <span class="hero-pill">{t("hero_crypto")}</span>
                <span class="hero-pill">{t("hero_etfs")}</span>
                <span class="hero-pill">{t("hero_technical")}</span>
                <span class="hero-pill">{t("hero_financials")}</span>
                <span class="hero-pill">{t("hero_comparison")}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# UI — SEARCH SECTION
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="search-section">', unsafe_allow_html=True)
st.markdown('<div class="search-inner">', unsafe_allow_html=True)

# Handle query params
ac_clicked = st.query_params.get("ac", "")
qp_clicked = st.query_params.get("qp", "")

if ac_clicked:
    st.session_state.live_search = ac_clicked
    st.session_state.search_val = ac_clicked
    st.query_params.clear()
    symbols = [ac_clicked.strip().upper()]
    st.session_state.analysis_symbols = symbols
    st.session_state.analysis_results = run_analysis_for_symbols(tuple(symbols), st.session_state.period_code)
    st.session_state.analysis_has_run = True
    st.session_state.last_period_used = st.session_state.period_code
    st.rerun()

if qp_clicked:
    quick_pick_analyze(qp_clicked)
    st.query_params.clear()
    st.rerun()

labels, codes, period_map = get_period_maps()

search_col, period_col = st.columns([5, 1])
with search_col:
    live_search = st.text_input(
        "search",
        value=st.session_state.get("live_search", ""),
        placeholder=t("search_placeholder"),
        label_visibility="collapsed",
        key="live_search_input"
    )
    if live_search != st.session_state.live_search:
        st.session_state.live_search = live_search

with period_col:
    period_display = st.selectbox(
        t("period_label"),
        labels,
        index=codes.index(st.session_state.period_code) if st.session_state.period_code in codes else 3,
        label_visibility="collapsed",
        key="period_selector_top",
    )
    st.session_state.period_code = period_map[period_display]

# Dynamic autocomplete — searches as you type
query = st.session_state.live_search.strip()
if len(query) >= 1:
    suggestions = search_symbols(query)
    # Only show if not an exact full match already
    is_exact = any(s.upper() == query.upper() for s, _, _ in suggestions) and len(suggestions) == 1
    if suggestions and not is_exact:
        badge_cls = {"Stock": "ac-badge-stock", "Crypto": "ac-badge-crypto", "ETF": "ac-badge-etf"}
        ac_html = '<div class="ac-box">'
        for sym, name, typ in suggestions[:7]:
            bc = badge_cls.get(typ, "ac-badge-stock")
            ac_html += f'''
                <a class="ac-item" href="?ac={sym}" target="_self">
                    <div>
                        <div class="ac-sym">{sym}</div>
                        <div class="ac-name">{name}</div>
                    </div>
                    <span class="ac-badge {bc}">{typ.upper()}</span>
                </a>
            '''
        ac_html += '</div>'
        st.markdown(ac_html, unsafe_allow_html=True)

analyze_clicked = st.button(t("btn_analyze"), type="primary", use_container_width=True)

# Quick picks
quick_picks = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "GLD"]
st.markdown(f'<div class="qp-label">{t("quick_picks")}</div>', unsafe_allow_html=True)
qp_html = '<div class="qp-wrap">'
for sym in quick_picks:
    qp_html += f'<a class="qp-pill" href="?qp={sym}" target="_self">{sym}</a>'
qp_html += '</div>'
st.markdown(qp_html, unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)  # close search-inner + search-section

# ═══════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════

if analyze_clicked:
    raw = st.session_state.live_search.strip()
    # Support comma or space separated
    symbols = [s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()]
    if not symbols:
        st.error(t("no_symbol_error"))
    else:
        prog = st.progress(0, text=t("fetching"))
        for i, sym in enumerate(symbols):
            prog.progress((i + 0.5) / len(symbols), text=t("analyzing").format(sym))
        prog.progress(1.0, text=t("done").format(", ".join(symbols)))
        prog.empty()
        results = run_analysis_for_symbols(tuple(symbols), st.session_state.period_code)
        if not results:
            st.error(t("no_data_error"))
        else:
            st.session_state.analysis_results = results
            st.session_state.analysis_symbols = symbols
            st.session_state.analysis_has_run = True
            st.session_state.last_period_used = st.session_state.period_code
            st.session_state.search_val = raw

# Re-run if period changed
if (
    st.session_state.analysis_has_run
    and st.session_state.analysis_symbols
    and st.session_state.last_period_used != st.session_state.period_code
):
    results = run_analysis_for_symbols(tuple(st.session_state.analysis_symbols), st.session_state.period_code)
    st.session_state.analysis_results = results
    st.session_state.last_period_used = st.session_state.period_code

all_results = st.session_state.analysis_results

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════

def section_hdr(label):
    st.markdown(
        f'<div class="section-hdr">'
        f'<div class="section-hdr-line"></div>'
        f'<div class="section-hdr-text">{label}</div>'
        f'<div class="section-hdr-line"></div>'
        f'</div>',
        unsafe_allow_html=True
    )

if st.session_state.analysis_has_run and all_results:

    if len(all_results) > 1:
        all_returns = {s: d["roi"] for s, d in all_results.items()}
        st.plotly_chart(return_comparison_chart(all_returns), use_container_width=True)

    for sym, data in all_results.items():
        hist = data["hist"]
        tech = data["tech"]
        fund = data["fund"]
        score = data["score"]
        roi = data["roi"]
        statements = data["statements"]

        cp = fund.get("_current_price", np.nan)
        if not _is_valid(cp):
            try:
                cp = float(hist["Close"].iloc[-1])
            except Exception:
                cp = np.nan

        name = fund["_name"]
        sector = fund["_sector"]
        industry = fund["_industry"]
        mcap = fund["_market_cap"]
        beta = fund.get("Beta", np.nan)
        risk_lbl, risk_col = risk_level(beta)
        h52 = tech.get("Price_52w_High", np.nan)
        l52 = tech.get("Price_52w_Low", np.nan)
        p_hi, p_lo, pos52 = pct_from_52w(cp, h52, l52)
        cp_str = f"${cp:,.2f}" if _is_valid(cp) else "N/A"

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── Asset header ──
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(
                f"""
                <div class="asset-header">
                    <div class="asset-sym">{sym}</div>
                    <div class="asset-name">{name}</div>
                    <div class="asset-meta">{sector}  ·  {industry}  ·  {fund['_country']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            vt, vc = score["verdict"]
            st.markdown(
                f"""
                <div class="verdict-card">
                    <div class="verdict-text" style="color:{vc}">{vt}</div>
                    <div class="verdict-score">{score['overall']:.0f} / 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Business description FIRST ──
        if fund["_summary"]:
            st.markdown(f'<div class="biz-desc">{fund["_summary"]}</div>', unsafe_allow_html=True)

        # ── Metrics row ──
        section_hdr("OVERVIEW")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("price")}</div><div class="metric-value">{cp_str}</div></div>', unsafe_allow_html=True)
        with m2:
            rc = "#00d09c" if roi >= 0 else "#ff4f6a"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("return")} {period_label_from_code(st.session_state.period_code)}</div><div class="metric-value" style="color:{rc}">{roi:+.1f}%</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("market_cap")}</div><div class="metric-value">{fmt_large(mcap)}</div></div>', unsafe_allow_html=True)
        with m4:
            rc2 = {"green": "#00d09c", "yellow": "#f0b429", "red": "#ff4f6a", "gray": "#6b7a8f"}.get(risk_col, "#6b7a8f")
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("risk_beta")}</div><div class="metric-value" style="color:{rc2}">{risk_lbl}</div><div class="metric-sub">β = {fmt_val(beta)}</div></div>', unsafe_allow_html=True)
        with m5:
            tp = fund.get("_target_price", np.nan)
            up = fund.get("_upside", np.nan)
            up_c = "#00d09c" if (_is_valid(up) and up > 0) else "#ff4f6a"
            tp_str = f"${tp:,.2f}" if _is_valid(tp) else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("analyst_target")}</div><div class="metric-value">{tp_str}</div><div class="metric-sub" style="color:{up_c}">{t("upside")} {fmt_pct(up)}</div></div>', unsafe_allow_html=True)
        with m6:
            an_count = fund["_analyst_count"]
            an_str = str(int(an_count)) if _is_valid(an_count) else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("analyst_consensus")}</div><div class="metric-value" style="font-size:.95rem">{fund["_analyst_rating"]}</div><div class="metric-sub">{an_str} {t("analysts")}</div></div>', unsafe_allow_html=True)

        if _is_valid(pos52):
            pos52c = min(max(pos52, 0), 100)
            lo_str = f"${l52:,.2f}" if _is_valid(l52) else "N/A"
            hi_str = f"${h52:,.2f}" if _is_valid(h52) else "N/A"
            st.markdown(
                f"""
                <div class="range-card">
                    <div class="range-label">52-WEEK RANGE · LOW {lo_str} → HIGH {hi_str}</div>
                    <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.70rem;color:#3b5070">
                        <span>From High: {p_hi:.1f}%</span>
                        <span>From Low: +{p_lo:.1f}%</span>
                    </div>
                    <div class="range-bar-bg">
                        <div class="range-bar-fill" style="width:{pos52c:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Technical chart ──
        section_hdr(t("technical_analysis").upper())
        st.plotly_chart(price_chart(hist, tech, sym), use_container_width=True)

        # ── Technical signals grid ──
        section_hdr(t("technical_signals").upper())
        rsi_val = tech.get("RSI", np.nan)
        macd_bull = tech.get("MACD_Bullish", False)
        golden = tech.get("Golden_Cross", False)
        abv_200 = tech.get("Price_vs_SMA200", False)
        obv_trend = tech.get("OBV_Trend", "N/A")
        stk_k = tech.get("Stochastic_K", np.nan)
        bb_pb = tech.get("BB_PctB", np.nan)
        vol_r = tech.get("Volume_vs_MA", np.nan)

        ts1, ts2, ts3, ts4 = st.columns(4)
        with ts1:
            rsi_ok = 40 <= rsi_val <= 60 if _is_valid(rsi_val) else False
            rsi_lbl = t("oversold") if rsi_val < 30 else (t("overbought") if rsi_val > 70 else t("neutral"))
            st.markdown(sig_card("RSI (14)", f"{fmt_val(rsi_val)} · {rsi_lbl}", rsi_ok), unsafe_allow_html=True)
        with ts2:
            st.markdown(sig_card("MACD", t("bullish") if macd_bull else t("bearish"), macd_bull), unsafe_allow_html=True)
        with ts3:
            st.markdown(sig_card("Golden Cross", t("yes") if golden else t("no"), golden), unsafe_allow_html=True)
        with ts4:
            st.markdown(sig_card("SMA200", t("above") if abv_200 else t("below"), abv_200), unsafe_allow_html=True)

        ts5, ts6, ts7, ts8 = st.columns(4)
        with ts5:
            st.markdown(sig_card("OBV Trend", t("rising") if obv_trend == "Rising" else t("falling"), obv_trend == "Rising"), unsafe_allow_html=True)
        with ts6:
            stk_ok = 20 <= stk_k <= 80 if _is_valid(stk_k) else False
            st.markdown(sig_card("Stochastic %K", fmt_val(stk_k), stk_ok), unsafe_allow_html=True)
        with ts7:
            bb_ok = 0.2 <= bb_pb <= 0.8 if _is_valid(bb_pb) else False
            st.markdown(sig_card("Bollinger %B", fmt_val(bb_pb), bb_ok), unsafe_allow_html=True)
        with ts8:
            vol_ok = _is_valid(vol_r) and vol_r > 1
            st.markdown(sig_card("Volume vs MA", f"{fmt_val(vol_r)}x", vol_ok), unsafe_allow_html=True)

        # ── Fundamental indicators ──
        section_hdr(t("fundamentals_title").upper())
        rows = []
        for ind, meta in INDICATOR_META.items():
            raw = fund.get(ind, np.nan)
            if ind == "RSI":
                raw = tech.get(ind, np.nan)
            dv = fmt_val(raw) if _is_valid(raw) else "N/A"
            sl = score_indicator(ind, raw)[1] if _is_valid(raw) else "N/A"
            rows.append({
                t("indicator_col"): ind,
                t("value_col"): dv,
                t("range_col"): meta["optimal"],
                t("signal_col"): sl,
                t("category_col"): meta["category"][_lang()],
                t("what_col"): meta["desc"][_lang()],
                t("how_col"): meta["assessment"][_lang()],
            })
        df_f = pd.DataFrame(rows)
        st.dataframe(
            df_f.style.map(color_signal_df, subset=[t("signal_col")]),
            use_container_width=True,
            hide_index=True,
            height=640
        )

        # ── Financial statements ──
        section_hdr("FINANCIAL STATEMENTS")
        annual = statements["annual"]
        quarterly = statements["quarterly"]
        annual_income = build_statement_table(annual["income"], INCOME_METRICS)
        annual_balance = build_statement_table(annual["balance"], BALANCE_METRICS)
        annual_cash = build_statement_table(annual["cashflow"], CASHFLOW_METRICS)
        quarterly_income = build_statement_table(quarterly["income"], INCOME_METRICS)
        quarterly_balance = build_statement_table(quarterly["balance"], BALANCE_METRICS)
        quarterly_cash = build_statement_table(quarterly["cashflow"], CASHFLOW_METRICS)

        if all(df.empty for df in [annual_income, annual_balance, annual_cash, quarterly_income, quarterly_balance, quarterly_cash]):
            st.info(t("no_financials"))
        else:
            tab_a, tab_q = st.tabs([t("annual"), t("quarterly")])
            with tab_a:
                for lbl, tbl, chart_rows, chart_title in [
                    (t("income_statement"), annual_income, ["Revenue", "EBITDA", "Net Income"], "Annual Revenue / EBITDA / Net Income"),
                    (t("balance_sheet"), annual_balance, ["Total Assets", "Stockholders Equity", "Total Debt", "Cash"], "Annual Assets / Equity / Debt / Cash"),
                    (t("cash_flow"), annual_cash, ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"], "Annual Cash Flow"),
                ]:
                    st.markdown(f"**{lbl}**")
                    c1, c2 = st.columns([1.2, 1])
                    with c1:
                        if not tbl.empty:
                            st.dataframe(format_statement_table(tbl), use_container_width=True)
                        else:
                            st.info(f"No {lbl} data available.")
                    with c2:
                        sub = tbl.loc[[i for i in tbl.index if i in chart_rows]] if not tbl.empty else pd.DataFrame()
                        chart = trend_chart_from_table(sub, chart_title)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

            with tab_q:
                for lbl, tbl, chart_rows, chart_title in [
                    (t("income_statement"), quarterly_income, ["Revenue", "EBITDA", "Net Income"], "Quarterly Revenue / EBITDA / Net Income"),
                    (t("balance_sheet"), quarterly_balance, ["Total Assets", "Stockholders Equity", "Total Debt", "Cash"], "Quarterly Assets / Equity / Debt / Cash"),
                    (t("cash_flow"), quarterly_cash, ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"], "Quarterly Cash Flow"),
                ]:
                    st.markdown(f"**{lbl}**")
                    c1, c2 = st.columns([1.2, 1])
                    with c1:
                        if not tbl.empty:
                            st.dataframe(format_statement_table(tbl), use_container_width=True)
                        else:
                            st.info(f"No {lbl} data available.")
                    with c2:
                        sub = tbl.loc[[i for i in tbl.index if i in chart_rows]] if not tbl.empty else pd.DataFrame()
                        chart = trend_chart_from_table(sub, chart_title)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

        # ── Score radar + category breakdown ──
        section_hdr("SCORE ANALYSIS")
        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(radar_chart(score["categories"], sym), use_container_width=True)
        with r2:
            for cat_name, val in score["categories"].items():
                if not _is_valid(val):
                    continue
                cc = "#00d09c" if val >= 65 else ("#f0b429" if val >= 45 else "#ff4f6a")
                st.markdown(
                    f"""
                    <div style="margin-bottom:14px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="font-size:.78rem;color:#8892a4;font-family:'IBM Plex Mono',monospace">{cat_name}</span>
                            <span style="font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:{cc};font-weight:600">{val:.0f}/100</span>
                        </div>
                        <div style="background:#111d2f;border-radius:2px;height:4px">
                            <div style="background:{cc};border-radius:2px;height:4px;width:{min(val,100):.1f}%"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div class="metric-card" style="margin-top:12px;text-align:center">
                    <div class="metric-label">{t("overall_score")}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;color:{score['verdict'][1]};margin-top:6px">{score['overall']:.0f}/100</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:{score['verdict'][1]};margin-top:3px;letter-spacing:.06em">{score['verdict'][0]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Investment assessment ──
        section_hdr(t("investment_assessment").upper())
        summary = generate_investment_summary(sym, fund, tech, score, roi, period_label_from_code(st.session_state.period_code))
        vc2 = score["verdict"][1]
        bg_map = {"#00d09c": "rgba(0,208,156,0.04)", "#f0b429": "rgba(240,180,41,0.04)", "#ff4f6a": "rgba(255,79,106,0.04)"}
        bg_col = bg_map.get(vc2, "rgba(10,15,30,0.6)")
        st.markdown(
            f'<div class="summary-box" style="background:{bg_col};border:1px solid {vc2}22;border-left:3px solid {vc2};">',
            unsafe_allow_html=True,
        )
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── CSV Download ──
        export = {k: v for k, v in {**fund, **tech}.items() if not k.startswith("_") and isinstance(v, (int, float, np.floating))}
        csv_b = pd.DataFrame([export]).to_csv(index=False).encode()
        st.download_button(
            f"{t('download_csv')} — {sym}",
            csv_b,
            f"{sym}_analysis.csv",
            mime="text/csv",
            key=f"csv_{sym}",
        )

        st.markdown("<hr style='border-color:#1a2335;margin:32px 0'>", unsafe_allow_html=True)

# ── Footer ──
st.markdown(f'<div class="footer-bar">{t("footer")}</div>', unsafe_allow_html=True)
