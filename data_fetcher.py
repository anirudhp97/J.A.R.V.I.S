import yfinance as yf
from nsepython import nse_eq, nse_quote_ltp
import pandas as pd
import json
import os

try:
    from tradingview_ta import TA_Handler, Interval
except ImportError:
    TA_Handler = None
    Interval = None

def fetch_live_stock_price(ticker):
    """
    Fetches the true real-time Last Traded Price (LTP) 
    directly from the National Stock Exchange of India with smart asset naming.
    """
    clean_ticker = str(ticker).upper().strip()
    asset_labels = {
        "GOLDBEES": "Nippon India ETF Gold BeES",
        "SILVERBEES": "Nippon India ETF Silver BeES",
        "NIFTYBEES": "Nippon India ETF Nifty BeES"
    }
    
    try:
        raw_data = nse_eq(clean_ticker)
        company_name = None
        if isinstance(raw_data, dict):
            company_name = raw_data.get('info', {}).get('companyName')
        
        if not company_name:
            company_name = asset_labels.get(clean_ticker, clean_ticker)
            
        ltp = None
        if isinstance(raw_data, dict):
            ltp = raw_data.get('priceInfo', {}).get('lastPrice', None)
        
        if not ltp:
            ltp = nse_quote_ltp(clean_ticker)

        return {
            "status": "success",
            "ticker": clean_ticker,
            "company_name": company_name,
            "ltp": float(ltp) if ltp else None,
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"NSE Live Price Retrieval Fault: {str(e)}")
        try:
            yf_ticker = f"{clean_ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            todays_data = stock.history(period='1d')
            if not todays_data.empty:
                ltp = todays_data['Close'].iloc[-1]
                return {
                    "status": "success",
                    "ticker": clean_ticker,
                    "company_name": asset_labels.get(clean_ticker, clean_ticker),
                    "ltp": float(ltp),
                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as ex:
            print(f"YFinance Backup Feed Fault: {str(ex)}")
            
        return {"status": "error", "message": f"All telemetry paths severed: {str(e)}"}

def fetch_market_news(ticker):
    """
    Pulls structured, context-rich financial news streams for sentiment matrix injection.
    """
    clean_ticker = str(ticker).upper().strip()
    yf_ticker = f"{clean_ticker}.NS"
    try:
        stock = yf.Ticker(yf_ticker)
        news_list = stock.news
        if not news_list:
            return "No localized real-time headline streams discovered inside the matrix index."
            
        formatted_news = []
        for index, item in enumerate(news_list[:4]):
            title = item.get('title', 'Unknown Title')
            publisher = item.get('publisher', 'Unknown Publisher')
            formatted_news.append(f"[{index+1}] Headline: {title} | Source: {publisher}")
            
        return "\n".join(formatted_news)
    except Exception as e:
        return f"News harvesting array failure: {str(e)}"

def fetch_gold_trend_analysis(ticker):
    """
    Computes true technical parameters (moving averages, momentum vector) 
    using historical telemetry blocks.
    """
    clean_ticker = str(ticker).upper().strip()
    yf_ticker = f"{clean_ticker}.NS"
    try:
        stock = yf.Ticker(yf_ticker)
        df = stock.history(period="6mo")
        if df.empty or len(df) < 50:
            return {"status": "error", "message": "Insufficient historical data depth inside logs."}
            
        latest_close = df['Close'].iloc[-1]
        sma_days = 50
        sma_baseline = df['Close'].rolling(window=sma_days).mean().iloc[-1]
        
        if pd.isna(sma_baseline):
            sma_days = 20
            sma_baseline = df['Close'].rolling(window=sma_days).mean().iloc[-1]
            
        deviation_pct = ((latest_close - sma_baseline) / sma_baseline) * 100
        momentum = "BULLISH ACCELERATION" if deviation_pct > 0.5 else ("BEARISH DRIFT" if deviation_pct < -0.5 else "CONSOLIDATED EQUILIBRIUM")
        
        return {
            "status": "success",
            "ticker": clean_ticker,
            "latest_close": round(float(latest_close), 2),
            "sma_baseline": round(float(sma_baseline), 2),
            "sma_days": sma_days,
            "deviation_pct": round(float(deviation_pct), 2),
            "momentum": momentum
        }
    except Exception as e:
        return {"status": "error", "message": f"Trend processing array offline: {str(e)}"}

def fetch_tradingview_gauge(ticker, timeframe="1D"):
    """
    Connects to the TradingView high-frequency technical compilation server matrix.
    """
    if not TA_Handler:
        return "TradingView library missing. Gauge tracking layer offline."
        
    clean_ticker = str(ticker).upper().strip()
    intervals = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1D": Interval.INTERVAL_1_DAY,
        "1W": Interval.INTERVAL_1_WEEK
    }
    
    tv_interval = intervals.get(timeframe, Interval.INTERVAL_1_DAY)
    try:
        handler = TA_Handler(
            symbol=clean_ticker,
            exchange="NSE",
            screener="india",
            interval=tv_interval,
            timeout=5
        )
        analysis = handler.get_analysis()
        summary = analysis.summary
        return (
            f"TradingView Gauge Stream ({timeframe}): {summary.get('RECOMMENDATION')} | "
            f"[Buy: {summary.get('BUY')}, Sell: {summary.get('SELL')}, Neutral: {summary.get('NEUTRAL')}]"
        )
    except Exception as e:
        return f"TradingView structural telemetry unreadable on frequency {timeframe}: {str(e)}"

def generate_forecast_chart_data(ticker, trend_data, forecast_periods=5):
    """
    Generates a momentum-based drift projection.
    Returns a DataFrame where 'Date' is a regular string column for robust Streamlit rendering.
    """
    try:
        if not trend_data or trend_data.get("status") == "error":
            return None
            
        latest_close = trend_data.get("latest_close")
        deviation_pct = trend_data.get("deviation_pct", 0)
        momentum_str = str(trend_data.get("momentum", "")).upper()

        drift_percentage = (float(deviation_pct) / 100) / 10.0
        if "BULLISH" in momentum_str:
            drift_direction = 1.0
            if drift_percentage <= 0:
                drift_percentage = 0.001
        elif "BEARISH" in momentum_str:
            drift_direction = -1.0
            if drift_percentage >= 0:
                drift_percentage = -0.001
        else:
            drift_direction = 0.0
            drift_percentage = 0.0

        projection_series = []
        current_simulated_price = float(latest_close)
        future_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=forecast_periods, freq='B')

        for date in future_dates:
            step_change = current_simulated_price * drift_percentage * drift_direction
            current_simulated_price += step_change
            projection_series.append({
                "Date": date.strftime('%Y-%m-%d'),
                "Projected Target": round(current_simulated_price, 2)
            })

        projection_df = pd.DataFrame(projection_series)
        return projection_df

    except Exception as e:
        print(f"Error in forecast engine for {ticker}: {str(e)}")
        return None

# ===================================================
# SESSION BACKUP PERSISTENCE SUBSYSTEM
# ===================================================
SESSION_CACHE_FILE = ".jarvis_session_history.json"

def save_chat_session(messages):
    """
    Serializes and commits the session memory array to a hidden local file.
    """
    try:
        serializable_messages = []
        for msg in messages:
            clean_msg = {}
            for k, v in msg.items():
                # Avoid trying to serialize pandas DataFrames if stored dynamically
                if isinstance(v, pd.DataFrame):
                    continue
                clean_msg[k] = v
            serializable_messages.append(clean_msg)
            
        with open(SESSION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_messages, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[SYSTEM CRITICAL] Failed to archive timeline datastream: {str(e)}")

def load_chat_session():
    """
    Loads and re-establishes the archived timeline datastream from file.
    """
    if os.path.exists(SESSION_CACHE_FILE):
        try:
            with open(SESSION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[SYSTEM WARN] Archived datastream corrupted. Initializing fresh context: {str(e)}")
            return []
    return []