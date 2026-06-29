import yfinance as yf
from nsepython import nse_eq, nse_quote_ltp
import pandas as pd
from tradingview_ta import TA_Handler, Interval

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
            "company": company_name,
            "price": ltp
        }
    except Exception as e:
        try:
            yf_symbol = f"{clean_ticker}.NS"
            yf_ticker = yf.Ticker(yf_symbol)
            live_price = yf_ticker.fast_info.last_price
            
            return {
                "status": "success",
                "ticker": clean_ticker,
                "company": asset_labels.get(clean_ticker, clean_ticker),
                "price": round(live_price, 2)
            }
        except Exception as inner_error:
            return {"status": "error", "message": f"All connection endpoints failed. Primary: {str(e)}. Fallback: {str(inner_error)}"}

def fetch_market_news(ticker):
    """
    Scrapes real-time market news summaries using Yahoo Finance vectors.
    """
    try:
        clean_ticker = str(ticker).upper().strip()
        yf_symbol = f"{clean_ticker}.NS"
        yf_ticker = yf.Ticker(yf_symbol)
        news_list = yf_ticker.news
        
        if not news_list:
            return f"No synchronized mainstream headlines available for {clean_ticker} vector at this period."
            
        compiled_headlines = ""
        for index, item in enumerate(news_list[:3]):
            title = item.get('title')
            publisher = item.get('publisher')
            compiled_headlines += f"[{index+1}] \"{title}\" via {publisher}\n"
            
        return compiled_headlines
    except Exception as e:
        return f"News diagnostic link unresponsive: {str(e)}"

def fetch_gold_trend_analysis(ticker, period="1mo"):
    """
    Generates a trend momentum blueprint using rolling Simple Moving Averages.
    """
    try:
        clean_ticker = str(ticker).upper().strip()
        yf_symbol = f"{clean_ticker}.NS"
        yf_ticker = yf.Ticker(yf_symbol)
        history = yf_ticker.history(period=period)
        
        if history.empty or len(history) < 2:
            return {"status": "error", "message": f"Insufficient data available for {clean_ticker} period: {period}"}
            
        total_days = len(history)
        if total_days <= 5:
            sma_window = 2    
        elif total_days <= 10:
            sma_window = 5    
        else:
            sma_window = min(20, total_days // 2) 
            
        history['Dynamic_SMA'] = history['Close'].rolling(window=sma_window).mean()
        
        latest_close = float(history['Close'].iloc[-1])
        sma_value = float(history['Dynamic_SMA'].iloc[-1]) if not pd.isna(history['Dynamic_SMA'].iloc[-1]) else float(history['Close'].mean())
        percent_deviation = ((latest_close - sma_value) / sma_value) * 100
        
        if latest_close > sma_value:
            momentum = f"BULLISH (Trading above the trailing {sma_window}-day baseline)"
        else:
            momentum = f"BEARISH (Trading below the trailing {sma_window}-day baseline)"
            
        return {
            "status": "success",
            "latest_close": round(latest_close, 2),
            "sma_baseline": round(sma_value, 2),
            "sma_days": sma_window,
            "deviation_pct": round(percent_deviation, 2),
            "momentum": momentum,
            "lookback_period": period
        }
    except Exception as e:
        return {"status": "error", "message": f"Trend processing failure: {str(e)}"}

def fetch_tradingview_gauge(ticker, timeframe="1d"):
    """
    Fetches the precise consensus dashboard summary from TradingView's backend scanning arrays.
    """
    tf_mapping = {
        "5d": Interval.INTERVAL_1_HOUR,
        "1mo": Interval.INTERVAL_1_DAY,
        "3mo": Interval.INTERVAL_1_DAY,
        "1y": Interval.INTERVAL_1_WEEK
    }
    selected_interval = tf_mapping.get(timeframe, Interval.INTERVAL_1_DAY)
    clean_ticker = str(ticker).upper().strip()
    
    try:
        handler = TA_Handler(
            symbol=clean_ticker,
            exchange="NSE",
            screener="india",
            interval=selected_interval
        )
        analysis = handler.get_analysis()
        return {
            "status": "success",
            "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
            "buy_signals": analysis.summary.get("BUY", 0),
            "sell_signals": analysis.summary.get("SELL", 0),
            "neutral_signals": analysis.summary.get("NEUTRAL", 0)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to TradingView core database: {str(e)}"
        }

def generate_forecast_chart_data(ticker, trend_data, forecast_periods=5):
    """
    Generates a momentum-based drift projection and returns a clean 
    DataFrame for direct rendering in Streamlit.
    """
    try:
        if not trend_data or trend_data.get("status") == "error":
            return None
            
        latest_close = trend_data.get("latest_close")
        deviation_pct = trend_data.get("deviation_pct", 0)
        momentum_str = trend_data.get("momentum", "").upper()
        
        # Calculate drift based on moving average deviation
        drift_percentage = (deviation_pct / 100) / 10.0  
        if "BULLISH" in momentum_str:
            drift_direction = 1.0
            if drift_percentage <= 0: drift_percentage = 0.001
        elif "BEARISH" in momentum_str:
            drift_direction = -1.0
            if drift_percentage >= 0: drift_percentage = -0.001
        else:
            drift_direction = 0.0
            drift_percentage = 0.0
            
        projection_series = []
        current_simulated_price = latest_close
        
        # Generate future business day timestamps
        future_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=forecast_periods, freq='B')
        
        for date in future_dates:
            # Apply iterative momentum drift decay model
            step_change = current_simulated_price * drift_percentage * drift_direction
            current_simulated_price += step_change
            projection_series.append({
                "Date": date.strftime('%Y-%m-%d'),
                "Projected Target": round(current_simulated_price, 2)
            })
            
        # Create DataFrame - Note: Do not set Date as index to allow Streamlit 
        # to map the x-axis explicitly via st.line_chart parameters.
        projection_df = pd.DataFrame(projection_series)
        return projection_df
        
    except Exception as e:
        print(f"Error in forecast engine: {str(e)}")
        return None