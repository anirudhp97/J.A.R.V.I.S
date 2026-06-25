import yfinance as yf
from nsepython import nse_eq, nse_quote_ltp
import pandas as pd

def fetch_live_stock_price(ticker):
    """
    Fetches the true real-time Last Traded Price (LTP) 
    directly from the National Stock Exchange of India with smart asset naming.
    """
    asset_labels = {
        "GOLDBEES": "Nippon India ETF Gold BeES",
        "SILVERBEES": "Nippon India ETF Silver BeES",
        "NIFTYBEES": "Nippon India ETF Nifty BeES"
    }
    
    try:
        raw_data = nse_eq(ticker)
        company_name = raw_data.get('info', {}).get('companyName')
        
        if not company_name:
            company_name = asset_labels.get(ticker.upper(), ticker)
            
        ltp = raw_data.get('priceInfo', {}).get('lastPrice', None)
        
        if not ltp:
            ltp = nse_quote_ltp(ticker)

        return {
            "status": "success",
            "ticker": ticker,
            "company": company_name,
            "price": ltp
        }
    except Exception as e:
        try:
            yf_symbol = f"{ticker}.NS"
            yf_ticker = yf.Ticker(yf_symbol)
            live_data = yf_ticker.history(period="1d")
            if not live_data.empty:
                return {
                    "status": "success",
                    "ticker": ticker,
                    "company": asset_labels.get(ticker.upper(), ticker),
                    "price": round(live_data['Close'].iloc[-1], 2)
                }
        except:
            pass
        return {"status": "error", "message": f"Market Ingestion failed: {str(e)}"}

def fetch_market_news(ticker):
    """
    Queries Yahoo Finance media endpoints to parse and structure
    recent news headlines related to the specific target asset.
    """
    try:
        yf_symbol = f"{ticker}.NS"
        tracker = yf.Ticker(yf_symbol)
        news_payload = tracker.news
        
        if not news_payload:
            return f"No breaking market announcements tracked for {ticker} over the last 48 hours."
        
        formatted_headlines = ""
        for index, article in enumerate(news_payload[:3], start=1):
            title = article.get('title', 'Unknown Headline')
            publisher = article.get('publisher', 'Financial Network')
            formatted_headlines += f"Headline {index}: '{title}' published by {publisher}.\n"
            
        return formatted_headlines
    except Exception as e:
        return f"Failed to retrieve secondary news context: {str(e)}"

def fetch_gold_trend_analysis(ticker, period="1mo"):
    """
    Downloads historical data based on user-specified periods ('5d', '1mo', '3mo', '1y')
    and evaluates price momentum for the specific selected asset vector.
    """
    try:
        yf_symbol = f"{ticker}.NS"
        tracker = yf.Ticker(yf_symbol)
        history = tracker.history(period=period)
        
        if history.empty or len(history) < 2:
            return {"status": "error", "message": f"Insufficient data available for {ticker} period: {period}"}
            
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
        return {"status": "error", "message": str(e)}