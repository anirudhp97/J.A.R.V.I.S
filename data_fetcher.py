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
            live_price = yf_ticker.fast_info.last_price
            
            return {
                "status": "success",
                "ticker": ticker,
                "company": asset_labels.get(ticker.upper(), ticker),
                "price": round(live_price, 2)
            }
        except Exception as inner_error:
            return {"status": "error", "message": f"All connection endpoints failed. Primary: {str(e)}. Fallback: {str(inner_error)}"}

def fetch_market_news(ticker):
    """
    Scrapes real-time market news summaries using Yahoo Finance vectors.
    """
    try:
        yf_symbol = f"{ticker}.NS"
        yf_ticker = yf.Ticker(yf_symbol)
        news_list = yf_ticker.news
        
        if not news_list:
            return f"No synchronized mainstream headlines available for {ticker} vector at this period."
            
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
        yf_symbol = f"{ticker}.NS"
        yf_ticker = yf.Ticker(yf_symbol)
        history = yf_ticker.history(period=period)
        
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
        return {"status": "error", "message": f"Trend processing failure: {str(e)}"}