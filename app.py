import streamlit as st
import streamlit.components.v1 as components
import io
import re
import pandas as pd
import os
from pydub import AudioSegment

from data_fetcher import (
    fetch_live_stock_price, 
    fetch_market_news, 
    fetch_gold_trend_analysis, 
    fetch_tradingview_gauge, 
    generate_forecast_chart_data,
    save_chat_session,
    load_chat_session
)
from router_agent import classify_intent, generate_financial_forecast, generate_live_price_response, get_tts_bytes, transcribe_audio_with_groq

# ===================================================
# 0. ADVANCED TEXTSTREAM PARSING MATRIX ENGINE
# ===================================================
def parse_llm_response_for_forecast(llm_text_response):
    """
    Scans J.A.R.V.I.S.'s verbal text string for a structured DATASTREAM block
    and dynamically converts those exact numbers into a rendering DataFrame.
    """
    if "DATASTREAM_START" not in llm_text_response:
        return None
        
    try:
        pattern = r"DATASTREAM_START\n(.*?)\nDATASTREAM_END"
        match = re.search(pattern, llm_text_response, re.DOTALL)
        
        if not match:
            return None 
            
        raw_rows = match.group(1).strip().split('\n')
        projection_series = []
        
        for row in raw_rows:
            if "Projected Target" in row or "---" in row or "Date" in row:
                continue
            if '|' in row:
                parts = [p.strip() for p in row.split('|')]
                if len(parts) >= 3:
                    projection_series.append({
                        "Date": parts[1],
                        "Projected Target": float(parts[2])
                    })
        if projection_series:
            return pd.DataFrame(projection_series)
    except Exception as e:
        print(f"Matrix extraction error: {str(e)}")
    return None

# ===================================================
# 1. HUD THEME & VISUAL INTERFACE CONFIGURATION
# ===================================================
st.set_page_config(page_title="J.A.R.V.I.S. Core HUD", layout="wide", initial_sidebar_state="expanded")

# Custom injection matrix to force deep terminal aesthetics
st.markdown("""
    <style>
    .reportview-container { background: #0b0202; }
    header, footer { visibility: hidden; }
    h1, h2, h3 { color: #b97d10 !important; font-family: 'Courier New', monospace; font-weight: bold; }
    .stChatInput { border: 1px solid #b97d10 !important; border-radius: 4px; background-color: #0b0202; }
    .stChatMessage { border-radius: 8px; margin-bottom: 10px; padding: 15px; }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1a0f00 !important;
        border-left: 5px solid #ffaa00 !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #0c0802 !important;
        border-left: 5px solid #b97d10 !important;
    }
    div[data-testid="stExpander"] { background-color: #0b0202 !important; border: 1px solid #b97d10 !important; }
    </style>
""", unsafe_allow_html=True)

def render_tradingview_gauge_ui(ticker):
    """
    Renders the gauge using a CSS-only dark-mode filter 
    and a custom container frame to match the UI.
    """
    clean_ticker = str(ticker).upper().replace(" ", "")
    tv_html = f"""
    <div style="
        background-color: #0b0202; 
        border: 2px solid #b97d10; 
        border-radius: 8px; 
        padding: 10px;
        filter: invert(1) hue-rotate(180deg);
    ">
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
          {{
            "interval": "1D",
            "width": "100%",
            "height": 330,
            "symbol": "NSE:{clean_ticker}",
            "theme": "light"
          }}
          </script>
        </div>
    </div>
    """
    components.html(tv_html, height=360)

# ===================================================
# 2. RUNTIME TELEMETRY CORE INITIALIZATION
# ===================================================
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_session() # Load from persistence file on browser initialize

if "awaiting_graph_confirmation" not in st.session_state:
    st.session_state.awaiting_graph_confirmation = False
if "staged_trend_data" not in st.session_state:
    st.session_state.staged_trend_data = None
if "staged_llm_text" not in st.session_state:
    st.session_state.staged_llm_text = None
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False

# ===================================================
# 3. SIDEBAR CONTROLS & UTILITIES MAPPING
# ===================================================
st.sidebar.markdown("# 🖥️ J.A.R.V.I.S. INTERFACE")
st.sidebar.markdown("---")
selected_timeframe = st.sidebar.selectbox("📈 GAUGE REFRESH MATRIX", ["1m", "5m", "15m", "1h", "4h", "1D", "1W"], index=5)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ CORES OPERATIONS")

# Purge control to easily clean local storage
if st.sidebar.button("💀 PURGE MEMORY MATRIX", use_container_width=True):
    st.session_state.messages = []
    if os.path.exists(".jarvis_session_history.json"):
        os.remove(".jarvis_session_history.json")
    st.rerun()

text_color = "#e6eaf1"

# ===================================================
# 4. PRIMARY REPLAY ENGINE (HISTORY DISPLAY)
# ===================================================
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(f"<span style='color:{text_color};'>{message['content']}</span>", unsafe_allow_html=True)
            
        elif message["type"] == "tradingview_gauge":
            st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>📊 TRADINGVIEW TECHNICAL ALIGNMENT:</span>", unsafe_allow_html=True)
            render_tradingview_gauge_ui(message["ticker"])
            
        elif message["type"] == "forecast_chart":
            st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>📊 PROJECTED HORIZON DATASTREAM:</span>", unsafe_allow_html=True)
            
            forecast_data = parse_llm_response_for_forecast(message["llm_source_text"])
            if forecast_data is None:
                forecast_data = generate_forecast_chart_data(message["ticker"], message["trend_data"], forecast_periods=5)
                
            if forecast_data is not None and not forecast_data.empty:
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
                st.line_chart(data=forecast_data, x="Date", y="Projected Target", color="#ffaa00")
                st.caption(f"Predictive tracking simulation captured for {message['ticker']}.")
            else:
                st.error("Sir, target graphical datablock corrupted inside logs.")

# TTS playback automation engine
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and not st.session_state.audio_played:
    last_msg = st.session_state.messages[-1]
    if last_msg["type"] == "text":
        clean_text = re.sub(r'<[^>]*>', '', last_msg["content"])
        clean_text = re.sub(r'DATASTREAM_START.*?DATASTREAM_END', '[Datastream Matrix]', clean_text, flags=re.DOTALL)
        audio_bytes = get_tts_bytes(clean_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    st.session_state.audio_played = True

# ===================================================
# 5. INPUT CHANNELS PROCESSING ENGINE (TEXT & AUDIO)
# ===================================================
user_prompt = None

audio_value = st.chat_input("Access console query terminal...")
audio_file = st.file_uploader("", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

if audio_file is not None:
    audio_bytes = audio_file.read()
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    with st.spinner("Decoding vocal telemetry stream..."):
        transcription = transcribe_audio_with_groq(wav_buffer)
        if transcription:
            user_prompt = transcription
else:
    user_prompt = audio_value

# ===================================================
# 6. PIPELINE ORCHESTRATION & STATE MANAGEMENT
# ===================================================
if user_prompt:
    processed_prompt = user_prompt.strip()
    
    # Text injection to display query immediately
    st.session_state.messages.append({"role": "user", "type": "text", "content": processed_prompt})
    save_chat_session(st.session_state.messages) # Persist user update
    
    with st.chat_message("user"):
        st.markdown(f"<span style='color:{text_color};'>{processed_prompt}</span>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        # Confirmation router layer path
        if st.session_state.awaiting_graph_confirmation:
            if any(confirm in processed_prompt.upper() for confirm in ["YES", "PROJECT", "PLOT", "DISPLAY", "SHOW"]):
                t_data = st.session_state.staged_trend_data
                ticker = t_data.get("ticker") if t_data else "Asset"
                
                jarvis_response = f"Affirmative, sir. Accessing computing array grids. Projecting temporal trajectory analytics for {ticker} now."
                st.markdown(f"<span style='color:{text_color};'>{jarvis_response}</span>", unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": jarvis_response})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "forecast_chart", 
                    "ticker": ticker, 
                    "trend_data": t_data,
                    "llm_source_text": st.session_state.staged_llm_text
                })
                
                st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>📊 PROJECTED HORIZON DATASTREAM:</span>", unsafe_allow_html=True)
                forecast_data = generate_forecast_chart_data(ticker, t_data, forecast_periods=5)
                if forecast_data is not None and not forecast_data.empty:
                    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
                    st.line_chart(data=forecast_data, x="Date", y="Projected Target", color="#ffaa00")
                
                st.session_state.awaiting_graph_confirmation = False
                st.session_state.staged_trend_data = None
                st.session_state.staged_llm_text = None
                save_chat_session(st.session_state.messages) # Persist chart injection
            else:
                jarvis_response = "Understood, sir. Chart rendering aborted. Standing by for alternative parameters."
                st.markdown(f"<span style='color:{text_color};'>{jarvis_response}</span>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": jarvis_response})
                st.session_state.awaiting_graph_confirmation = False
                st.session_state.staged_trend_data = None
                st.session_state.staged_llm_text = None
                save_chat_session(st.session_state.messages)
                
        else:
            with st.spinner("Synthesizing neural array network data..."):
                intent, target_ticker = classify_intent(processed_prompt)
                price_data = fetch_live_stock_price(target_ticker)
                trend_data = fetch_gold_trend_analysis(target_ticker)

            if intent == "PRICE":
                jarvis_response = generate_live_price_response(processed_prompt, price_data, trend_data)
                st.markdown(f"<span style='color:{text_color};'>{jarvis_response}</span>", unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": jarvis_response})
                st.session_state.messages.append({"role": "assistant", "type": "tradingview_gauge", "ticker": target_ticker})
                render_tradingview_gauge_ui(target_ticker)
                save_chat_session(st.session_state.messages) # Persist gauge session update
                    
            elif intent == "NEWS":
                news_headlines = fetch_market_news(target_ticker)
                tv_gauge = fetch_tradingview_gauge(target_ticker, timeframe=selected_timeframe)
                
                base_forecast = generate_financial_forecast(
                    processed_prompt, price_data, news_headlines, trend_data, tv_gauge, ticker=target_ticker
                )
                
                jarvis_response = base_forecast + f"\n\nShall I project the dynamic visual trend trajectory forecast chart for {target_ticker} across the upcoming trading sessions, sir?"
                st.markdown(f"<span style='color:{text_color};'>{jarvis_response}</span>", unsafe_allow_html=True)
                
                st.session_state.staged_trend_data = trend_data
                st.session_state.staged_llm_text = base_forecast
                st.session_state.awaiting_graph_confirmation = True
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": jarvis_response})
                save_chat_session(st.session_state.messages) # Persist staging query text
                
            else:
                jarvis_response = f"At your service, sir. Systems are focused on the {target_ticker} tracking array. Specify if you require real-time price monitoring or a predictive analysis block."
                st.markdown(f"<span style='color:{text_color};'>{jarvis_response}</span>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": jarvis_response})
                save_chat_session(st.session_state.messages)

    st.session_state.audio_played = False  
    st.rerun()