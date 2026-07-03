import streamlit as st
import streamlit.components.v1 as components
import io
import re
import pandas as pd
import os
import html

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

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
            if "Projected Target" in row or "---" in row or "Date" in row or not row.strip():
                continue
                
            if "|" in row:
                parts = [p.strip() for p in row.split("|") if p.strip()]
                if len(parts) >= 2:
                    date_str = parts[0]
                    val_str = parts[1]
                    
                    if date_str == "Date" or "---" in date_str:
                        continue
                    try:
                        # Strip any stray currency signs or characters
                        val_clean = re.sub(r'[^\d.]', '', val_str)
                        projection_series.append({
                            "Date": date_str,
                            "Projected Target": float(val_clean)
                        })
                    except ValueError:
                        continue
                        
        if len(projection_series) == 0:
            return None
            
        return pd.DataFrame(projection_series)
    except Exception as e:
        print(f"[SYSTEM ALARM] Token tracking exception parsed: {str(e)}")
        return None


def contains_kannada(text):
    try:
        return bool(re.search(r'[\u0C80-\u0CFF]', str(text)))
    except Exception:
        return False

# 1. Page Config and Advanced Mark 42 Armor Hybrid Custom CSS Style Block
st.set_page_config(page_title="J.A.R.V.I.S. CORE HUD", layout="centered")

st.markdown("""
    <style>
        @keyframes reactorPulse {
            0% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%), radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%); }
            50% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.22) 0%, rgba(11, 2, 2, 0) 55%), radial-gradient(circle at 50% 30%, #2a0505 0%, #0b0202 70%); }
            100% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%), radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%); }
        }
        .stApp { background-color: #0b0202; color: #F7F1E3; font-family: 'Courier New', Courier, monospace; animation: reactorPulse 6s infinite ease-in-out; background-attachment: fixed; }
        h1 { color: #00E5FF !important; text-shadow: 0 0 10px #00E5FF, 0 0 20px #aa0505; letter-spacing: 4px; font-weight: 900; text-align: center; }
        [data-testid="stSidebar"] { background-color: #1a0505 !important; border-right: 3px solid #b97d10 !important; box-shadow: 5px 0px 15px rgba(185, 125, 16, 0.2); }
        .stChatMessage { border-radius: 6px !important; padding: 15px !important; margin-bottom: 12px !important; background-color: rgba(21, 5, 5, 0.85) !important; border: 1px solid #4d0907 !important; }
        [data-testid="stChatMessage"] div { font-family: 'Courier New', Courier, monospace !important; }
        .stCaption { color: #b97d10 !important; text-shadow: 0 0 3px rgba(185, 125, 16, 0.5); text-align: center; font-weight: bold; }
        hr { border-color: #b97d10 !important; }
    </style> """, unsafe_allow_html=True)

st.title("J.A.R.V.I.S.")
st.caption("System operational. All modules online. Awaiting directives, sir.")

def get_saved_ticker_from_query():
    ticker_param = st.query_params.get("ticker") if hasattr(st, "query_params") else None
    if isinstance(ticker_param, (list, tuple)):
        ticker_param = ticker_param[0]
    if isinstance(ticker_param, str):
        ticker_param = ticker_param.strip().upper()
        if ticker_param in ["GOLDBEES", "SILVERBEES", "NIFTYBEES"]:
            return ticker_param
    return None


def get_saved_timeframe_from_query():
    timeframe_param = st.query_params.get("timeframe") if hasattr(st, "query_params") else None
    if isinstance(timeframe_param, (list, tuple)):
        timeframe_param = timeframe_param[0]
    if isinstance(timeframe_param, str):
        timeframe_param = timeframe_param.strip().lower()
        if timeframe_param in ["5d", "1mo", "3mo", "1y"]:
            return timeframe_param
    return None


def get_saved_language_from_query():
    lang_param = st.query_params.get("language") if hasattr(st, "query_params") else None
    if isinstance(lang_param, (list, tuple)):
        lang_param = lang_param[0]
    if isinstance(lang_param, str):
        lang_param = lang_param.strip()
        if lang_param in ["English", "Kannada", "english", "kannada"]:
            # Normalize capitalization
            return "Kannada" if lang_param.lower() == "kannada" else "English"
    return None


def render_tradingview_gauge_ui(ticker):
    """
    Renders the gauge using a CSS-only dark-mode filter 
    and a custom container frame to match the J.A.R.V.I.S. UI.
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
    </div> """
    components.html(tv_html, height=360)

# Initialize Session State Machine Variables with local archival check
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = get_saved_ticker_from_query() or "GOLDBEES"

if "selected_timeframe" not in st.session_state:
    st.session_state.selected_timeframe = get_saved_timeframe_from_query() or "1mo"

if "messages" not in st.session_state:
    saved_history = load_chat_session()
    if saved_history:
        st.session_state.messages = saved_history

        # Restore the last known ticker only when no selection has been made yet.
        if st.session_state.active_ticker == "GOLDBEES":
            last_ticker = "GOLDBEES"
            for msg in reversed(saved_history):
                if "ticker" in msg and msg["ticker"]:
                    last_ticker = msg["ticker"]
                    break
            st.session_state.active_ticker = last_ticker
    else:
        st.session_state.messages = [{"role": "assistant", "type": "text", "content": "Hello sir, how may I help you today? Systems are fully functional. Tap the console input below to scan market metrics or generate a tactical telemetry forecast."}]

# Initialize language configuration state matrix
if "system_language" not in st.session_state:
    st.session_state.system_language = get_saved_language_from_query() or "English"

if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "last_valid_prompt" not in st.session_state:
    st.session_state.last_valid_prompt = None
if "awaiting_graph_confirmation" not in st.session_state:
    st.session_state.awaiting_graph_confirmation = False
if "staged_trend_data" not in st.session_state:
    st.session_state.staged_trend_data = None
if "staged_llm_text" not in st.session_state:
    st.session_state.staged_llm_text = None

# Sidebar System Controls
st.sidebar.markdown("<h3 style='color:#b97d10; text-shadow: 0 0 5px #b97d10;'>🛡️ ARMOR DIAGNOSTICS</h3>", unsafe_allow_html=True)

# 1. Multi-Language Selection Interface Widget Tracker
selected_lang = st.sidebar.selectbox(
    "Select System Language Interface:",
    options=["English", "Kannada"],
    key="language_select_widget"
)

if selected_lang != st.session_state.system_language:
    st.session_state.system_language = selected_lang
    try:
        st.query_params["language"] = st.session_state.system_language
    except Exception:
        pass

ticker_options = ["GOLDBEES", "SILVERBEES", "NIFTYBEES"]
try:
    current_ticker_index = ticker_options.index(st.session_state.active_ticker)
except ValueError:
    current_ticker_index = 0

selected_ticker = st.sidebar.selectbox(
    "Select Target Core Vector:", 
    options=ticker_options, 
    index=current_ticker_index,
    key="ticker_select_widget",
    format_func=lambda x: {
        "GOLDBEES": "🥇 GOLD BeES Monitor", 
        "SILVERBEES": "🥈 SILVER BeES Monitor", 
        "NIFTYBEES": "📈 NIFTY BeES Index"
    }.get(x, x)
)

if selected_ticker != st.session_state.active_ticker:
    st.session_state.active_ticker = selected_ticker

try:
    st.query_params["ticker"] = st.session_state.active_ticker
except Exception:
    pass

timeframe_options = ["5d", "1mo", "3mo", "1y"]
try:
    current_timeframe_index = timeframe_options.index(st.session_state.selected_timeframe)
except ValueError:
    current_timeframe_index = 1

selected_timeframe = st.sidebar.selectbox(
    "Select Trajectory Horizon:",
    options=timeframe_options,
    index=current_timeframe_index,
    key="timeframe_select_widget",
    format_func=lambda x: {"5d": "Past Week [5 Days]", "1mo": "Past Month [30 Days]", "3mo": "Quarterly Trend [3M]", "1y": "Annual Trend [1Y]"}[x]
)

if selected_timeframe != st.session_state.selected_timeframe:
    st.session_state.selected_timeframe = selected_timeframe

try:
    st.query_params["ticker"] = st.session_state.active_ticker
    st.query_params["timeframe"] = st.session_state.selected_timeframe
    st.query_params["language"] = st.session_state.system_language
except Exception:
    pass

st.sidebar.markdown(f"""
<div style='border: 1px solid #b97d10; padding: 12px; border-radius: 4px; background-color: rgba(170,5,5,0.15); margin-top: 25px; margin-bottom: 15px;'>
    <span style='color: #00E5FF; font-size: 11px; font-weight: bold; letter-spacing: 1px;'>HUD CHANNEL FEED</span><br>
    <span style='color: #4AF2A1; font-size: 12px;'>● ACTIVE VECT: {st.session_state.active_ticker}</span><br>
    <span style='color: #FCE154; font-size: 10px;'>LANG MODEL: {st.session_state.system_language.upper()}</span>
</div> """, unsafe_allow_html=True)

# Purge control option
if st.sidebar.button("💀 PURGE TERMINAL LOGS", use_container_width=True):
    st.session_state.messages = [{"role": "assistant", "type": "text", "content": "Core memory matrix flushed, sir. Operational arrays re-initialized."}]
    if os.path.exists(".jarvis_session_history.json"):
        os.remove(".jarvis_session_history.json")
    st.session_state.active_ticker = "GOLDBEES"
    st.rerun()

st.sidebar.markdown("<h3 style='color:#00E5FF; text-shadow: 0 0 5px #00E5FF;'>📊 HUD VISUALS</h3>", unsafe_allow_html=True)
with st.sidebar.expander(f"🔮 {st.session_state.active_ticker} Core Consensus", expanded=True):
    render_tradingview_gauge_ui(st.session_state.active_ticker)

# Display Chat History Flow Architecture
for index, message in enumerate(st.session_state.messages):
    prefix = "T. STARK [COM-LINK]:" if message["role"] == "user" else "J.A.R.V.I.S.:"
    text_color = "#FCC200" if message["role"] == "user" else "#00E5FF"
    
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            raw_content = message.get('content', '')
            # Remove any previously-saved HTML tags to avoid printing raw markup
            raw_no_tags = re.sub(r'<[^>]+>', '', str(raw_content))
            safe_content = html.escape(raw_no_tags)
            # Apply Kannada-capable font when Kannada glyphs are present to avoid gibberish
            if contains_kannada(raw_content):
                content_style = "color:#E2F1FF; font-family: 'Nirmala UI', 'Noto Sans Kannada', 'Arial Unicode MS', sans-serif;"
            else:
                content_style = "color:#E2F1FF; font-family: 'Courier New', Courier, monospace;"

            st.markdown(
                f"<span style='color:{text_color}; font-weight:bold;'>{prefix}</span> "
                f"<span style='{content_style}'>{safe_content}</span>",
                unsafe_allow_html=True
            )
            
            if message["role"] == "assistant" and (index == len(st.session_state.messages) - 1) and not st.session_state.audio_played:
                with st.spinner("Initializing vocal transmission channels..."):
                    # Pass chosen interface language to render specific tone arrays
                    audio_bytes = get_tts_bytes(message["content"], language=st.session_state.system_language)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav", autoplay=True)
                        st.session_state.audio_played = True
                        
        elif message["type"] == "forecast_chart":
            st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>📊 PROJECTED HORIZON DATASTREAM:</span>", unsafe_allow_html=True)
            
            # Retrieve data points correctly
            forecast_data = parse_llm_response_for_forecast(message["llm_source_text"])
            if forecast_data is None:
                forecast_data = generate_forecast_chart_data(message["ticker"], message["trend_data"], forecast_periods=5)
                
            if forecast_data is not None and not forecast_data.empty:
                if not isinstance(forecast_data.index, pd.DatetimeIndex):
                    forecast_data.index = pd.to_datetime(forecast_data.index)

                if "Projected Target" in forecast_data.columns and forecast_data["Projected Target"].notna().any():
                    st.line_chart(forecast_data, y="Projected Target", color="#ffaa00")
                    st.caption(f"Predictive tracking simulation captured from vector context data matrix for {message['ticker']}.")
                else:
                    st.error("Sir, target graphical datablock contains no usable numeric values.")
            else:
                st.error("Sir, target graphical datablock corrupted inside logs.")

st.divider()

# Input UI Controls
if "mic_rotation_counter" not in st.session_state:
    st.session_state.mic_rotation_counter = 0

widget_key = f"voice_recorder_v_{st.session_state.mic_rotation_counter}_{st.session_state.active_ticker}_{st.session_state.system_language}"
audio_file = st.audio_input("🎙️ ACTIVATE VOCAL RECEIVER", key=widget_key)
user_text_input = st.chat_input("Input terminal override command here...")

processed_prompt = None

if audio_file:
    with st.spinner("Decoding vocal sequence arrays via Groq Whisper..."):
        try:
            if AudioSegment is None:
                raise ImportError("pydub is not installed")

            audio_bytes_raw = audio_file.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_raw))
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # Forward active target language matrix to Whisper decoder
            detected_text = transcribe_audio_with_groq(wav_buffer, language=st.session_state.system_language)
            if detected_text: processed_prompt = detected_text
            st.session_state.mic_rotation_counter += 1
        except Exception as e:
            st.error(f"Core communication breakdown: {str(e)}.")
            st.session_state.mic_rotation_counter += 1
elif user_text_input:
    processed_prompt = user_text_input

# Main Pipeline Execution Engine
if processed_prompt:
    clean_prompt = processed_prompt.upper().strip()
    
    is_confirmation = any(clean_prompt.startswith(word) for word in ["YES", "YEA", "OK", "PLEASE", "SURE", "GO AHEAD", "GENERATE", "PROJECT", "ಹೌದು", "ಮಾಡು"])
    is_negation = any(clean_prompt.startswith(word) for word in ["NO", "DON'T", "STOP", "NEVER", "SKIP", "NAH", "ಬೇಡ", "ನಿಲ್ಲಿಸು"])
    
    # Store ticker context metadata inside user message to help state restoration during refresh
    st.session_state.messages.append({
        "role": "user", 
        "type": "text", 
        "content": processed_prompt,
        "ticker": st.session_state.active_ticker
    })
    save_chat_session(st.session_state.messages)
    
    if st.session_state.awaiting_graph_confirmation:
        st.session_state.awaiting_graph_confirmation = False
        if is_confirmation:
            st.session_state.messages.append({
                "role": "assistant",
                "type": "forecast_chart",
                "ticker": st.session_state.active_ticker,
                "trend_data": st.session_state.staged_trend_data,
                "llm_source_text": st.session_state.staged_llm_text
            })
            if st.session_state.system_language == "Kannada":
                jarvis_response = f"ಅರ್ಥವಾಯಿತು ಸರ್. ರೆಂಡರಿಂಗ್ ಪ್ರಕ್ರಿಯೆ ಆರಂಭಿಸಲಾಗಿದೆ. {st.session_state.active_ticker} ಮುನ್ಸೂಚನೆ ಚಾರ್ಟ್ ಅನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಲಾಗ್ ಮಾಡಲಾಗಿದೆ."
            else:
                jarvis_response = f"Understood, sir. Initializing rendering sequences. Visual forecast tracks for the {st.session_state.active_ticker} vector have been securely committed to the logs above."
        elif is_negation:
            if st.session_state.system_language == "Kannada":
                jarvis_response = "ಖಂಡಿತ ಸರ್, ಚಾರ್ಟ್ ಪ್ರಕ್ಷೇಪಣವನ್ನು ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ. ನಾನು ಮುಂದಿನ ಆಜ್ಞೆಗಾಗಿ ಕಾಯುತ್ತಿದ್ದೇನೆ."
            else:
                jarvis_response = "Acknowledged, sir. Bypassing graphical predictive tracking. I shall keep a silent watch on active telemetry frequencies on standby."
        else:
            if st.session_state.system_language == "Kannada":
                jarvis_response = "ಟೆಲಿಮೆಟ್ರಿ ರಿಸೆಟ್ ಮಾಡಲಾಗಿದೆ. ಲೈವ್ ಬೆಲೆ ಅಥವಾ ಟ್ರೆಂಡ್ ವಿಶ್ಲೇಷಣೆ ಬೇಕೇ ಎಂದು ತಿಳಿಸಿ ಸರ್."
            else:
                jarvis_response = "Telemetry projection reset, sir. State your objective: live price tracking or trend analysis modeling."
            
        st.session_state.messages.append({
            "role": "assistant", 
            "type": "text", 
            "content": jarvis_response,
            "ticker": st.session_state.active_ticker
        })
        st.session_state.staged_trend_data = None
        st.session_state.staged_llm_text = None
        save_chat_session(st.session_state.messages)
            
    else:
        if any(w in clean_prompt for w in ["GOLD BEES", "GOLDBEES", "PRICE OF GOLD", "ಚಿನ್ನ"]): st.session_state.active_ticker = "GOLDBEES"
        elif any(w in clean_prompt for w in ["SILVER BEES", "SILVERBEES", "ಬೆಳ್ಳಿ"]): st.session_state.active_ticker = "SILVERBEES"
        elif any(w in clean_prompt for w in ["NIFTY BEES", "NIFTYBEES", "ನಿಫ್ಟಿ"]): st.session_state.active_ticker = "NIFTYBEES"
        
        target_ticker = st.session_state.active_ticker
        
        with st.spinner("J.A.R.V.I.S. is compiling data streams..." if st.session_state.system_language == "English" else "ಮಾಹಿತಿಯನ್ನು ಸಂಗ್ರಹಿಸಲಾಗುತ್ತಿದೆ..."):
            intent = classify_intent(processed_prompt)
            trend_data = fetch_gold_trend_analysis(target_ticker, period=selected_timeframe)
            price_data = fetch_live_stock_price(target_ticker)
            
            if intent == "LIVE":
                if price_data["status"] == "success":
                    jarvis_response = generate_live_price_response(
                        processed_prompt, price_data, trend_data, language=st.session_state.system_language
                    )
                else:
                    if st.session_state.system_language == "Kannada":
                        jarvis_response = f"ಸರ್, ಎಕ್ಸ್‌ಚೇಂಜ್ ಸರ್ವರ್ ಸಂಪರ್ಕಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ: {price_data.get('message')}"
                    else:
                        jarvis_response = f"Sir, I am unable to connect to the exchange floor: {price_data.get('message')}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "text", 
                    "content": jarvis_response,
                    "ticker": target_ticker
                })
                save_chat_session(st.session_state.messages)
                    
            elif intent == "NEWS":
                news_headlines = fetch_market_news(target_ticker)
                tv_gauge = fetch_tradingview_gauge(target_ticker, timeframe=selected_timeframe)
                
                base_forecast = generate_financial_forecast(
                    processed_prompt, price_data, news_headlines, trend_data, tv_gauge, ticker=target_ticker, language=st.session_state.system_language
                )
                
                if st.session_state.system_language == "Kannada":
                    jarvis_response = base_forecast + f"\n\nನಾನು {target_ticker} ಗಾಗಿ ಮುನ್ಸೂಚನೆ ಚಾರ್ಟ್ ಅನ್ನು ತೋರಿಸಲೇ ಸರ್?"
                else:
                    jarvis_response = base_forecast + f"\n\nShall I project the dynamic visual trend trajectory forecast chart for {target_ticker} across the upcoming trading sessions, sir?"
                
                st.session_state.staged_trend_data = trend_data
                st.session_state.staged_llm_text = base_forecast
                st.session_state.awaiting_graph_confirmation = True
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "text", 
                    "content": jarvis_response,
                    "ticker": target_ticker
                })
                save_chat_session(st.session_state.messages)
                
            else:
                if st.session_state.system_language == "Kannada":
                    jarvis_response = f"ನಿಮ್ಮ ಸೇವೆಗೆ ಸಿದ್ಧನಿದ್ದೇನೆ ಸರ್. ಸಿಸ್ಟಮ್ ಪ್ರಸ್ತುತ {target_ticker} ಮೇಲೆ ಕೇಂದ್ರೀಕೃತವಾಗಿದೆ. ಲೈವ್ ಬೆಲೆ ಅಥವಾ ಮುನ್ಸೂಚನೆ ಬೇಕೇ ಎಂದು ತಿಳಿಸಿ."
                else:
                    jarvis_response = f"At your service, sir. Systems are focused on the {target_ticker} tracking array. Specify if you require real-time price monitoring or a predictive analysis block."
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "text", 
                    "content": jarvis_response,
                    "ticker": target_ticker
                })
                save_chat_session(st.session_state.messages)

    st.session_state.audio_played = False  
    st.rerun()