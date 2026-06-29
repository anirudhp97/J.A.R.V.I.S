import streamlit as st
import streamlit.components.v1 as components
import io
import re
import pandas as pd  
import speech_recognition as sr
from data_fetcher import fetch_live_stock_price, fetch_market_news, fetch_gold_trend_analysis, fetch_tradingview_gauge
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
                
            if "|" in row:
                parts = row.split("|")
                date_str = parts[0].strip()
                price_str = parts[1].strip()
                
                # Sanitize out casual symbols (₹, $, letters) safely
                price_clean = re.sub(r'[^\d.]', '', price_str)
                
                if date_str and price_clean:
                    projection_series.append({
                        "Date": date_str,
                        "Projected Target": float(price_clean)
                    })
        
        if not projection_series:
            return None
            
        projection_df = pd.DataFrame(projection_series)
        projection_df.set_index("Date", inplace=True)
        return projection_df
        
    except Exception as e:
        print(f"[JARVIS SYSTEM ALARM] Text stream parsing matrix error: {str(e)}")
        return None

# ===================================================
# 1. Page Config and Advanced Mark 42 Armor Hybrid Custom CSS Style Block
# ===================================================
st.set_page_config(page_title="M.A.R.K. XLII CORE HUD", page_icon="🦾", layout="centered")

st.markdown("""
    <style>
        /* Deep space cyber background */
        .stApp {
            background-color: #0b0202;
            color: #F7F1E3;
            font-family: 'Courier New', Courier, monospace;
            background-image: radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%);
        }
        
        /* Stark Industries Glow Header */
        h1 {
            color: #00E5FF !important;
            text-shadow: 0 0 10px #00E5FF, 0 0 20px #aa0505;
            letter-spacing: 4px;
            font-weight: 900;
            text-align: center;
        }
        
        /* Sidebar Styling: Mark 42 Crimson Armor Plating with Gold Borders */
        [data-testid="stSidebar"] {
            background-color: #1a0505 !important;
            border-right: 3px solid #b97d10 !important;
            box-shadow: 5px 0px 15px rgba(185, 125, 16, 0.2);
        }
        
        /* Chat Bubble Custom Formatting overrides */
        .stChatMessage {
            border-radius: 6px !important;
            padding: 15px !important;
            margin-bottom: 12px !important;
            background-color: rgba(21, 5, 5, 0.85) !important;
            border: 1px solid #4d0907 !important;
        }
        
        [data-testid="stChatMessage"] div {
            font-family: 'Courier New', Courier, monospace !important;
        }
        
        .stCaption {
            color: #b97d10 !important;
            text-shadow: 0 0 3px rgba(185, 125, 16, 0.5);
            text-align: center;
            font-weight: bold;
        }
        
        hr {
            border-color: #b97d10 !important;
        }

        /* 🛠️ TRADINGVIEW IFRAME LIGHT-THEME INJECTION LAYER */
        div[data-testid="stHtmlWrapper"] span, 
        div[data-testid="stHtmlWrapper"] p, 
        div[data-testid="stHtmlWrapper"] div,
        div[data-testid="stHtmlWrapper"] b {
            color: #ffffff !important;
        }
        div[class*="dropdown"], select, option, button {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("J.A.R.V.I.S.")
st.caption("Systems Operational. Core Diagnostics Active. Awaiting Command Input...")

# Analytics Control Console on Sidebar (Champagne Gold Accents)
st.sidebar.markdown("<h3 style='color:#b97d10; text-shadow: 0 0 5px #b97d10;'>🛡️ ARMOR DIAGNOSTICS</h3>", unsafe_allow_html=True)

# Multi-Asset Tracker Selector Matrix
selected_ticker = st.sidebar.selectbox(
    "Select Target Core Vector:",
    options=["GOLDBEES", "SILVERBEES", "NIFTYBEES"],
    index=0,
    format_func=lambda x: {
        "GOLDBEES": "🥇 GOLD BeES Monitor",
        "SILVERBEES": "🥈 SILVER BeES Monitor",
        "NIFTYBEES": "📈 NIFTY BeES Index"
    }[x]
)

selected_timeframe = st.sidebar.selectbox(
    "Select Trajectory Horizon:",
    options=["5d", "1mo", "3mo", "1y"],
    index=1,
    format_func=lambda x: {
        "5d": "Past Week [5 Days]",
        "1mo": "Past Month [30 Days]",
        "3mo": "Quarterly Trend [3M]",
        "1y": "Annual Trend [1Y]"
    }[x]
)

st.sidebar.markdown(f"""
<div style='border: 1px solid #b97d10; padding: 12px; border-radius: 4px; background-color: rgba(170,5,5,0.15); margin-top: 25px;'>
    <span style='color: #00E5FF; font-size: 11px; font-weight: bold; letter-spacing: 1px;'>HUD CHANNEL FEED</span><br>
    <span style='color: #4AF2A1; font-size: 12px;'>● ACTIVE VECT: {selected_ticker}</span><br>
    <span style='color: #FCE154; font-size: 10px;'>CORE STABILITY: LOCKED</span>
</div>
""", unsafe_allow_html=True)

# ===================================================
# 2. Session State Vector Initialization
# ===================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello sir, how may I help you today? Systems are fully functional. Tap the console input below to scan market metrics or generate a tactical telemetry forecast."
        }
    ]
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "awaiting_graph_confirmation" not in st.session_state:
    st.session_state.awaiting_graph_confirmation = False
if "staged_trend_data" not in st.session_state:
    st.session_state.staged_trend_data = None

# ===================================================
# 3. Render Historical Chat Log with Custom Overrides
# ===================================================
for index, message in enumerate(st.session_state.messages):
    message_payload = message.get("content", "")
    role = message.get("role", "user")
    
    if role == "user":
        prefix = "T. STARK [COM-LINK]:"
        text_color = "#FCC200" 
    else:
        prefix = "J.A.R.V.I.S.:"
        text_color = "#00E5FF" 
        
    with st.chat_message(role):
        # Clean out raw chart string blocks before showing user text strings
        if "DATASTREAM_START" in message_payload:
            display_text = re.sub(r"DATASTREAM_START\n.*?\nDATASTREAM_END", "", message_payload, flags=re.DOTALL).strip()
        else:
            display_text = message_payload
            
        st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>{prefix}</span> <span style='color:#E2F1FF;'>{display_text}</span>", unsafe_allow_html=True)
        
        # Render out charts if a valid telemetry matrix was decoded
        forecast_df = parse_llm_response_for_forecast(message_payload)
        if forecast_df is not None:
            st.markdown("### 📊 PROJECTED HORIZON DATASTREAM:")
            st.line_chart(forecast_df)
            st.dataframe(forecast_df, use_container_width=True)

        if role == "assistant" and not st.session_state.audio_played and (index == len(st.session_state.messages) - 1):
            with st.spinner("Initializing vocal transmission channels..."):
                audio_bytes = get_tts_bytes(display_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
            st.session_state.audio_played = True

st.divider()

# ===================================================
# 4. Input Controls Framework
# ===================================================
if "mic_rotation_counter" not in st.session_state:
    st.session_state.mic_rotation_counter = 0

widget_key = f"voice_recorder_v_{st.session_state.mic_rotation_counter}_{selected_ticker}"
audio_file = st.audio_input("🎙️ ACTIVATE VOCAL RECEIVER", key=widget_key)
user_text_input = st.chat_input("Input terminal override command here...")

processed_prompt = None

# Pipeline Phase 1: Input Audio Capture & Local STT Processing
if audio_file:
    with st.spinner("Decoding vocal sequence arrays..."):
        try:
            audio_bytes_raw = audio_file.read()
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes_raw)) as source:
                audio_data = recognizer.record(source)
                detected_text = recognizer.recognize_google(audio_data)
                processed_prompt = detected_text
                
            st.session_state.mic_rotation_counter += 1
                
        except sr.UnknownValueError:
            st.error("Signal interference detected. Please repeat the command clearly.")
            st.session_state.mic_rotation_counter += 1
        except Exception as e:
            st.error(f"Core communication breakdown: {str(e)}.")
            st.session_state.mic_rotation_counter += 1

elif user_text_input:
    processed_prompt = user_text_input

# ===================================================
# 5. Pipeline Phase 2: Agentic Routing and Operational Loop
# ===================================================
if processed_prompt:
    cleaned_prompt = processed_prompt.upper().strip()
    st.session_state.messages.append({"role": "user", "content": processed_prompt})
    
    with st.spinner("J.A.R.V.I.S. is compiling data streams..."):
        
        # Route confirmation states safely via textual confirmation responses
        if st.session_state.get("awaiting_graph_confirmation", False):
            if "YES" in cleaned_prompt:
                jarvis_response = (
                    f"Acknowledged, sir. Compiling the visual trajectory projection metrics for {selected_ticker}.\n\n"
                    "Here is the projected datastream sequence for the upcoming session windows:\n\n"
                )
                
                generation_prompt = (
                    f"Generate a short 4-5 business day forecast response for {selected_ticker} "
                    "concluding with the mandatory DATASTREAM_START and DATASTREAM_END block containing dates and projected target prices."
                )
                
                news_headlines = fetch_market_news(selected_ticker)
                tv_gauge = fetch_tradingview_gauge(selected_ticker)
                
                jarvis_response += generate_financial_forecast(
                    generation_prompt, 
                    fetch_live_stock_price(selected_ticker), 
                    news_headlines, 
                    st.session_state.staged_trend_data, 
                    tv_gauge, 
                    ticker=selected_ticker
                )
                st.session_state.awaiting_graph_confirmation = False
            else:
                jarvis_response = "Understood, sir. Trajectory plotting stands down. Standing by for further monitoring requests."
                st.session_state.awaiting_graph_confirmation = False
        
        else:
            # Standard Path Intent Evaluation
            intent = classify_intent(processed_prompt)
            trend_data = fetch_gold_trend_analysis(selected_ticker, period=selected_timeframe)
            price_data = fetch_live_stock_price(selected_ticker)
            
            if intent == "LIVE":
                if price_data["status"] == "success":
                    jarvis_response = generate_live_price_response(processed_prompt, price_data, trend_data)
                else:
                    jarvis_response = f"Sir, I am unable to connect to the exchange floor: {price_data.get('message')}"
                    
            elif intent == "NEWS":
                news_headlines = fetch_market_news(selected_ticker)
                tv_gauge = fetch_tradingview_gauge(selected_ticker, timeframe=selected_timeframe)
                
                base_forecast = generate_financial_forecast(
                    processed_prompt, price_data, news_headlines, trend_data, tv_gauge, ticker=selected_ticker
                )
                
                jarvis_response = base_forecast + f"\n\nShall I project the dynamic visual trend trajectory forecast chart for {selected_ticker} across the upcoming trading sessions, sir?"
                
                st.session_state.staged_trend_data = trend_data
                st.session_state.awaiting_graph_confirmation = True
                
            else:
                jarvis_response = (
                    f"At your service, sir. The system's diagnostic analytics are currently targeted at the {selected_ticker} vector "
                    f"across a {selected_timeframe} horizon. Specify if you would like me to extract live price telemetry or formulate a macro prediction."
                )
            
    st.session_state.messages.append({"role": "assistant", "content": jarvis_response})
    st.session_state.audio_played = False
    st.rerun()