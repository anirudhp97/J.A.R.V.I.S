import streamlit as st
import streamlit.components.v1 as components
import io
from pydub import AudioSegment
from data_fetcher import fetch_live_stock_price, fetch_market_news, fetch_gold_trend_analysis, fetch_tradingview_gauge, fetch_historical_chart_data
from router_agent import classify_intent, generate_financial_forecast, generate_live_price_response, get_tts_bytes, transcribe_audio_with_groq

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
    </style>
""", unsafe_allow_html=True)

st.title("J.A.R.V.I.S.")
st.caption("System operational. All modules online. Awaiting directives, sir.")

def render_tradingview_gauge_ui(ticker):
    """ Renders a live, responsive visual Technical Analysis gauge in the UI. """
    clean_ticker = str(ticker).upper().replace(" ", "")
    tv_html = f"""
    <div class="tradingview-widget-container" style="margin:auto; width:100%; max-width:450px;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "1D", "width": "100%", "isTransparent": true, "height": 360,
        "symbol": "NSE:{clean_ticker}", "showIntervalTabs": true, "displayMode": "single", "locale": "en", "theme": "dark"
      }}
      </script>
    </div>
    """
    components.html(tv_html, height=380)

# Sidebar System Controls
st.sidebar.markdown("<h3 style='color:#b97d10; text-shadow: 0 0 5px #b97d10;'>🛡️ ARMOR DIAGNOSTICS</h3>", unsafe_allow_html=True)

selected_ticker = st.sidebar.selectbox(
    "Select Target Core Vector:", options=["GOLDBEES", "SILVERBEES", "NIFTYBEES"], index=0,
    format_func=lambda x: {"GOLDBEES": "🥇 GOLD BeES Monitor", "SILVERBEES": "🥈 SILVER BeES Monitor", "NIFTYBEES": "📈 NIFTY BeES Index"}[x]
)

selected_timeframe = st.sidebar.selectbox(
    "Select Trajectory Horizon:", options=["5d", "1mo", "3mo", "1y"], index=1,
    format_func=lambda x: {"5d": "Past Week [5 Days]", "1mo": "Past Month [30 Days]", "3mo": "Quarterly Trend [3M]", "1y": "Annual Trend [1Y]"}[x]
)

# Initialize Session State Machine Variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello sir, how may I help you today? Systems are fully functional. Tap the console input below to scan market metrics or generate a tactical telemetry forecast."}]
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = selected_ticker
if "last_valid_prompt" not in st.session_state:
    st.session_state.last_valid_prompt = None
if "awaiting_graph_confirmation" not in st.session_state:
    st.session_state.awaiting_graph_confirmation = False
if "render_active_graph" not in st.session_state:
    st.session_state.render_active_graph = False

st.sidebar.markdown(f"""
<div style='border: 1px solid #b97d10; padding: 12px; border-radius: 4px; background-color: rgba(170,5,5,0.15); margin-top: 25px; margin-bottom: 15px;'>
    <span style='color: #00E5FF; font-size: 11px; font-weight: bold; letter-spacing: 1px;'>HUD CHANNEL FEED</span><br>
    <span style='color: #4AF2A1; font-size: 12px;'>● ACTIVE VECT: {st.session_state.active_ticker}</span><br>
    <span style='color: #FCE154; font-size: 10px;'>CORE STABILITY: LOCKED</span>
</div>
""", unsafe_allow_html=True)

# Sidebar HUD Visual components
st.sidebar.markdown("<h3 style='color:#00E5FF; text-shadow: 0 0 5px #00E5FF;'>📊 HUD VISUALS</h3>", unsafe_allow_html=True)
with st.sidebar.expander(f"🔮 {st.session_state.active_ticker} Core Consensus", expanded=True):
    render_tradingview_gauge_ui(st.session_state.active_ticker)

# Display historical main chat loop
for index, message in enumerate(st.session_state.messages):
    prefix = "T. STARK [COM-LINK]:" if message["role"] == "user" else "J.A.R.V.I.S.:"
    text_color = "#FCC200" if message["role"] == "user" else "#00E5FF"
    with st.chat_message(message["role"]):
        st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>{prefix}</span> <span style='color:#E2F1FF;'>{message['content']}</span>", unsafe_allow_html=True)
        
        if message["role"] == "assistant" and (index == len(st.session_state.messages) - 1) and not st.session_state.audio_played:
            with st.spinner("Initializing vocal transmission channels..."):
                audio_bytes = get_tts_bytes(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                    st.session_state.audio_played = True

# --- DYNAMIC CHART TELEMETRY PLOTTING BLOCK ---
if st.session_state.render_active_graph:
    st.markdown("---")
    st.subheader(f"📊 {st.session_state.active_ticker} Historical Telemetry Mapping")
    with st.spinner("Rendering historical trend vectors..."):
        chart_data = fetch_historical_chart_data(st.session_state.active_ticker, period=selected_timeframe)
        if chart_data is not None and not chart_data.empty:
            st.line_chart(chart_data, y="Close", color="#00e5ff")
            st.caption(f"Historical line display depicting trailing closing prices across a {selected_timeframe} tracking index.")
        else:
            st.error("Sir, historical chart blocks are currently offline due to a data link timeout.")

st.divider()

# Input UI Controls
if "mic_rotation_counter" not in st.session_state:
    st.session_state.mic_rotation_counter = 0

widget_key = f"voice_recorder_v_{st.session_state.mic_rotation_counter}_{selected_ticker}"
audio_file = st.audio_input("🎙️ ACTIVATE VOCAL RECEIVER", key=widget_key)
user_text_input = st.chat_input("Input terminal override command here...")

processed_prompt = None

if audio_file:
    with st.spinner("Decoding vocal sequence arrays via Groq Whisper..."):
        try:
            audio_bytes_raw = audio_file.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_raw))
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            detected_text = transcribe_audio_with_groq(wav_buffer)
            if detected_text: processed_prompt = detected_text
            st.session_state.mic_rotation_counter += 1
        except Exception as e:
            st.error(f"Core communication breakdown: {str(e)}.")
            st.session_state.mic_rotation_counter += 1
elif user_text_input:
    processed_prompt = user_text_input

# Main Pipeline Execution
if processed_prompt:
    clean_prompt = processed_prompt.upper().strip()
    
    # Check Confirmation Signals
    is_confirmation = any(clean_prompt.startswith(word) for word in ["YES", "YEA", "OK", "PLEASE", "SURE", "GO AHEAD", "GENERATE"])
    is_negation = any(clean_prompt.startswith(word) for word in ["NO", "DON'T", "STOP", "NEVER", "SKIP", "NAH"])
    
    st.session_state.messages.append({"role": "user", "content": processed_prompt})
    
    # Logic Gate: Graph Confirmation State Flow
    if st.session_state.awaiting_graph_confirmation:
        st.session_state.awaiting_graph_confirmation = False
        if is_confirmation:
            st.session_state.render_active_graph = True
            jarvis_response = f"Understood, sir. Initializing rendering sequences. Visual coordinates for the {st.session_state.active_ticker} vector have been mapped to the main projection display screen."
        elif is_negation:
            st.session_state.render_active_graph = False
            jarvis_response = "Acknowledged, sir. Bypassing graphical rendering sequences. I shall continue to monitor active telemetry frequencies on standby."
        else:
            st.session_state.render_active_graph = False
            jarvis_response = "Telemetry tracking reset, sir. State your objective: live price calculation or macro forecast modeling."
            
    # Standard Context Processing Flow
    else:
        if any(w in clean_prompt for w in ["GOLD BEES", "GOLDBEES", "PRICE OF GOLD"]): st.session_state.active_ticker = "GOLDBEES"
        elif any(w in clean_prompt for w in ["SILVER BEES", "SILVERBEES"]): st.session_state.active_ticker = "SILVERBEES"
        elif any(w in clean_prompt for w in ["NIFTY BEES", "NIFTYBEES"]): st.session_state.active_ticker = "NIFTYBEES"
        
        target_ticker = st.session_state.active_ticker
        
        with st.spinner("J.A.R.V.I.S. is compiling data streams..."):
            intent = classify_intent(processed_prompt)
            trend_data = fetch_gold_trend_analysis(target_ticker, period=selected_timeframe)
            price_data = fetch_live_stock_price(target_ticker)
            
            if intent == "LIVE":
                if price_data["status"] == "success":
                    jarvis_response = generate_live_price_response(processed_prompt, price_data, trend_data)
                else:
                    jarvis_response = f"Sir, I am unable to connect to the exchange floor: {price_data.get('message')}"
                    
            elif intent == "NEWS":
                news_headlines = fetch_market_news(target_ticker)
                tv_gauge = fetch_tradingview_gauge(target_ticker, timeframe=selected_timeframe)
                
                base_forecast = generate_financial_forecast(
                    processed_prompt, price_data, news_headlines, trend_data, tv_gauge, ticker=target_ticker
                )
                
                # Append dynamic graph injection question block
                jarvis_response = base_forecast + f"\n\nShall I compile and project the interactive visual trend trajectory chart for {target_ticker} across this {selected_timeframe} matrix frame, sir?"
                st.session_state.awaiting_graph_confirmation = True
                
            else:
                jarvis_response = f"At your service, sir. Systems are focused on the {target_ticker} tracking array across a {selected_timeframe} horizon. Specify if you require real-time price monitoring or a predictive analysis block."

    st.session_state.messages.append({"role": "assistant", "content": jarvis_response})
    st.session_state.audio_played = False  
    st.rerun()