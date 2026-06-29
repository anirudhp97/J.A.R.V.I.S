import streamlit as st
import streamlit.components.v1 as components
import io
import re
import pandas as pd  # Added missing pandas import
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
# 1. Page Config and Advanced Mark 42 Armor Custom CSS
# ===================================================
st.set_page_config(page_title="J.A.R.V.I.S. CORE HUD", layout="centered")

st.markdown("""
    <style>
        @keyframes reactorPulse {
            0% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%), radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%); }
            50% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.22) 0%, rgba(11, 2, 2, 0) 55%), radial-gradient(circle at 50% 30%, #2a0505 0%, #0b0202 70%); }
            100% { background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%), radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%); }
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0202 !important;
            animation: reactorPulse 8s infinite ease-in-out;
            color: #ff4444 !important;
            font-family: 'Courier New', Courier, monospace;
        }
        
        .stChatMessage {
            background-color: rgba(25, 5, 5, 0.65) !important;
            border: 1px solid #ff1111 !important;
            box-shadow: 0 0 12px rgba(255, 17, 17, 0.2);
            border-radius: 4px !important;
            margin-bottom: 12px;
        }
        
        h1, h2, h3, p, span, li {
            color: #ff5555 !important;
            text-shadow: 0 0 4px rgba(255, 68, 68, 0.4);
        }
        
        div[data-testid="stChatInput"] {
            border: 1px solid #00e5ff !important;
            background-color: #100202 !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
        }
        
        .stSpinner > div {
            border-top-color: #00e5ff !important;
        }

        /* ONLY ADDED BELOW TO TARGET TRADINGVIEW TEXT INSIDE IFRAMES WITHOUT CHANGING THE REST OF THE HUD */
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

st.title("🤖 J.A.R.V.I.S. MARK XLII")
st.subheader("TACTICAL MARKET INTELLIGENCE TELEMETRY HUD")

# ===================================================
# 2. State Vector Initialization
# ===================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "awaiting_graph_confirmation" not in st.session_state:
    st.session_state.awaiting_graph_confirmation = False
if "staged_trend_data" not in st.session_state:
    st.session_state.staged_trend_data = None
if "target_ticker" not in st.session_state:
    st.session_state.target_ticker = "GOLDBEES"

# ===================================================
# 3. Micro-HUD Peripheral Sidebar Configuration
# ===================================================
with st.sidebar:
    st.markdown("### 🛠️ CORE SYSTEMS STATUS")
    st.success("ROUTING MATRIX: ACTIVE")
    st.success("GROQ ANALYTICAL BEAM: CONNECTED")
    
    st.markdown("---")
    ticker_input = st.text_input("VECTOR TARGET TRACKER:", value=st.session_state.target_ticker).upper().strip()
    if ticker_input != st.session_state.target_ticker:
        st.session_state.target_ticker = ticker_input
        
    selected_timeframe = st.selectbox("HORIZON MATRIX WINDOW:", ["5d", "1mo", "3mo", "1y"], index=1)

# ===================================================
# 4. Voice Input Interface Component
# ===================================================
st.markdown("### 🎙️ VOCAL COMM-LINK ANCHOR")
audio_data_received = components.html("""
    <div style="display: flex; gap: 10px; align-items: center; justify-content: center; background: rgba(30,5,5,0.4); padding: 10px; border-radius: 6px; border: 1px dashed #ff3333;">
        <button id=\"startRec\" style=\"background: #cc0000; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold;\">OPEN COMMS</button>
        <button id=\"stopRec\" style=\"background: #444; color: #aaa; border: none; padding: 8px 16px; border-radius: 4px; cursor: not-allowed; font-weight: bold;\" disabled>CLOSE COMMS</button>
        <div id=\"status\" style=\"color: #00e5ff; font-family: monospace; font-size: 12px;\">COMMS IDLE</div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        const startBtn = document.getElementById('startRec');
        const stopBtn = document.getElementById('stopRec');
        const statusDiv = document.getElementById('status');

        startBtn.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    window.parent.postMessage({ type: 'AUDIO_TRANSMISSION', data: reader.result }, '*');
                };
            };
            
            mediaRecorder.start();
            startBtn.disabled = true; startBtn.style.background = '#333';
            stopBtn.disabled = false; stopBtn.style.background = '#00e5ff'; stopBtn.style.color = '#000';
            statusDiv.innerText = "🎙️ STREAMING VOCAL CAPTURE...";
        };

        stopBtn.onclick = () => {
            mediaRecorder.stop();
            startBtn.disabled = false; startBtn.style.background = '#cc0000';
            stopBtn.disabled = true; stopBtn.style.background = '#444'; stopBtn.style.color = '#aaa';
            statusDiv.innerText = "PROCESSING METRIC BUNDLE...";
        };
    </script>
""", height=80)

# ===================================================
# 5. Historic Message HUD Rendering Loop
# ===================================================
for index, msg in enumerate(st.session_state.messages):
    message_payload = msg.get("content", "")
    role = msg.get("role", "user")
    
    with st.chat_message(role):
        if "DATASTREAM_START" in message_payload:
            display_text = re.sub(r"DATASTREAM_START\n.*?\nDATASTREAM_END", "", message_payload, flags=re.DOTALL).strip()
        else:
            display_text = message_payload
            
        st.write(display_text)
        
        forecast_df = parse_llm_response_for_forecast(message_payload)
        if forecast_df is not None:
            st.markdown("### 📊 PROJECTED HORIZON DATASTREAM:")
            st.line_chart(forecast_df)
            st.dataframe(forecast_df, use_container_width=True)

        if role == "assistant" and not st.session_state.audio_played and index == len(st.session_state.messages) - 1:
            with st.spinner("Synthesizing audio readout..."):
                tts_bytes = get_tts_bytes(display_text)
                if tts_bytes:
                    st.audio(tts_bytes, format="audio/mp3", autoplay=True)
            st.session_state.audio_played = True

# ===================================================
# 6. Incoming Text Prompt & Audio Payload Resolution
# ===================================================
user_prompt = st.chat_input("Input vector command command or transmit audio uplink, sir...")

if user_prompt:
    processed_prompt = user_prompt.upper().strip()
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Synchronizing satellite array arrays..."):
            
            # Route confirmation states safely via textual responses
            if st.session_state.get("awaiting_graph_confirmation", False):
                if "YES" in processed_prompt:
                    jarvis_response = (
                        f"Acknowledged, sir. Compiling the visual trajectory projection metrics for {st.session_state.target_ticker}.\n\n"
                        "Here is the projected datastream sequence for the upcoming session windows:\n\n"
                    )
                    
                    generation_prompt = (
                        f"Generate a short 4-5 business day forecast response for {st.session_state.target_ticker} "
                        "concluding with the mandatory DATASTREAM_START and DATASTREAM_END block containing dates and projected target prices."
                    )
                    
                    news_headlines = fetch_market_news(st.session_state.target_ticker)
                    tv_gauge = fetch_tradingview_gauge(st.session_state.target_ticker)
                    
                    jarvis_response += generate_financial_forecast(
                        generation_prompt, 
                        fetch_live_stock_price(st.session_state.target_ticker), 
                        news_headlines, 
                        st.session_state.staged_trend_data, 
                        tv_gauge, 
                        ticker=st.session_state.target_ticker
                    )
                    st.session_state.awaiting_graph_confirmation = False
                    
                else:
                    jarvis_response = "Understood, sir. Trajectory plotting stands down. Standing by for further monitoring requests."
                    st.session_state.awaiting_graph_confirmation = False
                    
            else:
                # Default Intent Routing Blueprint
                intent = classify_intent(user_prompt)
                target_ticker = st.session_state.target_ticker
                
                price_data = fetch_live_stock_price(target_ticker)
                trend_data = fetch_gold_trend_analysis(target_ticker, period=selected_timeframe)
                
                if intent == "PRICE":
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
                    
                    jarvis_response = base_forecast + f"\n\nShall I project the dynamic visual trend trajectory forecast chart for {target_ticker} across the upcoming trading sessions, sir?"
                    
                    st.session_state.staged_trend_data = trend_data
                    st.session_state.awaiting_graph_confirmation = True
                    
                else:
                    jarvis_response = f"At your service, sir. Systems are focused on the {target_ticker} tracking array. Specify if you require real-time price monitoring or a predictive analysis block."

    st.session_state.messages.append({"role": "assistant", "content": jarvis_response})
    st.session_state.audio_played = False  
    st.rerun()