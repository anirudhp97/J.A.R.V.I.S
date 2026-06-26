import streamlit as st
import io
from pydub import AudioSegment
from data_fetcher import fetch_live_stock_price, fetch_market_news, fetch_gold_trend_analysis
from router_agent import classify_intent, generate_financial_forecast, get_tts_bytes, transcribe_audio_with_groq

# 1. Page Config and Advanced Mark 42 Armor Hybrid Custom CSS Style Block
st.set_page_config(page_title="J.A.R.V.I.S. CORE HUD", page_icon="🦾", layout="centered")

st.markdown("""
    <style>
        @keyframes reactorPulse {
            0% {
                background-image: 
                    radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%),
                    radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%);
            }
            50% {
                background-image: 
                    radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.22) 0%, rgba(11, 2, 2, 0) 55%),
                    radial-gradient(circle at 50% 30%, #2a0505 0%, #0b0202 70%);
            }
            100% {
                background-image: 
                    radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.12) 0%, rgba(11, 2, 2, 0) 45%),
                    radial-gradient(circle at 50% 30%, #200404 0%, #0b0202 70%);
            }
        }

        .stApp {
            background-color: #0b0202;
            color: #F7F1E3;
            font-family: 'Courier New', Courier, monospace;
            animation: reactorPulse 6s infinite ease-in-out;
            background-attachment: fixed;
        }
        
        h1 {
            color: #00E5FF !important;
            text-shadow: 0 0 10px #00E5FF, 0 0 20px #aa0505;
            letter-spacing: 4px;
            font-weight: 900;
            text-align: center;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1a0505 !important;
            border-right: 3px solid #b97d10 !important;
            box-shadow: 5px 0px 15px rgba(185, 125, 16, 0.2);
        }
        
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
    </style>
""", unsafe_allow_html=True)

st.title("J.A.R.V.I.S.")
st.caption("System operational. All modules online. Awaiting directives, sir.")

# Analytics Control Console on Sidebar
st.sidebar.markdown("<h3 style='color:#b97d10; text-shadow: 0 0 5px #b97d10;'>🛡️ ARMOR DIAGNOSTICS</h3>", unsafe_allow_html=True)

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

# 2. Session State Initialization for Chat History & Context Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello sir, how may I help you today? Systems are fully functional. Tap the console input below to scan market metrics or generate a tactical telemetry forecast."
        }
    ]
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = selected_ticker
if "last_valid_prompt" not in st.session_state:
    st.session_state.last_valid_prompt = None
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False

st.sidebar.markdown(f"""
<div style='border: 1px solid #b97d10; padding: 12px; border-radius: 4px; background-color: rgba(170,5,5,0.15); margin-top: 25px;'>
    <span style='color: #00E5FF; font-size: 11px; font-weight: bold; letter-spacing: 1px;'>HUD CHANNEL FEED</span><br>
    <span style='color: #4AF2A1; font-size: 12px;'>● ACTIVE VECT: {st.session_state.active_ticker}</span><br>
    <span style='color: #FCE154; font-size: 10px;'>CORE STABILITY: LOCKED</span>
</div>
""", unsafe_allow_html=True)

# 3. Render Historical Chat Log using custom character identity tags
for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        prefix = "T. STARK [COM-LINK]:"
        text_color = "#FCC200" 
    else:
        prefix = "J.A.R.V.I.S.:"
        text_color = "#00E5FF" 
        
    with st.chat_message(message["role"]):
        st.markdown(f"<span style='color:{text_color}; font-weight:bold;'>{prefix}</span> <span style='color:#E2F1FF;'>{message['content']}</span>", unsafe_allow_html=True)
        
        if message["role"] == "assistant" and (index == len(st.session_state.messages) - 1) and not st.session_state.audio_played:
            with st.spinner("Initializing vocal transmission channels..."):
                audio_bytes = get_tts_bytes(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                    st.session_state.audio_played = True

st.divider()

# 4. Input Controls
if "mic_rotation_counter" not in st.session_state:
    st.session_state.mic_rotation_counter = 0

widget_key = f"voice_recorder_v_{st.session_state.mic_rotation_counter}_{selected_ticker}"
audio_file = st.audio_input("🎙️ ACTIVATE VOCAL RECEIVER", key=widget_key)
user_text_input = st.chat_input("Input terminal override command here...")

processed_prompt = None

# 5. Pipeline Phase 1: Input Audio Capture & Transcoded Groq Whisper STT Processing
if audio_file:
    with st.spinner("Decoding vocal sequence arrays via Groq Whisper..."):
        try:
            audio_bytes_raw = audio_file.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_raw))
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            detected_text = transcribe_audio_with_groq(wav_buffer)
            if detected_text:
                processed_prompt = detected_text
            else:
                st.error("Could not compile audio transmission stream.")
            st.session_state.mic_rotation_counter += 1
        except Exception as e:
            st.error(f"Core communication breakdown: {str(e)}.")
            st.session_state.mic_rotation_counter += 1

elif user_text_input:
    processed_prompt = user_text_input

# 6. Pipeline Phase 2: Agentic Routing and Context-Aware Data Aggregation
if processed_prompt:
    clean_prompt = processed_prompt.upper().strip()
    
    # Check if this is a follow-up confirmation (e.g., "YES", "YEAH", "PLEASE", "DO IT")
    is_confirmation = any(clean_prompt.startswith(word) for word in ["YES", "YEA", "OK", "PLEASE", "SURE", "GO AHEAD"])
    
    # If we are waiting for a follow-up and the user says "yes", restore the last historical query
    if st.session_state.awaiting_followup and is_confirmation and st.session_state.last_valid_prompt:
        routing_prompt = st.session_state.last_valid_prompt
    else:
        routing_prompt = processed_prompt
        # Keep track of this prompt if it looks like a genuine data tracking intent
        if any(w in clean_prompt for w in ["GOLD", "SILVER", "NIFTY", "BEES"]):
            st.session_state.last_valid_prompt = processed_prompt

    clean_routing = routing_prompt.upper()

    # Dynamically update ticker session state
    if any(w in clean_routing for w in ["GOLD BEES", "GOLD BEAST", "GOLD BEADS", "GOLDBEES", "PRICE OF GOLD"]):
        st.session_state.active_ticker = "GOLDBEES"
    elif any(w in clean_routing for w in ["SILVER BEES", "SILVER BEAST", "SILVER BEADS", "SILVERBEES"]):
        st.session_state.active_ticker = "SILVERBEES"
    elif any(w in clean_routing for w in ["NIFTY BEES", "NIFTYBEES"]):
        st.session_state.active_ticker = "NIFTYBEES"

    target_ticker = st.session_state.active_ticker
    st.session_state.messages.append({"role": "user", "content": processed_prompt})
    
    with st.spinner("J.A.R.V.I.S. is compiling data streams..."):
        # If the user confirmed, force an analytical 'NEWS' routing context, otherwise classify standard
        if st.session_state.awaiting_followup and is_confirmation:
            intent = "NEWS"
        else:
            intent = classify_intent(routing_prompt)
        
        trend_data = fetch_gold_trend_analysis(target_ticker, period=selected_timeframe)
        price_data = fetch_live_stock_price(target_ticker)
        
        if intent == "LIVE":
            st.session_state.awaiting_followup = False
            if price_data["status"] == "success":
                jarvis_response = (
                    f"Live data retrieved, sir. {price_data['company']} is currently trading on the NSE at "
                    f"Rs. {price_data['price']}. Trajectory metrics for the {trend_data.get('lookback_period')} horizon "
                    f"indicate a structural {trend_data.get('momentum')} orientation."
                )
            else:
                jarvis_response = f"Sir, I am unable to connect to the exchange floor: {price_data.get('message')}"
                
        elif intent == "NEWS":
            st.session_state.awaiting_followup = False
            news_headlines = fetch_market_news(target_ticker)
            jarvis_response = generate_financial_forecast(routing_prompt, price_data, news_headlines, trend_data, ticker=target_ticker)
            
        else:
            # System hits default fallback. We arm the follow-up state flags.
            st.session_state.awaiting_followup = True
            jarvis_response = (
                f"At your service, sir. The system's diagnostic analytics are currently targeted at the {target_ticker} vector "
                f"across a {selected_timeframe} horizon. Specify if you would like me to extract live price telemetry or formulate a macro prediction."
            )
            
    st.session_state.messages.append({"role": "assistant", "content": jarvis_response})
    st.session_state.audio_played = False  
    st.rerun()