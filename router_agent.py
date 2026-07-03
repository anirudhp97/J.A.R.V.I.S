import os
import io
import asyncio

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

API_KEY = os.getenv("GROQ_API_KEY")
client = None
if OpenAI is not None and API_KEY:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=API_KEY
    )

def transcribe_audio_with_groq(wav_io_buffer, language="English"):
    if client is None:
        return None

    try:
        wav_io_buffer.name = "audio.wav"

        # Guide Whisper's spelling matrix based on the user's language setting
        whisper_prompt = "GOLDBEES, SILVERBEES, NIFTYBEES, stock price, market trend"
        if language == "Kannada":
            whisper_prompt = "ಚಿನ್ನದ ಬೆಲೆ, ಬೆಳ್ಳಿ ಬೆಲೆ, ನಿಫ್ಟಿ, GOLDBEES, SILVERBEES, NIFTYBEES, ಮಾರ್ಕೆಟ್ ಟ್ರೆಂಡ್"

        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=wav_io_buffer,
            prompt=whisper_prompt,
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        print(f"Groq Whisper STT Error: {str(e)}")
        return None

def classify_intent(user_prompt):
    u_prompt = user_prompt.upper()
    forecast_tokens = ["FORECAST", "TREND", "FUTURE", "PREDICT", "OUTLOOK", "PROJECTION", "CORE CARD", "VALUE OF", "ಮುನ್ಸೂಚನೆ", "ಟ್ರೆಂಡ್"]
    if any(token in u_prompt for token in forecast_tokens):
        return "NEWS"

    if client is None:
        return "UNKNOWN"

    """
    Two-Step Verification Intent Router.
    Enhanced with fuzzy phonetic structural matching.
    """
    system_instruction = (
        "You are a strict financial assistant routing engine. Your absolute sole responsibility is to "
        "classify the user's intent into exactly ONE of the following uppercase words: LIVE, NEWS, or UNKNOWN.\n\n"
        "CLASSIFICATION RULES:\n"
        "- LIVE: The user specifically wants real-time raw price feeds, ticker metrics or current costs right now.\n"
        "- NEWS: The user is explicitly or implicitly asking for an outlook, future trends, general forecasts, or market summaries.\n"
        "- UNKNOWN: Casual conversation, single greetings, simple responses, or completely unrelated commands.\n\n"
        "FEW-SHOT TRAINING EXAMPLES:\n"
        "Input: 'can you check the latest gold bees price' -> Response: LIVE\n"
        "Input: 'how much is silver bees right now' -> Response: LIVE\n"
        "Input: 'give me an analysis on the trend' -> Response: NEWS\n"
        "Input: 'What is the latest value' -> Response: NEWS\n"
        "Input: 'yes please' -> Response: UNKNOWN\n\n"
        "CRITICAL: Output ONLY the raw uppercase word: LIVE, NEWS, or UNKNOWN. Do not add punctuation."
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Input text to route: '{user_prompt}'"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().upper()
        intent = "".join([char for char in intent if char.isalnum()])
        
        if intent in ["LIVE", "NEWS", "UNKNOWN"]:
            return intent
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def generate_live_price_response(user_query, price_data, trend_data, language="English"):
    """
    Low-latency presentation tier utilizing a smaller model to maximize speed
    when delivering real-time exchange ticker variables.
    """
    if client is None:
        return "Live telemetry is currently unavailable because the AI service is not configured."
    system_instruction = (
        "You are J.A.R.V.I.S., a high-performance AI assistant for Tony Stark. "
        "Your tone must be crisp, polite, futuristic, and clear. Address the user as 'sir'. "
        "Clearly provide the live exchange price details and its basic directional orientation. "
        "Keep your output limited to 1 or 2 clear sentences maximum. "
    )
    
    if language == "Kannada":
        system_instruction += "CRITICAL: You must provide your entire response in clear, fluent Kannada script."
    else:
        system_instruction += "Response must be in English."
    
    context = (
        f"Asset Vector Identifier: {price_data.get('ticker')}\n"
        f"Target Enterprise: {price_data.get('company')}\n"
        f"Live Exchange Value (LTP): Rs. {price_data.get('price')}\n"
        f"Momentum Classification: {trend_data.get('momentum')}\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context Metrics:\n{context}\n\nUser Voice Comm: {user_query}"}
            ],
            temperature=0.3,
            max_tokens=150 if language == "Kannada" else 80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Live telemetry generation bypassed: {str(e)}"

def generate_financial_forecast(user_query, price_data, news_headlines, trend_data, tv_gauge, ticker="ASSET", language="English"):
    if client is None:
        return "Analytical forecasting is currently unavailable because the AI service is not configured."

    system_instruction = (
        "You are J.A.R.V.I.S., a sophisticated, ultra-intelligent AI assistant tailored specifically for Tony Stark. "
        "Your tone must be exceptionally polite, crisp, highly analytical, authoritative, and slightly futuristic. "
        "Address the user as 'sir'. You are evaluating financial telemetry for specified asset vectors.\n\n"
        "CRITICAL DIRECTIVES:\n"
        "1. Synthesize market prices, SMA momentum deviations, multi-indicator TradingView consensus data, and recent headlines directly into a high-fidelity macro outlook.\n"
        "2. Balance structural news narratives against the hard mathematical oscillator summary from TradingView to eliminate directional blindspots.\n"
        "3. Do not offer bland generic trading disclosures or tell the user to consult a financial planner. Tony Stark makes his own decisions.\n"
        "4. Be specific, numbers-driven, and brief. Keep your response under 4-5 concise sentences maximum."
    )
    
    if language == "Kannada":
        system_instruction += "\n5. CRITICAL: Translate and generate this full synthesis analysis purely in elegant, formal Kannada script."
    else:
        system_instruction += "\n5. Response must be in English."
    
    if tv_gauge and tv_gauge.get("status") == "success":
        tv_telemetry = (
            f"TradingView Indicator Summary: {tv_gauge.get('recommendation')}\n"
            f"Aggregate Oscillators & MAs Breakdown -> BUY: {tv_gauge.get('buy_signals')}, SELL: {tv_gauge.get('sell_signals')}, NEUTRAL: {tv_gauge.get('neutral_signals')}\n"
        )
    else:
        tv_telemetry = "TradingView Indicator Summary: Data Connection Stream Unavailable.\n"

    context = (
        f"Target Core Vector Identification: {ticker}\n"
        f"Exchange Last Traded Price (LTP): Rs. {price_data.get('price')} via National Stock Exchange\n"
        f"Historical Asset Baseline: 20-Day Dynamic Rolling Simple Moving Average is Rs. {trend_data.get('sma_baseline')} calculated over {trend_data.get('sma_days')} trading sessions\n"
        f"Current Momentum State: {trend_data.get('momentum')} (Asset is currently {trend_data.get('deviation_pct')}% away from its trailing baseline)\n"
        f"{tv_telemetry}"
        f"Recent Market News Context:\n{news_headlines}\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context Metrics:\n{context}\n\nUser Predictive Request: {user_query}"}
            ],
            temperature=0.5,
            max_tokens=450 if language == "Kannada" else 300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Analytical reasoning layer timed out during cloud synthesis: {str(e)}"

def get_tts_bytes(text, language="English"):
    if edge_tts is None:
        return None

    # Dynamic toggle switches voice models between English (Ryan) and Kannada (Gagan)
    voice_profile = "kn-IN-GaganNeural" if language == "Kannada" else "en-GB-RyanNeural"

    async def generate_audio():
        communicate = edge_tts.Communicate(text, voice_profile)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    try:
        audio_bytes = asyncio.run(generate_audio())
        return io.BytesIO(audio_bytes).getvalue()
    except Exception as e:
        print(f"TTS Framework Error: {str(e)}")
        return None