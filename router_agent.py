import os
import io
import asyncio
import edge_tts
from openai import OpenAI

API_KEY = os.getenv("GROQ_API_KEY") 
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)

def transcribe_audio_with_groq(wav_io_buffer):
    try:
        wav_io_buffer.name = "audio.wav"
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=wav_io_buffer,
            prompt="GOLDBEES, SILVERBEES, NIFTYBEES, stock price, market trend",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        print(f"Groq Whisper STT Error: {str(e)}")
        return None

def classify_intent(user_prompt):
    """
    Two-Step Verification Intent Router.
    Enhanced with fuzzy phonetic structural matching.
    """
    u_prompt = user_prompt.upper()
    
    forecast_tokens = ["FORECAST", "TREND", "FUTURE", "PREDICT", "OUTLOOK", "PROJECTION", "CORE CARD", "VALUE OF"]
    if any(token in u_prompt for token in forecast_tokens):
        return "NEWS"
        
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

def generate_live_price_response(user_query, price_data, trend_data):
    """
    Low-latency presentation tier utilizing a smaller model to maximize speed
    when delivering real-time exchange ticker variables.
    """
    system_instruction = (
        "You are J.A.R.V.I.S., a high-performance AI assistant for Tony Stark. "
        "Your tone must be crisp, polite, futuristic, and clear. Address the user as 'sir'. "
        "Clearly provide the live exchange price details and its basic directional orientation. "
        "Keep your output limited to 1 or 2 clear sentences maximum."
    )
    
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
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Live telemetry generation bypassed: {str(e)}"

def generate_financial_forecast(user_query, price_data, news_headlines, trend_data, tv_gauge, ticker="ASSET"):
    system_instruction = (
        "You are J.A.R.V.I.S., a sophisticated, ultra-intelligent AI assistant tailored specifically for Tony Stark. "
        "Your tone must be exceptionally polite, crisp, highly analytical, authoritative, and slightly futuristic. "
        "Address the user as 'sir'. You are evaluating financial telemetry for specified asset vectors.\n\n"
        "CRITICAL DIRECTIVES:\n"
        "1. Synthesize market prices, SMA momentum deviations, multi-indicator TradingView consensus data, and recent headlines directly into a high-fidelity macro outlook.\n"
        "2. Avoid linear prediction bias. You MUST present a balanced, two-sided tactical forecast:\n"
        "   - Outlining the BULLISH boundary case (upward target levels if current momentum sustains or reverses upward).\n"
        "   - Outlining the BEARISH boundary case (downward target support levels if momentum corrects or encounters resistance).\n"
        "   - Specifying a clear, calculated expected price range (e.g., 'Rs. X to Rs. Y') for the upcoming sessions to account for market volatility.\n"
        "3. Balance structural news narratives against the hard mathematical oscillator summary from TradingView to eliminate directional blindspots.\n"
        "4. Do not offer bland generic trading disclosures or tell the user to consult a financial planner. Tony Stark makes his own decisions.\n"
        "5. Be specific, numbers-driven, and brief. Keep your response under 5-6 concise sentences maximum.\n"
        "6. DATASTREAM REQUIREMENT: At the very end of your response, output a structured table block for 5 upcoming trading days formatted EXACTLY like this:\n"
        "DATASTREAM_START\n"
        "Date | Projected Target\n"
        "YYYY-MM-DD | <numeric_value>\n"
        "YYYY-MM-DD | <numeric_value>\n"
        "YYYY-MM-DD | <numeric_value>\n"
        "YYYY-MM-DD | <numeric_value>\n"
        "YYYY-MM-DD | <numeric_value>\n"
        "DATASTREAM_END"
    )
    
    if tv_gauge and tv_gauge.get("status") == "success":
        tv_telemetry = (
            f"TradingView Indicator Summary: {tv_gauge.get('recommendation')}\n"
            f"Aggregate Oscillators & MAs Breakdown -> BUY: {tv_gauge.get('buy_signals')}, SELL: {tv_gauge.get('sell_signals')}, NEUTRAL: {tv_gauge.get('neutral_signals')}\n"
        )
    else:
        tv_telemetry = "TradingView Indicator Summary: Data Connection Stream Unavailable.\n"

    current_price = price_data.get('price')
    sma_baseline = trend_data.get('sma_baseline')
    deviation_pct = trend_data.get('deviation_pct', 0)
    
    approx_daily_vol = 0.015
    projected_upper = round(current_price * (1 + (approx_daily_vol * 2.23)), 2) if current_price else "N/A"
    projected_lower = round(current_price * (1 - (approx_daily_vol * 2.23)), 2) if current_price else "N/A"

    context = (
        f"Target Core Vector Identification: {ticker}\n"
        f"Exchange Last Traded Price (LTP): Rs. {current_price} via National Stock Exchange\n"
        f"Historical Asset Baseline: 20-Day Dynamic Rolling Simple Moving Average is Rs. {sma_baseline} calculated over {trend_data.get('sma_days')} trading sessions\n"
        f"Current Momentum State: {trend_data.get('momentum')} (Asset is currently {deviation_pct}% away from its trailing baseline)\n"
        f"Rough Boundary Approximations (1-Week Horizon): Bullish Target Cap ~ Rs. {projected_upper} | Bearish Support Floor ~ Rs. {projected_lower}\n"
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
            max_tokens=450
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Analytical reasoning layer timed out during cloud synthesis: {str(e)}"

def get_tts_bytes(text):
    voice_profile = "en-GB-RyanNeural"
    async def generate_audio():
        communicate = edge_tts.Communicate(text, voice_profile)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(generate_audio())
        else:
            return asyncio.run(generate_audio())
    except Exception as e:
        print(f"TTS Framework Error: {str(e)}")
        return None