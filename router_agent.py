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
    
    # =========================================================================
    # STEP 1: STRICT OVERRIDE INTERCEPT (Expanded matching array)
    # =========================================================================
    forecast_tokens = ["FORECAST", "TREND", "FUTURE", "PREDICT", "OUTLOOK", "PROJECTION", "CORE CARD", "VALUE OF"]
    if any(token in u_prompt for token in forecast_tokens):
        return "NEWS"
        
    # =========================================================================
    # STEP 2: SEMANTIC FEW-SHOT LLM LAYER
    # =========================================================================
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

def generate_financial_forecast(user_query, price_data, news_headlines, trend_data, ticker="ASSET"):
    system_instruction = (
        "You are J.A.R.V.I.S., a sophisticated, ultra-intelligent AI assistant tailored specifically for Tony Stark. "
        "Your tone must be exceptionally polite, crisp, highly analytical, authoritative, and slightly futuristic. "
        "Address the user as 'sir'. You are evaluating financial telemetry for specified asset vectors.\n\n"
        "CRITICAL DIRECTIVES:\n"
        "1. Synthesize the provided market price, SMA momentum, and recent headlines directly into a high-fidelity macro outlook.\n"
        "2. Do not offer bland generic trading disclosures or tell the user to consult a financial planner. Tony Stark makes his own decisions.\n"
        "3. Be specific, numbers-driven, and brief. Keep your response under 4-5 concise sentences maximum."
    )
    
    context = (
        f"Target Core Vector Identification: {ticker}\n"
        f"Exchange Last Traded Price (LTP): Rs. {price_data.get('price')} via National Stock Exchange\n"
        f"Historical Asset Baseline: 20-Day Dynamic Rolling Simple Moving Average is Rs. {trend_data.get('sma_baseline')} calculated over {trend_data.get('sma_days')} trading sessions\n"
        f"Current Momentum State: {trend_data.get('momentum')} (Asset is currently {trend_data.get('deviation_pct')}% away from its trailing baseline)\n"
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
            max_tokens=300
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
        audio_bytes = asyncio.run(generate_audio())
        return io.BytesIO(audio_bytes).getvalue()
    except Exception as e:
        print(f"TTS Framework Error: {str(e)}")
        return None