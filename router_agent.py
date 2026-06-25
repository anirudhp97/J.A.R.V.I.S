import os
import io
from openai import OpenAI
from gtts import gTTS

# Initialize the cloud client using environment variables/Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") 
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # High-speed hosted inference engine
    api_key=API_KEY
)

def transcribe_audio_with_groq(wav_io_buffer):
    """
    Sends transcoded in-memory WAV audio bytes to Groq's 
    Whisper Large V3 cloud architecture for precise transcription.
    """
    try:
        # Give the buffer object a name property so the SDK wrapper accepts it cleanly
        wav_io_buffer.name = "audio.wav"
        
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=wav_io_buffer,
            prompt="GOLDBEES, SILVERBEES, NIFTYBEES, stock price, market trend",  # Domain-specific prompt context string
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        print(f"Groq Whisper STT Error: {str(e)}")
        return None

def classify_intent(user_prompt):
    """
    Hardened intent routing with expanded financial phonetic matchers.
    """
    u_prompt = user_prompt.upper()
    
    # Catching phonetic slips like "life price" or "what is the gold bees..."
    if any(w in u_prompt for w in ["PRICE", "LIVE", "LIFE", "LTP", "COST", "TRADING", "CURRENT", "VALUE", "HOW MUCH"]):
        return "LIVE"
    if any(w in u_prompt for w in ["NEWS", "FORECAST", "PREDICT", "TREND", "FUTURE", "ANALYZE", "OUTLOOK", "PROJECTION"]):
        return "NEWS"

    system_instruction = (
        "You are a strict financial assistant router engine. Your job is to classify the user's intent "
        "into exactly ONE of the following uppercase words: LIVE, NEWS, or UNKNOWN.\n"
        "LIVE - If asking for current/latest/today's price or value.\n"
        "NEWS - If asking for updates, trends, or forecasts.\n"
        "UNKNOWN - General conversations."
    )
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().upper()
        if intent in ["LIVE", "NEWS", "UNKNOWN"]:
            return intent
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def generate_financial_forecast(user_query, price_data, news_headlines, trend_data, ticker="ASSET"):
    """
    Synthesizes real-time metrics and dynamic news contexts via Llama 3 70B to output a cohesive prediction blueprint.
    """
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
            model="llama3-70b-8192", 
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
    """
    Cloud-safe, 100% free Text-To-Speech using Google TTS with an elegant British voice profile.
    """
    try:
        tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"TTS Framework Error: {str(e)}")
        return None