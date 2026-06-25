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
    Robust Semantic Intent Router using Few-Shot LLM Prompting.
    Eliminates fragile hardcoded keyword matching arrays completely.
    """
    system_instruction = (
        "You are a strict financial assistant routing engine. Your absolute sole responsibility is to "
        "classify the user's intent into exactly ONE of the following uppercase words: LIVE, NEWS, or UNKNOWN.\n\n"
        "CLASSIFICATION RULES:\n"
        "- LIVE: The user wants real-time numbers, ticker costs, or current valuation. Even if they include words "
        "like 'forecast' or 'future', if the primary focus is checking a price right now, it is LIVE.\n"
        "- NEWS: The user is explicitly asking for a prediction, macro trend, trajectory outlook, headline summary, "
        "or a forward-looking analysis (e.g., 'next 5 days', 'future target').\n"
        "- UNKNOWN: Casual filler text, greetings, or completely unrelated commands.\n\n"
        "FEW-SHOT TRAINING EXAMPLES (LEARN FROM THESE):\n"
        "Input: 'what is the live price of gold bees' -> Response: LIVE\n"
        "Input: 'what is the life price of gold bees' -> Response: LIVE\n"
        "Input: 'how much is silver bees trading for right now' -> Response: LIVE\n"
        "Input: 'What is the forecast you see for the gold bees price in the next 5 days?' -> Response: NEWS\n"
        "Input: 'Can you analyze the future trend of niftybees?' -> Response: NEWS\n"
        "Input: 'give me a target projection for silver bees' -> Response: NEWS\n"
        "Input: 'Hey Jarvis, hope you are doing good morning' -> Response: UNKNOWN\n\n"
        "CRITICAL: You must output ONLY the single word (LIVE, NEWS, or UNKNOWN). Do not include any punctuation, "
        "conversational padding, or explanations. If you fail this, system components will crash."
    )
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Input text to route: '{user_prompt}'"}
            ],
            temperature=0.0, # Complete deterministic execution
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().upper()
        
        # Clean any accidental trailing punctuation from the LLM output
        intent = "".join([char for char in intent if char.isalnum()])
        
        if intent in ["LIVE", "NEWS", "UNKNOWN"]:
            return intent
            
        # Hard deterministic fallback only if the cloud client fails to follow structural bounds
        if "FORECAST" in user_prompt.upper() or "TREND" in user_prompt.upper():
            return "NEWS"
        return "LIVE"
        
    except Exception:
        # Code level exception safety valve
        if "FORECAST" in user_prompt.upper() or "TREND" in user_prompt.upper():
            return "NEWS"
        return "LIVE"

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