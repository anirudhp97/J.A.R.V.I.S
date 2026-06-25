import os
import io
from openai import OpenAI
from gtts import gTTS

# Initialize the cloud client using environment variables/Streamlit secrets
# For local testing, ensure you have set the GROQ_API_KEY environment variable.
# On Streamlit Cloud, add GROQ_API_KEY = "gsk_..." into your Advanced Secrets panel.
API_KEY = os.getenv("GROQ_API_KEY") 
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Free high-speed hosted inference cloud engine
    api_key=API_KEY
)

def classify_intent(user_prompt):
    """
    Cloud-safe intent routing using Groq's high-speed hosted Llama 3 model.
    """
    system_instruction = (
        "You are a strict financial assistant router engine. Your job is to classify the user's intent "
        "into exactly ONE of the following uppercase words:\n"
        "1. LIVE - If the user is asking for the current, live, today's, or latest price of a stock/gold/index.\n"
        "2. NEWS - If the user is asking for updates, market headlines, news, future forecasts, or trends.\n"
        "3. UNKNOWN - If the input is just general filler conversational text (like 'hello', 'good morning').\n\n"
        "CRITICAL: You must respond with only that single word (LIVE, NEWS, or UNKNOWN). Do not include any punctuation, "
        "explanations, conversational filler, or extra characters."
    )
    
    try:
        # Utilizing ultra-fast Llama3-8b for instantaneous classification routing
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        intent = response.choices[0].message.content.strip().upper()
        
        if intent in ["LIVE", "NEWS", "UNKNOWN"]:
            return intent
            
        # Hardcoded fallback logic in case of unpredictable LLM text output formatting
        u_prompt = user_prompt.upper()
        if any(keyword in u_prompt for keyword in ["FORECAST", "FUTURE", "NEWS", "TREND"]):
            return "NEWS"
        elif "PRICE" in u_prompt or "LIVE" in u_prompt:
            return "LIVE"
        return "UNKNOWN"
            
    except Exception as e:
        print(f"Cloud Routing Engine Error: {str(e)}")
        return "UNKNOWN"
    
def generate_financial_forecast(user_query, price_data, news_headlines, trend_data, ticker):
    """
    Assembles real-time and historical financial contexts and pushes execution 
    to a heavy-duty cloud intelligence model for future trend processing.
    """
    system_instruction = (
        "You are Jarvis, a brilliant, protective, and elite personal financial advisor. Your goal is to "
        "analyze current metrics against historical trends and news headlines to provide a predictive "
        "future forecast. Look closely at the percentage deviation and momentum: if the asset is far above "
        "its trend baseline, discuss potential short-term stabilization or continued bullish runs. "
        "Keep your forecast actionable, logical, and under 4 sentences. Structure it so it reads cleanly "
        "and authoritatively when spoken out loud by a text-to-speech voice engine."
    )
    
    context = (
        f"Target Core Asset Vector: {ticker}\n"
        f"Current Real-Time Price: Rs. {price_data.get('price')}\n"
        f"Historical Horizon Evaluated: {trend_data.get('lookback_period')}\n"
        f"Trailing Average Baseline: Rs. {trend_data.get('sma_baseline')} over {trend_data.get('sma_days')} trading sessions\n"
        f"Current Momentum State: {trend_data.get('momentum')} (Asset is currently {trend_data.get('deviation_pct')}% away from its trailing baseline)\n"
        f"Recent Market News Context:\n{news_headlines}\n"
    )
    
    try:
        # Using Llama3-70b for advanced data correlation, math evaluation, and reasoning
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
    Cloud-safe, 100% free Text-To-Speech using Google TTS.
    Requires zero API keys and works flawlessly on headless cloud servers.
    """
    try:
        # 'lang="en"' with 'tld="co.uk"' forces a clean British accent
        tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)

        # Save the audio stream directly into an in-memory bytes buffer
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        return fp
    except Exception as e:
        print(f"gTTS Cloud Exception: {str(e)}")
        return None