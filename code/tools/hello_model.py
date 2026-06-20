"""
hello_model.py  --  Your first model call, explained line by line.

This is the SIMPLEST possible version of what your whole project does 44 times.
Run it with:   python code/tools/hello_model.py
"""

# 1) We need to read your secret API key. It lives in code/.env (never shared).
#    python-dotenv loads that file so the program can see the key.
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")   # loads code/.env
api_key = os.environ.get("GEMINI_API_KEY")                  # read the key

if not api_key:
    print("No key found. Put GEMINI_API_KEY=... in code/.env")
    raise SystemExit

# 2) 'genai' is Google's toolkit (SDK) for talking to Gemini.
#    We create a 'client' = an open phone line, authenticated with your key.
from google import genai
client = genai.Client(api_key=api_key)

# 3) This is THE call. We hand the model:
#      - which model to use ("gemini-2.5-flash")
#      - what we want ("contents" = our question)
#    ...and it hands back an answer.
print("Calling the model... (this takes a few seconds)\n")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="In two simple sentences, explain what an API is to someone brand new to coding.",
    )
    # 4) The answer text is in response.text. Print it.
    print("THE MODEL SAID:\n")
    print(response.text)

except Exception as e:
    # If you've used your 20 free calls for the day, you'll land here.
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        print("Rate limit hit: you've used your free calls for today.")
        print("This is EXACTLY the limit we engineered around in the main project.")
        print("It resets daily — try again later, or use the no-cost demos below.")
    else:
        print("Something went wrong:\n", msg[:300])
