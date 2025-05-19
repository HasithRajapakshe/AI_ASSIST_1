import datetime
import webbrowser
import pyttsx3
import speech_recognition as sr
import time
import pyautogui
import subprocess
import os
import json
import pickle
import requests
from keras.models import load_model # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
import random
import numpy as np
import sys
import psutil
import wikipedia
from ollama import chat
from googlesearch import search

# --- CONFIGURATION ---
CONFIG = {
    "news_api_key": "1095f059a73f85f62c6ab4e08ae936df",  # <-- Your real GNews API key
    "weather_api_key": "YOUR_OPENWEATHER_API_KEY",        # <-- Replace with your real weather key if needed
    "wikipedia_lang": "en",
    "web_search_max_results": 3,
    "fallback_sources": ["knowledge_base", "wikipedia", "web_search","news_api", "weather_api"]
}

wikipedia.set_lang(CONFIG["wikipedia_lang"])

# --- LOAD INTENTS AND MODEL ---
try:
    with open("intents.json") as file:
        data = json.load(file)
except Exception as e:
    print(f"Error loading intents.json: {e}")
    sys.exit(1)

try:
    model = load_model("chat_model.h5")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
except Exception as e:
    print(f"Error loading model or preprocessing files: {e}")
    sys.exit(1)

# --- VOICE ENGINE ---
def initialize_engine():
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # Index 1 for female voice, 0 for male
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume+0.25)
    return engine

engine = initialize_engine() # Initialize engine once

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- COMMAND HANDLER ---
def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening......", end="", flush=True)
        r.adjust_for_ambient_noise(source, duration=0.5)
        r.pause_threshold = 0.8
        r.energy_threshold = 1000
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("\r" + " " * 20 + "\r", end="", flush=True)
            print("Recognizing.......", end="", flush=True)
            query = r.recognize_google(audio, language='en-US')
            print("\r" + " " * 20 + "\r", end="", flush=True)
            print(f"User said: {query}\n")
            return query
        except sr.WaitTimeoutError:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            print("No speech detected. Listening again...")
            return None
        except sr.UnknownValueError:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            speak("Sorry, I didn't catch that. Could you say it again?")
            return None
        except sr.RequestError as e:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            speak("Could not request results from Google Speech Recognition service.")
            print(f"SERVICE ERROR; {e}")
            return None
        except Exception as e:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            speak("An error occurred during speech recognition. Please try again.")
            print(f"RECOGNITION ERROR; {e}")
            return None

# --- SYSTEM UTILITIES ---
def cal_day():
    day_of_week = datetime.datetime.today().weekday()
    day_dict = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    return day_dict.get(day_of_week, "Unknown")

def wishMe():
    hour = int(datetime.datetime.now().hour)
    t = time.strftime("%I:%M %p")
    day = cal_day()
    greeting = ""
    if 0 <= hour < 12:
        greeting = "Good Morning"
    elif 12 <= hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    speak(f"{greeting} Hasith, It's {day} and the time is {t}")

def social_media(command_text):
    opened = False
    if 'facebook' in command_text:
        speak("Opening your Facebook")
        webbrowser.open("https://www.facebook.com/")
        opened = True
    elif 'youtube' in command_text:
        speak("Opening your YouTube")
        webbrowser.open("https://www.youtube.com/")
        opened = True
    elif 'instagram' in command_text:
        speak("Opening your Instagram")
        webbrowser.open("https://www.instagram.com/")
        opened = True
    elif 'twitter' in command_text:
        speak("Opening your Twitter")
        webbrowser.open("https://www.twitter.com/")
        opened = True
    elif 'google' in command_text and not "search" in command_text:
        speak("Opening Google")
        webbrowser.open("https://www.google.com/")
        opened = True
    elif 'linkedin' in command_text:
        speak("Opening your LinkedIn")
        webbrowser.open("https://www.linkedin.com/")
        opened = True
    return opened

def condition():
    usage = str(psutil.cpu_percent())
    speak(f"CPU is at {usage} percent")
    battery = psutil.sensors_battery()
    if battery:
        percentage = battery.percent
        speak(f"Boss, Battery is at {percentage} percent")
    else:
        speak("Battery information is not available.")

def get_current_datetime_response():
    current_time = time.strftime("%I:%M %p")
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return f"The current time is {current_time}, and today is {current_date}."

# --- MULTI-SOURCE INFORMATION RETRIEVAL ---
def check_knowledge_base(query_text):
    if "banana" in query_text.lower():
        return "Bananas are rich in potassium, according to my knowledge base."
    return None

def perform_web_search(query_text):
    try:
        results = []
        for result in search(
            query_text, 
            num_results=CONFIG["web_search_max_results"],
            advanced=True,
            lang="en"
        ):
            results.append(f"{result.title}\n{result.description}\n{result.url}")
        
        if not results:
            return "No relevant web results found."
            
        return "Top web results:\n\n" + "\n\n".join(results)
        
    except Exception as e:
        print(f"Web search error: {e}")
        return "Error performing web search. Please try again later."

def get_news(query_text):
    if not CONFIG["news_api_key"]:
        return "News API key is not configured."
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query_text,
        "apikey": CONFIG["news_api_key"],
        "lang": "en",
        "max": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if articles:
            return f"According to GNews, the latest on {query_text} is: {articles[0]['title']}"
        return f"I couldn't find any recent news articles about {query_text} on GNews."
    except requests.exceptions.RequestException as e:
        print(f"GNews API request error: {e}")
        return "Sorry, I'm having trouble fetching news right now."
    except Exception as e:
        print(f"GNews processing error: {e}")
        return "Sorry, an error occurred while processing news information."

def get_weather(query_text):
    if not CONFIG["weather_api_key"] or CONFIG["weather_api_key"] == "604352a1a9393b761f0bf57bb4c6f9f0":
        return "Weather API key is not configured."
    if "weather" in query_text.lower():
        city_name = "Colombo"
        words = query_text.lower().split()
        if "in" in words:
            try:
                city_index = words.index("in") + 1
                if city_index < len(words):
                    city_name = words[city_index].capitalize()
            except ValueError:
                pass
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_name,
            "appid": CONFIG["weather_api_key"],
            "units": "metric"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            if weather_data.get("cod") != 200:
                return f"Could not retrieve weather for {city_name}. Reason: {weather_data.get('message', 'Unknown error')}"
            description = weather_data['weather'][0]['description']
            temp = weather_data['main']['temp']
            return f"The current weather in {weather_data.get('name', city_name)} is {description} with a temperature of {temp} degrees Celsius."
        except requests.exceptions.RequestException as e:
            print(f"OpenWeather API request error: {e}")
            return f"Sorry, I'm having trouble fetching weather for {city_name}."
        except (KeyError, IndexError) as e:
            print(f"OpenWeather data parsing error: {e}")
            return f"Sorry, there was an issue processing the weather data for {city_name}."
        except Exception as e:
            print(f"OpenWeather general error: {e}")
            return f"Sorry, an unexpected error occurred while fetching weather for {city_name}."
    return None

def get_wikipedia(query_text):
    try:
        topic = query_text
        trigger_words = ["tell me about", "what is", "who is", "explain", "define", "search on wikipedia for", "wikipedia"]
        for trigger in trigger_words:
            if topic.lower().startswith(trigger):
                topic = topic[len(trigger):].strip()
        if not topic:
            return None
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            return f"According to Wikipedia: {wikipedia.summary(topic, sentences=2, auto_suggest=False)}"
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:3]
            if options:
                return f"Wikipedia has multiple pages for '{topic}'. Did you mean: {', '.join(options)}?"
            else:
                return f"Wikipedia has multiple results for '{topic}', but I couldn't list specific options."
        except wikipedia.exceptions.PageError:
            return f"Sorry, I couldn't find a Wikipedia page for '{topic}'."
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return "Sorry, I encountered an error while searching Wikipedia."
    except Exception as e:
        print(f"Wikipedia outer error: {e}")
        return "Sorry, I encountered an error while searching Wikipedia."
def get_local_llm_response(query_text):
    try:
        response = chat(
            model='llama2',  # You can use other models like 'mistral', 'phi', etc.
            messages=[{'role': 'user', 'content': query_text}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Local LLM Error: {e}")
        return "I encountered an error accessing the local language model."

def get_information(query_text):
    sources_order = CONFIG.get("fallback_sources", ["web_search","knowledge_base", "wikipedia", "news_api", "weather_api"])
    source_handlers = {
        "web_search": lambda q: perform_web_search(q) if "search" in q else None,
        "knowledge_base": check_knowledge_base,
        "wikipedia": get_wikipedia,
        "news_api": get_news,
        "weather_api": get_weather
    }
    for source_name in sources_order:
        handler = source_handlers.get(source_name)
        if handler:
            try:
                result = handler(query_text)
                if result:
                    return result
            except Exception as e:
                print(f"Error in source {source_name}: {e}")
                continue
    return None

def get_search_information(query_text):
    llm_response = get_local_llm_response(query_text)
    if llm_response and "error" not in llm_response.lower():
        return f"From Local LLM: {llm_response}"
    else:
        return "Sorry, I couldn't retrieve information from the local language model."
    
    result = get_wikipedia(query_text)
    if result and "Sorry, I couldn't find" not in result and "multiple pages for" not in result:
        return result
    result = get_news(query_text)
    if result and "couldn't find any recent news" not in result:
        return result
    return None

def extract_search_term(query_text):
    lower_query = query_text.lower()
    if "search on" in lower_query:
        return query_text.lower().split("search on",1)[1].strip()
    return None

# --- MAIN LOOP ---
if __name__ == "__main__":
    wishMe()
    while True:
        query = command()
        if not query:
            continue
        query = query.lower()
        if "exit" in query or "quit" in query or "goodbye" in query:
            speak("Goodbye Hasith! Have a great day.")
            sys.exit()
        if ("system condition" in query) or ("system status" in query):
            speak("Checking the system condition.")
            condition()
            continue

        if query.strip() == "search on" or query.startswith("search on "):
            search_term = extract_search_term(query)
            if not search_term:
                speak("What specific topic should I search for?")
                search_term_voice = command()
                if search_term_voice:
                    search_term = search_term_voice.lower()
                else:
                    search_term = "information"
                    speak("No topic provided. Searching for general information.")
            if search_term:
                speak(f"Searching for {search_term}...")
                answer = get_search_information(search_term)
                if answer:
                    speak(answer)
                else:
                    speak(f"Sorry, I couldn't find specific information for '{search_term}' from my primary search sources.")
            else:
                speak("No search query was provided.")
            continue

        if social_media(query):
            continue

        if ("volume up" in query) or ("increase volume" in query):
            pyautogui.press("volumeup")
            speak("Volume increased")
            continue
        elif ("volume down" in query) or ("decrease volume" in query):
            pyautogui.press("volumedown")
            speak("Volume decreased")
            continue
        elif ("volume mute" in query) or ("mute volume" in query):
            pyautogui.press("volumemute")
            speak("Volume muted")
            continue
        elif ("open calculator" in query) or ("open calc" in query):
            speak("Opening Calculator")
            subprocess.Popen('calc.exe')
            continue
        elif ("open notepad" in query) or ("open note" in query):
            speak("Opening Notepad")
            subprocess.Popen('notepad.exe')
            continue
        elif ("open browser" in query) or ("google search" in query) or ("search for" in query):
            speak("Boss ,What should I search for?")
            search_query_voice = command()
            if search_query_voice:
                speak(f"Searching for {search_query_voice}...")
                search_results = perform_web_search(search_query_voice)
                webbrowser.open(f"https://www.google.com/search?q={search_query_voice.replace(' ', '+')}")
                speak("\n" + search_results + "\n")
                speak("Here are the top results")
                speak(search_results.split("\n\n")[0])  # Read first result summary
                
            else:
                speak("No search query received")
            continue

        elif ("open explorer" in query) or ("open file explorer" in query) or ("open this pc" in query) or ("open my computer" in query):
            speak("Opening File Explorer")
            subprocess.Popen('explorer.exe')
            continue
        elif ("open downloads" in query):
            speak("Opening Downloads folder")
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            try:
                subprocess.Popen(f'explorer "{downloads_path}"')
            except FileNotFoundError:
                speak("Downloads folder not found.")
            continue
        elif ("close calculator" in query) or ("close calc" in query):
            speak("Closing Calculator")
            os.system("taskkill /f /im calc.exe")
            continue
        elif ("close notepad" in query) or ("close note" in query):
            speak("Closing Notepad")
            os.system("taskkill /f /im notepad.exe")
            continue
        elif ("close edge" in query) or ("close browser" in query) or ("close chrome" in query) or ("close brave" in query):
            speak("Attempting to close the browser.")
            browsers = ["msedge.exe", "chrome.exe", "brave.exe", "firefox.exe"]
            closed_any = False
            for browser_exe in browsers:
                if subprocess.call(f"taskkill /f /im {browser_exe}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    closed_any = True
            if closed_any:
                speak("Browser closed.")
            else:
                speak("No common browser process found or could not be closed.")
            continue
        elif ("close downloads" in query):
            speak("Downloads is typically a folder, not an application to close. File Explorer might be open.")
            continue

        # --- INTENT PREDICTION & HANDLING ---
        try:
            input_text_seq = tokenizer.texts_to_sequences([query])
            padded_sequences = pad_sequences(input_text_seq, maxlen=20, truncating='post')
            prediction = model.predict(padded_sequences)
            predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            handled_by_intent = False
            if predicted_tag == 'datetime':
                datetime_response = get_current_datetime_response()
                speak(datetime_response)
                handled_by_intent = True
            else:
                for intent_data in data['intents']:
                    if intent_data['tag'] == predicted_tag:
                        if intent_data['responses']:
                            speak(random.choice(intent_data['responses']))
                        else:
                            speak(f"I recognized the category as '{predicted_tag}', but I don't have a specific response for that phrase.")
                        handled_by_intent = True
                        break
            if handled_by_intent:
                continue
        except Exception as e:
            print(f"Error during intent prediction or handling: {e}")

        # --- FALLBACK: MULTI-SOURCE INFORMATION RETRIEVAL ---
        speak(f"Let me find some information about '{query}'.")
        information = get_information(query)
        if information:
            speak(information)
        else:
            speak(f"Sorry, I couldn't find any information about '{query}' from my available sources.")
