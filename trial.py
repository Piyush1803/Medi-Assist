import speech_recognition as sr
import pyttsx3
import re
import spacy
import streamlit as st
import pandas as pd

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize the recognizer and the text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Function to extract information from the recognized text using NLP
def extract_information(text):
    doc = nlp(text)

    information = {
        "Name of the patient": None,
        "Age": None,
        "Gender": None,
        "Phone Number": None,
        "Date of Birth": None,
        "Address": None
    }

    # Regex patterns to match specific fields
    patterns = {
        "Name of the patient": [r"name(?: of the)? patient\s*(?:is)?\s*([a-zA-Z\s]+)(?=\s*(age|gender|phone number|address|$))"],
        "Age": [r"age\s*(?:is)?\s*(\d+)", r"\b(\d+)\s+years?\s+old\b"],
        "Gender": [r"gender\s*(?:is)?\s*(\w+)", r"patient\s+is\s+(?:a\s+)?(\w+)\b"],
        "Phone Number": [r"phone number\s*(?:is)?\s*([\d\s]+)", r"contact\s+number\s*(?:is)?\s*([\d\s]+)"],
        "Date of Birth": [r"(?:date of birth|birth date)\s*(?:is)?\s*(.+)", r"born\s+on\s*(.+)"],
        "Address": [r"address\s*(?:is)?\s*(.+)", r"lives\s+at\s*(.+)"]
    }

    # Match regex patterns
    for field, patterns_list in patterns.items():
        if information[field] is None:
            for pattern in patterns_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    information[field] = match.group(1).strip()
                    break

    return information

# Function to listen for speech
def start_listening():
    st.write("Listening for speech...")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio_data = r.listen(source)

    # Store audio data in session state
    st.session_state['audio_data'] = audio_data
    st.success("Audio captured successfully!")

# Function to stop listening and process speech
def stop_listening_and_process():
    if 'audio_data' not in st.session_state:
        st.warning("No audio captured. Please press 'Start Speech' first.")
        return None

    audio_data = st.session_state['audio_data']

    try:
        st.write("Processing speech...")
        text = r.recognize_google(audio_data)
        st.write(f"Recognized Speech: {text}")

        # Extract information from recognized speech
        extracted_info = extract_information(text)

        # Create a DataFrame for the extracted information
        df = pd.DataFrame([extracted_info])
        
        st.write("Extracted Information:")
        st.dataframe(df)

        return df

    except sr.UnknownValueError:
        st.error("Could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from the speech recognition service; {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Streamlit App UI
def app():
    st.title("Doctor's Prescription Form")

    # Buttons to start and stop speech
    if st.button('Start Speech'):
        start_listening()

    if st.button('Stop Speech and Process'):
        df = stop_listening_and_process()
        
        # You can save the dataframe or display it
        if df is not None:
            st.write("Data saved to dataframe.")
            st.write(df)

# Run the Streamlit app
if __name__ == "__main__":
    app()
