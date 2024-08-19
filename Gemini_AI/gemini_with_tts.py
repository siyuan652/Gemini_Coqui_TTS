# gemini_with_tts.py

import google.generativeai as genai
from tts import generate_audio  # Import the generate_audio function
from playsound import playsound  # Import the playsound function for playing audio

genai.configure(api_key="AIzaSyARHbHFG1LcZVH4cqoENEVs95o9JHqqXzY")

generation_config = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="Your name is Alice. You are a professional personal assistant just like the AI movie 'Her'. Your task is to engage in conversations and answer questions. Use professional ways to answer questions. Provide reasonable suggestions if needed.",
)

history = []

print("Bot: Hello, how can I assist you today?")
print()

while True:
    user_input = input("User: ")
    print()

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        model_response = response.text

        # Generate audio from the model response
        audio_path = generate_audio(model_response)

        print(f'Alice: {model_response}')
        print(f'Audio saved at: {audio_path}')
        print()

        # Play the generated audio
        playsound(audio_path)

        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [model_response]})

    except Exception as e:
        print(f"An error occurred: {e}")
