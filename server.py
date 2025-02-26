import base64
import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
from worker import speech_to_text, text_to_speech, watsonx_process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return None


import requests


def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    base_url = '...'
    api_url = base_url + '/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }

    # Set up the body of our HTTP request
    body = audio_binary

    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()

    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text


def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    base_url = '...'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP Post reqeust to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content


def watsonx_process_message(user_message):
    # Set the prompt for Watsonx API
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
    Translate the query to Spanish: ```{user_message}```."""
    response_text = model.generate_text(prompt=prompt)
    print("wastonx response:", response_text)
    return response_text


# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import ModelTypes

# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
# API_KEY = "Your WatsonX API"
PROJECT_ID = "skills-network"

# Define the credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "apikey": API_KEY
}

# Specify model_id that will be used for inferencing
model_id = ModelTypes.FLAN_UL2

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = ModelTypes(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)


@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  # Get user's message from their request
    print('user_message', user_message)

    voice = request.json['voice']  # Get user\'s preferred voice from their request
    print('voice', voice)

    # Call watsonx_process_message function to process the user's message and get a response back
    watsonx_response_text = watsonx_process_message(user_message)

    # Clean the response to remove any emptylines
    watsonx_response_text = os.linesep.join([s for s in watsonx_response_text.splitlines() if s])

    # Call our text_to_speech function to convert Watsonx Api's reponse to speech
    watsonx_response_speech = text_to_speech(watsonx_response_text, voice)

    # convert watsonx_response_speech to base64 string so it can be sent back in the JSON response
    watsonx_response_speech = base64.b64encode(watsonx_response_speech).decode('utf-8')

    # Send a JSON response back to the user containing their message\'s response both in text and speech formats
    response = app.response_class(
        response=json.dumps(
            {"watsonxResponseText": watsonx_response_text, "watsonxResponseSpeech": watsonx_response_speech}),
        status=200,
        mimetype='application/json'
    )

    print(response)
    return response


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
