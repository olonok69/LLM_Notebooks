# streamlit and google-cloud-aiplatform


# https://streamlit.io/
# https://cloud.google.com/appengine/docs/an-overview-of-app-engine
# https://cloud.google.com/appengine/docs/flexible/managing-projects-apps-billing
# https://cloud.google.com/appengine/docs/flexible
# https://cloud.google.com/appengine/docs/flexible/configuration-files
# https://cloud.google.com/appengine/docs/flexible/reference/app-yaml?tab=python#syntax
# https://cloud.google.com/appengine/docs/flexible/python/runtime
# https://cloud.google.com/run/docs/configuring/session-affinity#yaml
# https://cloud.google.com/appengine/docs/flexible/reference/app-yaml?tab=python#services


import streamlit as st
from google.oauth2 import service_account
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, ChatSession
import vertexai.preview.generative_models as generative_models
from dotenv import dotenv_values
import json
from constants import question, question2


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


def main(chat):
    st.title("📝 File Q&A with Gemini Pro")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = "False"
    if "chat" not in st.session_state:
        st.session_state["chat"] = chat

    uploaded_files = st.file_uploader(
        "Upload the file to start a conversation",
        type=("pdf", "jpg"),
        accept_multiple_files=True,
    )

    content = []

    prompt = st.chat_input("Enter your questions here", disabled=not input)
    if uploaded_files:
        if st.session_state["initialized"] == "False":

            for uploaded_file in uploaded_files:
                im_bytes = uploaded_file.read()
                im_b64 = base64.b64encode(im_bytes).decode("utf8")
                image = Part.from_data(data=im_b64, mime_type="application/pdf")
                content.append(image)
            prompt1 = [f"""{question2} """] + content
            response = get_chat_response(st.session_state["chat"], prompt1)

            st.session_state["chat_answers_history"].append(response)
            st.session_state["user_prompt_history"].append(question2)
            st.session_state["chat_history"].append((question2, response))
            st.session_state["initialized"] = "True"

        elif st.session_state["initialized"] == "True":
            prompt = [f"""{prompt} """]
            response = get_chat_response(st.session_state["chat"], prompt)
            st.session_state["chat_answers_history"].append(response)
            st.session_state["user_prompt_history"].append(prompt[0])
            st.session_state["chat_history"].append((prompt[0], response))

        if st.session_state["chat_answers_history"]:
            for i, j in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
            ):
                message1 = st.chat_message("user")
                message1.write(j)
                message2 = st.chat_message("assistant")
                message2.write(i)


if __name__ == "__main__":
    config = dotenv_values("keys/.env")
    with open("keys/complete-tube-421007-9a7c35cd44e2.json") as source:
        info = json.load(source)

    vertex_credentials = service_account.Credentials.from_service_account_info(info)

    vertexai.init(
        project=config["PROJECT"],
        location=config["REGION"],
        credentials=vertex_credentials,
    )
    model = GenerativeModel(
        "gemini-1.5-pro-preview-0409",
        system_instruction=[
            """You a helpful agent who helps to extract relevant information from documents"""
        ],
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
        },
    )
    chat = model.start_chat(response_validation=False)
    main(chat=chat)
