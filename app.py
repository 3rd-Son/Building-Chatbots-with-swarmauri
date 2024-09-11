from dotenv import load_dotenv
import streamlit as st
import os
import gradio as gr
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import (
    SimpleConversationAgent,
)
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import (
    MaxSystemContextConversation,
)

load_dotenv()

API_KEY = os.getenv("GROQ_APIA_KEY")

llm = GroqModel(api_key=API_KEY)
allowed_models = llm.allowed_models

conversation = MaxSystemContextConversation()


# Assuming you have these classes and functions already defined
def load_model(selected_model):
    # Dummy model loader for demonstration purposes
    return GroqModel(api_key=API_KEY, name=selected_model)


def converse(input_text, system_context, model_name):
    st.write(f"System context: {system_context}")
    st.write(f"Selected model: {model_name}")

    # Load the model
    llm = load_model(model_name)

    # Assuming conversation and agent objects are defined elsewhere
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    agent.conversation.system_context = SystemMessage(content=system_context)

    # Process the input text
    input_text = str(input_text)
    st.write("Conversation history:", conversation.history)

    # Execute the model and get the result
    result = agent.exec(input_text)
    st.write("Result:", result)

    return str(result)


# Streamlit Interface
st.title("System Context Conversation")
st.write("Interact with the agent using a selected model and system context")

# Input fields for system context and model selection
system_context = st.text_input("System Context", "Provide the system context here...")
model_name = st.selectbox("Model Name", allowed_models)

# Textbox for user input
input_text = st.text_area("Your Input", "Type your message here...")

# Button to submit
if st.button("Submit"):
    if input_text and system_context:
        response = converse(input_text, system_context, model_name)
        st.write("Response from model:", response)
    else:
        st.write("Please provide both input text and system context.")
