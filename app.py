# Importing required libraries
import warnings
warnings.filterwarnings("ignore")

import json
import subprocess
import sys
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
import gradio as gr
from huggingface_hub import hf_hub_download
from typing import List, Tuple
from logger import logging
from exception import CustomExceptionHandling


# Download gguf model files
llm = None
llm_model = None

hf_hub_download(
    repo_id="bartowski/Dolphin3.0-Llama3.2-1B-GGUF",
    filename="Dolphin3.0-Llama3.2-1B-Q6_K.gguf",
    local_dir="./models",
)
hf_hub_download(
    repo_id="bartowski/Dolphin3.0-Qwen2.5-0.5B-GGUF",
    filename="Dolphin3.0-Qwen2.5-0.5B-Q6_K.gguf",
    local_dir="./models",
)

# Set the title and description
title = "Dolphin🐬 Llamacpp"
description = """Dolphin 3.0 is a powerful, general-purpose local AI model designed for coding, math, and various other tasks, aiming similar to the models like ChatGPT and Claude."""


def respond(
    message: str,
    history: List[Tuple[str, str]],
    model: str,
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
):
    """
    Respond to a message using the Dolphin-3 model via Llama.cpp.

    Args:
        - message (str): The message to respond to.
        - history (List[Tuple[str, str]]): The chat history.
        - model (str): The model to use.
        - system_message (str): The system message to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The temperature of the model.
        - top_p (float): The top-p of the model.
        - top_k (int): The top-k of the model.
        - repeat_penalty (float): The repetition penalty of the model.

    Returns:
        str: The response to the message.
    """
    try:
        # Load the global variables
        global llm
        global llm_model

        # Load the model
        if llm is None or llm_model != model:
            llm = Llama(
                model_path=f"models/{model}",
                flash_attn=False,
                n_gpu_layers=0,
                n_batch=32,
                n_ctx=8192,
            )
            llm_model = model
        provider = LlamaCppPythonProvider(llm)

        # Create the agent
        agent = LlamaCppAgent(
            provider,
            system_prompt=f"{system_message}",
            predefined_messages_formatter_type=MessagesFormatterType.CHATML,
            debug_output=True,
        )

        # Set the settings like temperature, top-k, top-p, max tokens, etc.
        settings = provider.get_provider_default_settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.max_tokens = max_tokens
        settings.repeat_penalty = repeat_penalty
        settings.stream = True

        messages = BasicChatHistory()

        # Add the chat history
        for msn in history:
            user = {"role": Roles.user, "content": msn[0]}
            assistant = {"role": Roles.assistant, "content": msn[1]}
            messages.add_message(user)
            messages.add_message(assistant)

        # Get the response stream
        stream = agent.get_chat_response(
            message,
            llm_sampling_settings=settings,
            chat_history=messages,
            returns_streaming_generator=True,
            print_output=False,
        )

        # Log the success
        logging.info("Response stream generated successfully")

        # Generate the response
        outputs = ""
        for output in stream:
            outputs += output
            yield outputs

    # Handle exceptions that may occur during the process
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


# Create a chat interface
demo = gr.ChatInterface(
    respond,
    examples=[["What is the capital of France?"], ["Why is the color of the sky blue?"], ["What is gravity?"]],
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Dropdown(
            choices=[
                "Dolphin3.0-Llama3.2-1B-Q6_K.gguf",
                "Dolphin3.0-Qwen2.5-0.5B-Q6_K.gguf",
            ],
            value="Dolphin3.0-Qwen2.5-0.5B-Q6_K.gguf",
            label="Model",
            info="Select the AI model to use for chat",
        ),
        gr.Textbox(
            value="You are Dolphin, a helpful AI assistant focused on accurate and ethical responses.",
            label="System Prompt",
            info="Define the AI assistant's personality and behavior",
            lines=2,
        ),
        gr.Slider(
            minimum=512,
            maximum=4096,
            value=2048,
            step=512,
            label="Max Tokens",
            info="Maximum length of response (higher = longer replies)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Creativity level (higher = more creative, lower = more focused)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
            info="Nucleus sampling threshold",
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
            info="Limit vocabulary choices to top K tokens",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition Penalty",
            info="Penalize repeated words (higher = less repetition)",
        ),
    ],
    theme="Ocean",
    submit_btn="Send",
    stop_btn="Stop",
    title=title,
    description=description,
    chatbot=gr.Chatbot(scale=1, show_copy_button=True),
    flagging_mode="never",
)


# Launch the chat interface
if __name__ == "__main__":
    demo.launch(debug=False)
