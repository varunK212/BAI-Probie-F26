"""
API Module

This module contains pure functions for interacting with OpenRouter through the
OpenAI-compatible Python SDK.
All functions here follow functional programming principles:
- No side effects (besides API calls)
- Clear inputs and outputs
- Predictable behavior

Key Concepts:
- Pure functions: Given the same input, always return the same output
- Type hints: Declare what types go in and what comes out
- Error handling: Gracefully handle things that can go wrong
"""

from typing import List

from openai import OpenAI

from src.settings import settings
from src.chatbot.constants import OPENROUTER_BASE_URL, SYSTEM_PROMPT
from src.chatbot.models import ChatMessage


# create_client function
# This function creates and returns an OpenAI client configured for OpenRouter.
# The client is what we use to make API calls to OpenRouter.
#
# Instructions:
# 1. Create a function called create_client that takes Settings as a parameter
# 2. Return type should be OpenAI
# 3. Create an OpenAI client using: OpenAI(api_key=settings.openai_api_key) # used openrouter 
# 4. Return the client
# 5. Add a docstring explaining what the function does
#
# Hint: Look at the import statements to see what's available!
#
# Example usage after completion:
#   client = create_client()
#
def create_client() -> OpenAI:
    """Create an OpenAI-compatible client configured for OpenRouter."""

    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


# get_chat_response function
# This is the main function that sends messages to OpenRouter and gets a response.
# It needs to handle the API call and potential errors gracefully.
#
# Instructions:
# 1. Create a function called get_chat_response with these parameters:
#    - client: OpenAI
#    - messages: List[ChatMessage]
#    - model: str (which GPT model to use)
# 2. Return type should be ChatMessage (the assistant's reply)
# 3. Convert the ChatMessage objects to dictionaries (the API expects dicts)
#    Hint: message.model_dump() converts a Pydantic model to a dict
# 4. Call the OpenRouter API using:
#    response = client.chat.completions.create(
#        model=model,
#        messages=message_dicts
#    )
# 5. Extract the assistant's message from the response
#    Hint: response.choices[0].message has the content and role
# 6. Create a ChatMessage with the assistant's role and content
# 7. Return the ChatMessage
# 8. Add a try-except block to catch errors and display helpful messages
#
# Example structure:
# def get_chat_response(...) -> ChatMessage:
#     """Your docstring here."""
#     try:
#         # Convert messages to dicts
#         message_dicts = [msg.model_dump() for msg in messages]
#
#         # Call OpenRouter API
#         response = client.chat.completions.create(...)
#
#         # Extract the assistant's message
#         assistant_message = response.choices[0].message
#
#         # Create and return ChatMessage
#         return ChatMessage(
#             role=assistant_message.role,
#             content=assistant_message.content
#         )
#     except Exception as e:
#         # Handle errors
#         # Hint: You might want to print the error and re-raise it
#         # so the main program can decide what to do
#         raise
#
def get_chat_response(
     
    client: OpenAI, messages: List[ChatMessage], model: str
) -> ChatMessage:
    """serialize message and calls chat completions. returns a chat message"""

    message_dicts = [message.model_dump() for message in messages]
    response = client.chat.completions.create(
        model=model,
        messages=message_dicts,
    )

    assistant_message = response.choices[0].message
    if assistant_message.content is None:
        raise ValueError("OpenRouter returned a response without text content.")

    return ChatMessage(
        role=assistant_message.role,
        content=assistant_message.content,
    )


# create_system_message function (Helper Function)
# This helper function creates the system message that defines the chatbot's personality.
# It's a simple function, but demonstrates good code organization.
#
# Instructions:
# 1. Create a function called create_system_message that takes no parameters
# 2. Return type should be ChatMessage
# 3. Return a ChatMessage with role="system" and content=SYSTEM_PROMPT (imported from constants)
# 4. Add a docstring
#
# Example usage after completion:
#   system_msg = create_system_message()
#
def create_system_message() -> ChatMessage:
    """Creates the initial system prompt"""

    return ChatMessage(role="system", content=SYSTEM_PROMPT)
