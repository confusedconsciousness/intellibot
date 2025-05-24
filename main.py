import logging
import os

from openai import responses
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError

from ai.AIModel import get_ai_model
from ai.AiModelChain import AIModelChain, get_default_ai_model_chain

load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
MAX_MESSAGE_PER_THREAD = 10

BOT_NAME = "Intellibot"


# --- Slack Message Handler ---
# This decorator registers a function to handle 'app_mention' events.
# The bot will only respond when explicitly mentioned in a channel.
@app.event("app_mention")
def handle_app_mention(body, say):
    """
    Handles incoming Slack 'app_mention' events, extracts the user's query,
    sends it to the LLM using google-generativeai, and posts the response.
    """
    # The 'text' field in an app_mention event includes the bot's mention,
    # so we need to remove it to get the actual user query.
    # The bot's user ID is usually the first part of the text.
    event = body["event"]

    text = event["text"]
    user_id = event["user"]
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])

    # Remove the bot's mention from the text to get the clean query
    # Example: "<@U0123ABC> What is the weather like?" -> "What is the weather like?"
    bot_user_id = app.client.auth_test()["user_id"]
    user_query = text.replace(f"<@{bot_user_id}>", "").strip()

    logger.info(f"Received app_mention from user {user_id} in channel {channel_id} with query: {user_query}")

    if not user_query:
        say(f"Hey <@{user_id}>, please provide a query after mentioning me!", thread_ts=thread_ts)
        return

    # build context for the LLM
    messages = app.client.conversations_replies(channel=channel_id, limit=MAX_MESSAGE_PER_THREAD, ts=thread_ts)[
        "messages"]
    prompt = build_context(messages)

    try:
        # Initialize the Gemini model inside the handler for this specific call.
        # We're now using 'gemini-pro' as requested.

        logger.info(f"Calling AI model with prompt: {prompt}")
        # Use the AI model chain to generate a response, which will try multiple models if one fails
        response = get_default_ai_model_chain().generate(prompt)
        answer = response
    except Exception as e:
        logger.error(f"An error occurred during AI model API call: {e}")
        answer = "Sorry, I couldn't get a response from AI model."

    # Send the LLM's response back to Slack
    say(answer, thread_ts=thread_ts)


def build_context(messages):
    """
    Builds a context string from the last few messages in the thread.
    """
    # build context for the LLM
    prompt_parts = []
    for message in messages:
        message_user_id = message.get("user")
        message_bot_id = message.get("bot_id")
        sender_name = "Unknown"
        if message_bot_id:
            sender_name = BOT_NAME
        elif message_user_id:
            try:
                sender_name = app.client.users_info(user=message_user_id)["user"]["name"]
            except SlackApiError:
                logger.error(f"Failed to fetch user info for user ID {message_user_id}")
                sender_name = "Unknown"

        prompt_parts.append(f"{sender_name}: {message.get('text', '')}")

    return "\n".join(prompt_parts)


if __name__ == "__main__":
    logger.info("Starting Slack app...")
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()
