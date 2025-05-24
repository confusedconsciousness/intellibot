import threading
import logging
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import google.generativeai as genai
from slack_sdk.errors import SlackApiError

load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
model = genai.GenerativeModel("gemini-2.0-flash")
MAX_MESSAGE_PER_THREAD = 10



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
    prompt_parts = []
    for message in messages:
        message_user_id = message.get("user")
        message_bot_id = message.get("bot_id")
        sender_name = "Unknown"
        if message_bot_id:
            sender_name = "Intellibot"
        elif message_user_id:
            try:
                sender_name = app.client.users_info(user=message_user_id)["user"]["name"]
            except SlackApiError:
                logger.error(f"Failed to fetch user info for user ID {message_user_id}")
                sender_name = "Unknown"

        prompt_parts.append(f"{sender_name}: {message.get('text', '')}")

    prompt = "\n".join(prompt_parts)

    try:
        # Initialize the Gemini model inside the handler for this specific call.
        # We're now using 'gemini-pro' as requested.

        logger.info(f"Calling Gemini API with prompt: {prompt}")
        response = model.generate_content(prompt)  # Pass the prompt directly

        # Extract the text from the Gemini response object
        answer = response.text
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}")
        answer = "Sorry, I couldn't get a response from Gemini."

    # Send the LLM's response back to Slack
    say(answer, thread_ts=thread_ts)


if __name__ == "__main__":
    logger.info("Starting Slack app...")
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()
