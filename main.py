import logging
import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError
from llm.ai_model_chain import get_default_ai_model_chain

from knowledge.rag_pipeline import RAGPipeline
from utils.constant import SOURCE_DIRECTORY, CHROMA_DB_DIRECTORY, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, \
    EMBEDDING_MODEL_NAME, FORCE_RECREATE_STORE
from utils.slack_utils import build_prompt_with_context

load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))
MAX_MESSAGE_PER_THREAD = 10

BOT_NAME = "Intellibot"

user_name_cache = {}

# Initialize RAG Pipeline
rag_pipeline = None


def initialize_rag():
    global rag_pipeline
    try:
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            source_dir=SOURCE_DIRECTORY,
            chroma_dir=CHROMA_DB_DIRECTORY,
            collection_name=COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )
        logger.info("Setting up Vector Store...")
        rag_pipeline.setup_vector_store(force_recreate=FORCE_RECREATE_STORE)
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        raise


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

    logger.info(f"[Slack] Received app_mention from user {user_id} in channel {channel_id} with query: {user_query}")

    # build context for the LLM
    messages = app.client.conversations_replies(channel=channel_id, limit=MAX_MESSAGE_PER_THREAD, ts=thread_ts)[
        "messages"]
    conversation_context = build_conversation_context(messages)

    # Get relevant documents using RAG
    if rag_pipeline and rag_pipeline.vector_store_manager.get_collection_count() > 0:
        relevant_docs = rag_pipeline.query(user_query, k=2)
    else:
        relevant_docs = []
        logger.warning("RAG Pipeline not initialized or empty vector store")

    prompt = build_prompt_with_context(
        query=user_query,
        conversation_history=conversation_context,
        relevant_docs=relevant_docs,
    )

    try:
        logger.info(f"Calling AI model with prompt: {prompt}")
        response = get_default_ai_model_chain().generate(prompt)
        answer = response
    except Exception as e:
        logger.error(f"An error occurred during AI model API call: {e}")
        answer = "Sorry, I couldn't get a response from AI model."

    # Send the LLM's response back to Slack
    say(answer, thread_ts=thread_ts)


def build_conversation_context(messages: list) -> str:
    """Builds a string representation of the conversation history."""
    context_parts = []
    for message in messages:
        message_user_id = message.get("user")
        message_bot_id = message.get("bot_id")

        if message_bot_id:
            sender_name = BOT_NAME
        elif message_user_id:
            sender_name = get_user_name(message_user_id)
        else:
            sender_name = "Unknown"

        context_parts.append(f"{sender_name}: {message.get('text', '')}")

    return "\n".join(context_parts)


def get_user_name(user_id):
    if user_id in user_name_cache:
        return user_name_cache[user_id]
    try:
        user_info = app.client.users_info(user=user_id)
        name = user_info["user"]["real_name"] or user_info["user"]["name"]
        user_name_cache[user_id] = name
        return name
    except SlackApiError:
        logger.error(f"Failed to fetch user info for user ID {user_id}")
        return "Unknown"


if __name__ == "__main__":
    try:
        # Initialize RAG pipeline before starting the Slack app
        initialize_rag()
        logger.info("Starting Slack app...")
        SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
