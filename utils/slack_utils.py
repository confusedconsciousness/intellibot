def build_prompt_with_context(query: str, conversation_history: str, relevant_docs: list) -> str:
    # Combine relevant documents into context
    doc_context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    Based on the following context and conversation history, please provide a helpful response.

    KNOWLEDGE BASE CONTEXT:
    {doc_context}

    CONVERSATION HISTORY:
    {conversation_history}

    CURRENT QUERY:
    {query}

    Please provide a clear and concise response that incorporates both the relevant knowledge base information and takes into account the conversation context."""

    return prompt
