import os
from typing import Iterable

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

from app.models.llm import Search
from app.services import config
from app.services.advanced_rag import judge_answer, retrieve_with_pipeline
from app.services.messagesData import get_messages
from app.services.prompts import llm_system_prompt, search_system_prompt

load_dotenv()

client = AsyncInferenceClient(
    model=os.getenv("huggingface_model_id"),
    api_key=os.getenv("hugging_face_key"),
)

FULL_CONTEXT_KEYWORDS = {
    "summary",
    "summarize",
    "overview",
    "explain",
    "bullet",
    "bullets",
    "key points",
    "important concepts",
    "action items",
    "decisions",
    "tell me about",
    "what is this about",
    "what's this about",
}
ACTION_QUERY_KEYWORDS = {
    "action items",
    "decisions",
    "next steps",
}
SMALL_TALK_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}
OUT_OF_CONTEXT_RESPONSE = "Sorry, this is out of context, I can't help you with it."


def format_conversation_history(messages: Iterable[dict[str, str]]) -> str:
    history_lines = [
        f"{message['role']} : {message['content']}"
        for message in messages
        if message["role"] != "system"
    ]
    return "\n".join(history_lines).strip()


def should_use_full_context(query: str) -> bool:
    normalized_query = query.strip().lower()
    return any(keyword in normalized_query for keyword in FULL_CONTEXT_KEYWORDS)


def is_small_talk(query: str) -> bool:
    return query.strip().lower() in SMALL_TALK_MESSAGES


def is_action_item_query(query: str) -> bool:
    normalized_query = query.strip().lower()
    return any(keyword in normalized_query for keyword in ACTION_QUERY_KEYWORDS)


def extract_actionable_lines(retrieved_context: str) -> list[str]:
    actionable_lines: list[str] = []
    seen: set[str] = set()
    keywords = (
        "must",
        "submit",
        "required",
        "should",
        "need to",
        "at least",
        "public",
        "deployed",
        "will not be evaluated",
    )

    for line in retrieved_context.splitlines():
        cleaned_line = line.strip(" -\t")
        if not cleaned_line:
            continue

        normalized_line = cleaned_line.lower()
        if not any(keyword in normalized_line for keyword in keywords):
            continue

        if cleaned_line in seen:
            continue

        seen.add(cleaned_line)
        actionable_lines.append(cleaned_line)

    return actionable_lines


def build_action_items_response(retrieved_context: str) -> str | None:
    actionable_lines = extract_actionable_lines(retrieved_context)
    if not actionable_lines:
        return None

    bullet_list = "\n".join(f"- {line}" for line in actionable_lines[:8])
    return f"## Action Items / Decisions\n\n{bullet_list}"


async def build_retrieval_query(query: str, conversation_history: str) -> str:
    # Query Translation / rewriting using a small model (RAG-2). Resolves
    # pronouns and history into a clean, standalone, keyword-rich search query.
    if not config.ENABLE_QUERY_REWRITE:
        return query

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "SearchSchema",
            "schema": Search.model_json_schema(),
            "strict": True,
        },
    }

    try:
        search_decision_json = (
            await client.chat_completion(
                messages=[
                    {"role": "system", "content": search_system_prompt},
                    {
                        "role": "user",
                        "content": f"{conversation_history}\nuser : {query}".strip(),
                    },
                ],
                response_format=response_format,
                temperature=0.1,
            )
        ).choices[0].message.content
        search_decision = Search.model_validate_json(search_decision_json)
        if search_decision.search_query:
            return search_decision.search_query
    except Exception:
        pass

    return query


def build_user_prompt(
    query: str,
    conversation_history: str,
    retrieved_context: str,
) -> str:
    context_block = retrieved_context or "[NO RELEVANT INFORMATION FOUND]"
    history_block = conversation_history or "[NO PRIOR CONVERSATION]"
    return (
        f"Retrieved Context:\n{context_block}\n\n"
        f"Conversation History:\n{history_block}\n\n"
        f"User Question:\n{query}"
    )


async def handle_user_query(query: str, session_id: str) -> dict:
    """Run the Advanced RAG pipeline for one user turn.

    Returns ``{"answer": str, "trace": dict}``. ``trace`` records which
    advanced-RAG stages ran (query rewrite, sub-query, HyDE, rerank, corrective
    RAG, judge) so the pipeline is observable from the client.
    """
    conversation_messages = get_messages(session_id)
    conversation_history = format_conversation_history(conversation_messages)

    trace: dict = {"stages": []}
    retrieved_information = ""

    if not is_small_talk(query):
        # Stage 1 -- Query Translation / rewriting (SLM).
        retrieval_query = await build_retrieval_query(query, conversation_history)
        if retrieval_query != query:
            trace["stages"].append("query_rewrite")
        trace["rewritten_query"] = retrieval_query

        # Stages 2-6 -- sub-query, HyDE, hybrid retrieval, rerank, corrective RAG.
        retrieved_information, retrieval_trace = await retrieve_with_pipeline(
            session_id=session_id,
            retrieval_query=retrieval_query,
            include_all=should_use_full_context(query),
        )
        trace["stages"].extend(retrieval_trace.get("stages", []))
        trace["retrieval"] = retrieval_trace
    else:
        trace["stages"].append("small_talk")

    if is_action_item_query(query):
        action_items_response = build_action_items_response(retrieved_information)
        if action_items_response:
            conversation_messages.append({"role": "user", "content": query})
            conversation_messages.append(
                {"role": "assistant", "content": action_items_response}
            )
            return {"answer": action_items_response, "trace": trace}

    user_prompt = build_user_prompt(query, conversation_history, retrieved_information)
    response = await client.chat_completion(
        messages=[
            {"role": "system", "content": llm_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content or OUT_OF_CONTEXT_RESPONSE

    # Stage 7 -- LLM judge: verify the answer is grounded in the context.
    # If it hallucinated facts not in the context, refuse instead of misleading.
    if retrieved_information and not is_small_talk(query):
        judgement = await judge_answer(retrieved_information, answer)
        trace["grounded"] = judgement.grounded
        if config.ENABLE_LLM_JUDGE:
            trace["stages"].append("llm_judge")
        if not judgement.grounded:
            answer = OUT_OF_CONTEXT_RESPONSE

    conversation_messages.append({"role": "user", "content": query})
    conversation_messages.append({"role": "assistant", "content": answer})

    return {"answer": answer, "trace": trace}


handel_user_query = handle_user_query
