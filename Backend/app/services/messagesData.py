from app.services.prompts import llm_system_prompt

session_messages: dict[str, list[dict[str, str]]] = {}

def get_messages(session_id: str) -> list[dict[str, str]]:
    if session_id not in session_messages:
        session_messages[session_id] = [
            {"role": "system", "content": llm_system_prompt}
        ]

    return session_messages[session_id]

def delete_session_id_from_map(session_id: str) -> None:
    session_messages.pop(session_id, None)


map_messages = session_messages
getMessages = get_messages
deleteSessionIdFromMap = delete_session_id_from_map
