import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


# ============================================================
# 🔹 Hàm gọi model bạn đã host qua vLLM serve
# ============================================================
def call_vllm_chat(messages):
    """
    Gửi hội thoại tới model đang host bằng vLLM serve (OpenAI API format)
    """
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# 🔹 Công cụ tìm kiếm web (DuckDuckGo — không cần API key)
# ============================================================
search_tool = DuckDuckGoSearchRun()


def search_web(query: str) -> str:
    """Tìm kiếm nhanh trên web (DuckDuckGo)."""
    try:
        result = search_tool.run(query)
        return result
    except Exception as e:
        return f"Lỗi khi tìm kiếm web: {e}"


# ============================================================
# 🔹 Bộ nhớ hội thoại ngắn (4 lượt gần nhất)
# ============================================================
memory = ConversationBufferWindowMemory(
    k=4,
    memory_key="chat_history",
    return_messages=True
)


# ============================================================
# 🔹 Prompt template
# ============================================================
template = """Bạn là một trợ lý AI thông minh, nói tiếng Việt.
Bạn luôn trả lời dựa trên thông tin mới nhất có thể tìm được từ web.

Lịch sử hội thoại gần đây:
{chat_history}

Người dùng hỏi:
{question}

Theo thông tin mà tôi tìm được từ web:
{web_context}

Hãy trả lời ngắn gọn, dễ hiểu, và tự nhiên nhất bằng tiếng Việt.
Nếu không chắc chắn, hãy nói rằng bạn chưa tìm được thông tin chính xác.
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "question", "web_context"],
    template=template
)


# ============================================================
# 🔹 Pipeline chính: Search → Combine → Gọi model
# ============================================================
def get_response(question: str) -> str:
    """Search web rồi gửi kết quả kèm prompt cho model."""
    print(f"🔍 Đang tìm kiếm thông tin trên web cho câu hỏi: {question}")
    web_context = search_web(question)

    # Chuẩn bị nội dung prompt hoàn chỉnh
    full_prompt = prompt.format(
        chat_history=memory.load_memory_variables({})["chat_history"],
        question=question,
        web_context=web_context
    )

    # Gửi sang vLLM model
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI thân thiện, trả lời bằng tiếng Việt."},
        {"role": "user", "content": full_prompt}
    ]

    print("🧠 Đang sinh câu trả lời từ model...")
    answer = call_vllm_chat(messages)

    # Lưu hội thoại lại
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)

    return answer