from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

config = {
    "model": "gemma3:1b",
    "temperature": 0.7,
    "top_p": 0.9
}

def load_model():
    # 1. initialize  LLM model supporting get an answer.
    llm = OllamaLLM(
        **config
    )

    # 2. design prompt 
    template = """Bạn là một trợ lý ảo thông minh và thân thiện, sử dụng tiếng Việt.
    Bạn luôn trả lời người dùng một cách ngắn gọn, dễ hiểu, và dựa trên lịch sử hội thoại gần đây.
    Nếu chưa có thông tin, hãy nói rõ bạn không biết thay vì đoán mò.

    Lịch sử trò chuyện gần đây:
    {chat_history}

    Câu hỏi mới:
    {question}

    Hãy trả lời câu hỏi bằng tiếng Việt, giữ phong cách tự nhiên, lịch sự và dễ hiểu.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"]
    )

    # 3. save chat_history following slide window
    memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        return_messages=True
    )

    # 4. create chain 
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return conversation_chain


# get response function
def get_response(conversation_chain, question: str) -> str: 
    response = conversation_chain.run(question)
    return response
