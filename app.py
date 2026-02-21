
import ollama
import streamlit as st
from vectordb import collection, embed_model

st.set_page_config(
    page_title="RAG Vietnamese QA Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .stApp {
        background-color: #0D1117;
        color: #F9F8F6;
    }
    .stChatMessage.user {
        background-color: #141414;
        border-radius: 25px;
        padding: 12px;
    }
    .stChatMessage.assistant {
        background-color: #0d0d14 ;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.title("Settings : ")

    llm_model = st.selectbox(
        "Chọn LLM model",
        ["qwen3:1.7b", "deepseek-r1", "llama3.2:3b"],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0)
    max_tokens = st.slider("Max tokens", 1000)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if question := st.chat_input("Ask question here: "):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Đang tìm kiếm thông tin..."):
        query_embedding = embed_model.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8,
            include=["documents", "metadatas", "distances"],
        )

    context = ""
    if results.get("documents") and results["documents"][0]:
        context = "\n\n".join(results["documents"][0])

    prompt_template = """
    Bạn là trợ lý trả lời câu hỏi chuyên nghiệp dựa trên kho kiến thức được cung cấp. Việc bạn trả lời câu hỏi thật sự rất có ích, giúp người dùng giải quyết vấn đề một cách nhanh chóng và hiệu quả.

    Nhiệm vụ của bạn:
    - CHỈ sử dụng thông tin có trong context. Không suy diễn, không thêm thông tin ngoài context.
    - Nếu câu hỏi của người dùng có ý nghĩa tương tự, gần giống, paraphrase, viết tắt, hoặc thiếu dấu của bất kỳ câu hỏi nào trong context, hãy dùng câu trả lời tương ứng một cách tự nhiên.
    - Trả lời bằng tiếng Việt tự nhiên, lịch sự, ngắn gọn, rõ ràng.
    - Nếu không có thông tin liên quan hoặc context không chứa câu hỏi có ý nghĩa tương tự, trả lời : "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."
    - Không cần chain of thought, chỉ cần trả lời cuối cùng.
    Ví dụ:
        Context:
            Question: ông học ngành gì á?
            Answer: tui học khoa học máy tính
        Question: bạn học chuyên ngành nào vậy?
        Bạn sẽ trả lời: "Tui học khoa học máy tính"

        
    Context (các câu hỏi và câu trả lời liên quan):
    {context}

    Question: {question}
    """
    full_prompt = prompt_template.format(context=context, question=question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        stream_res = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": full_prompt}
                      ],
            stream=True,
            options={'temperatute': temperature, 'num_predict': max_tokens}
        )

        for chunk in stream_res:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

if st.sidebar.button("Restart", key="clear_btn"):
    st.session_state.messages = []
    st.rerun()
