
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
from vectordb import collection, embed_model

MODEL = 'qwen3:4b'
TEMP = 0.2

model = OllamaLLM(model=MODEL)

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
prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | model

print('*' * 80)
print("\n\nWelcome to RAG VietNamese QA Chatbot !! \n\n")
while True:
    question = input("Ask question here (q to quit): ").strip()
    if question.lower() in ['q']:
        print("Bye!")
        break

    query_embedding = embed_model.embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )

    context = ""
    if results.get("documents") and results["documents"][0]:
        context = "\n\n".join(results["documents"][0])

    full_prompt = prompt.format(context=context, question=question)

    stream_res = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}
                  ],
        stream=True,
        options={'temperatute': TEMP}
    )

    print(f"\nResponse: ")

    for chunk in stream_res:
        if 'message' in chunk and 'content' in chunk['message']:
            print(chunk['message']['content'], end='', flush=True)
    print("\n" + "=" * 80)
