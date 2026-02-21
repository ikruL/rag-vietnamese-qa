
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectordb import collection, embed_model


model = OllamaLLM(model="qwen3:4b")

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

while True:
    question = input("\nQuestion to answer (q to quit): ").strip()
    if question.lower() in ['q']:
        print("Bye!")
        break

    query_embedding = embed_model.embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    context = ""
    if results["documents"][0]:
        context = "\n\n".join(results["documents"][0])

    response = chain.invoke({"context": context, "question": question})
    print(f"\nResponse: {response}")
