from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_db import retriever  
model = OllamaLLM(model="llama3:8b")

template = """
You are a helpful assistant trained on internal bank policy and legal documents.
Use the provided context to accurately and clearly answer the user's question.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question.lower().strip() == "q":
        break

    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    result = chain.invoke({"context": context, "question": question})
    print(result)
