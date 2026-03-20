from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

# ---------------- PIPELINE ----------------
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03
)

# ---------------- LLM ----------------
llm = HuggingFacePipeline(pipeline=pipe)

# ---------------- CHAT MODEL ----------------
chat_model = ChatHuggingFace(llm=llm)

# ---------------- INVOKE ----------------
result = chat_model.invoke("What is data science?")

print(result.content)
