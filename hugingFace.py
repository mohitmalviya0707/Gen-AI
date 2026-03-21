from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

)
texts = [
    "Hello this is Mohit Malviya ",     
    "you are my falak",
    "we both are good"
]



vector = embedding.embed_documents(texts)

print(vector)
