from langchain_huggingface import ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id = "google/flan-t5-base""


)

model = ChatHugingFace(llm=llm)

reponse = model.invoke("who create you ?")

print(reponse.content)
