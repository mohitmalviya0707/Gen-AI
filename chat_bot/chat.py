from dotenv import load_dotenv  
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---------------- MODEL ----------------
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

# ---------------- MODE SELECTION ----------------    
print("Choose your AI model")
print("Press 1 for Angry model")
print("Press 2 for Funny model")
print("Press 3 for Sad model")

choice = int(input("Enter your choice: "))

if choice == 1:
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == 2:
    mode = "You are a very funny AI agent. You respond with humor and jokes."
elif choice == 3:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."
else:
    print("Invalid choice, defaulting to normal mode.")
    mode = "You are a helpful AI assistant."

# ---------------- MESSAGE MEMORY ----------------
messages = [
    SystemMessage(content=mode)
]

print("------ Type 0 to exit ------")

# ---------------- CHAT LOOP ----------------
while True:
    prompt = input("You: ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)

print("Chat Ended")
