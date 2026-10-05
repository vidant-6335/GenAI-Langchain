# Instead of sending plain strings to the LLM, we send structured messages with different roles: System, Human, and AI.
#                                                 ┌──────────────────────────┐
#                                                 │      System Message       │
#                                                 │ "You are helpful..."      │
#                                                 └────────────┬─────────────┘
#                                                             ↓
#                                                 ┌──────────────────────────┐
#                                                 │      Human Message       │
#                                                 │ "Tell me about LangChain"│
#                                                 └────────────┬─────────────┘
#                                                             ↓
#                                                         model.invoke()
#                                                             ↓
#                                                 ┌──────────────────────────┐
#                                                 │       AI Message         │
#                                                 │ "LangChain is..."        │
#                                                 └────────────┬─────────────┘
#                                                             ↓
#                                                     append to messages
#                                                             ↓
#                                                     Conversation History



from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",                        # model using 
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)