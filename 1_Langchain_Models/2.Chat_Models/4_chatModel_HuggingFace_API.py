from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",                        # model using 
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)