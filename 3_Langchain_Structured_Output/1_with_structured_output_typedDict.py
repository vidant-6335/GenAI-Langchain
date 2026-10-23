from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",                        # model using 
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

    # give discription to the llm so that it can search accordingly to the given discription


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it is an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I am gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver. The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality. However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful                          
Review by Nitish Singh
""")

print(result)
print(result['name'])


# The main problem with TypedDict is that it does not enforce data types at runtime. It is primarily used for static type checking. Both dictionaries print successfully, even though new_person2 has a string '25' instead of an integer 25.

# from typing import TypedDict
# class Person(TypedDict):
#     name: str
#     age: int
# new_person1: Person = {'name':'Vidant', 'age':25}
# new_person2: Person = {'name':'Vidant', 'age':'25'}
# print(new_person1)
# print(new_person2)
