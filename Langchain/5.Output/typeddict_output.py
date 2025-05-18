from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')

# schema

class Review(TypedDict):
    # # generate default system prompt according to keys
    # summary:str
    # sentimanet:str
    
    # we can specify also description with each key
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# defining model with structured output not working with GoogleAI works with OpenAI
struct_model = model.with_structured_output(Review)

result =model.invoke("""This is good product""")
print(result)
