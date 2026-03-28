
from dotenv import load_dotenv

load_dotenv()
import os



def hello():
    print("Hello")

    key= os.getenv("GROQ_API_KEY")
    print(key)

hello()