from fastapi import FastAPI

from simplifier import Simplifier

simplifier = None
app = FastAPI()


# create a route
@app.get("/")
def index():
    return {"text": "Text Simplifier"}


@app.on_event("startup")
def startup_event():
    global simplifier
    simplifier = Simplifier()


# Your FastAPI route handlers go here
@app.get("/simplify")
def simplify_text(text: str):
    simplified_text = simplifier.simplify(text)

    response = {
        'text': text,
        'simplified_text': simplified_text
    }

    return response