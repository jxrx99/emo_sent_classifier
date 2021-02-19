from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import malaya
from fastapi.staticfiles import StaticFiles


emotion_model = malaya.emotion.transformer(model = 'tiny-bert')
sentiment_model = malaya.sentiment.transformer(model = 'tiny-bert')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")

@app.get("/test")
def home():
    return {"Hello": "FastAPI"}

@app.get("/emotion")
def form_get(request: Request):
    text = "Enter text"
    return templates.TemplateResponse('index.html', context={'request': request,'text': text})


@app.post("/emotion")
def form_post(request: Request, text: str = Form(...)):
    emotion_result = emotion_model.predict([text])
    sentiment_result = sentiment_model.predict([text])
    return templates.TemplateResponse('index.html', context={'request': request, 'emotion_result': emotion_result, 'sentiment_result': sentiment_result, 'text': text})