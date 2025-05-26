from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FinAI backend está funcionando 🚀"}


@app.get("/seila")
def read_root():
    return {"message": "FinAI DALEEEEEEEEEE"}
