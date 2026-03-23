from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ Better model
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # CPU (Render compatible)
)

class Request(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API running"}

@app.post("/generate")
def generate(req: Request):
    
    # ✅ Add instruction wrapper (VERY IMPORTANT)
    formatted_prompt = f"""
        Follow instructions strictly.
        TASK:
        {req.prompt}
    """

    result = generator(
        formatted_prompt,
        max_length=400,
        do_sample=True,
        temperature=0.7
    )

    return {
        "output": result[0]["generated_text"]
    }