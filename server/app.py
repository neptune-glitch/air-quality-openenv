from fastapi import FastAPI
from env import AirQualityEnv, Action, Observation

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(difficulty: str = "easy"):
    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    return {"message": "ok"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()