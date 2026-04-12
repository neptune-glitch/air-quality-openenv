from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import AirQualityEnv, Action

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(difficulty: str = "easy"):
    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    return obs.model_dump()

@app.get("/")
def run_env(difficulty: str = "easy"):
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "easy"

    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    obs_dict = obs.model_dump()

    output = []
    output.append(f"[START] task=air_quality env=custom model=rule-based difficulty={difficulty}")

    done = False
    rewards = []

    while not done:
        if obs_dict["factories"] > 0 and obs_dict["aqi"] > 150:
            action_str = "shutdown_factory"
        elif obs_dict["aqi"] > 105:
            action_str = "reduce_emission"
        elif obs_dict["aqi"] > 80:
            action_str = "increase_monitoring"
        else:
            action_str = "do_nothing"

        obs, reward, done, info = env.step(Action(action=action_str))
        obs_dict = obs.model_dump()
        rewards.append(reward.value)
        output.append(
            f"[STEP] step={obs.step} action={action_str} "
            f"aqi={obs.aqi} reward={reward.value:.2f} "
            f"done={str(done).lower()} error={info['error'] or 'null'}"
        )

    score = env.get_score()
    success = obs_dict["aqi"] < env.task["target_aqi"]
    output.append(f"[END] success={str(success).lower()} steps={obs.step} rewards={','.join([f'{r:.2f}' for r in rewards])}")
    output.append(f"Final AQI: {obs_dict['aqi']} | Target: {env.task['target_aqi']} | Score: {score:.4f}")

    return HTMLResponse("<br>".join(output))
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()