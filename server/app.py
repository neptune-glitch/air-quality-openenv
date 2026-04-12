import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from env import AirQualityEnv, Action

app = FastAPI()

def get_client():
    api_key  = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=api_base)

def get_agent_action(client, obs_dict, difficulty):
    prompt = f"""You are an AI agent controlling industrial emissions to reduce air pollution.

Current state:
- AQI: {obs_dict['aqi']} (target: {obs_dict['target_aqi']})
- Emission level: {obs_dict['emission']}
- Active factories: {obs_dict['factories']}
- Step: {obs_dict['step']}

Available actions:
- reduce_emission: lowers AQI by ~15
- shutdown_factory: lowers AQI by ~25 (only if factories > 0)
- increase_monitoring: lowers AQI by ~5
- do_nothing: AQI increases by 10 (bad)

Reply with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower()
        if action not in ["reduce_emission", "shutdown_factory", "increase_monitoring", "do_nothing"]:
            action = "reduce_emission"
        return action
    except Exception:
        if obs_dict["factories"] > 0 and obs_dict["aqi"] > 150:
            return "shutdown_factory"
        elif obs_dict["aqi"] > 105:
            return "reduce_emission"
        elif obs_dict["aqi"] > 80:
            return "increase_monitoring"
        return "do_nothing"

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

    client = get_client()
    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    obs_dict = obs.model_dump()

    output = []
    output.append(f"[START] task=air_quality env=custom model=gpt-3.5-turbo difficulty={difficulty}")

    done = False
    rewards = []

    while not done:
        action_str = get_agent_action(client, obs_dict, difficulty)
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