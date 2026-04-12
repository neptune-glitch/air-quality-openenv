import os
from flask import Flask, request
from env import AirQualityEnv, Action

app = Flask(__name__)

@app.route("/")
def run_env():
    difficulty = request.args.get("difficulty", "easy")
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
        output.append(f"[STEP] step={obs.step} action={action_str} aqi={obs.aqi} reward={reward.value:.2f} done={str(done).lower()} error={info['error'] or 'null'}")

    score = env.get_score()
    success = obs_dict["aqi"] < env.task["target_aqi"]
    output.append(f"[END] success={str(success).lower()} steps={obs.step} rewards={','.join([f'{r:.2f}' for r in rewards])}")
    output.append(f"Final AQI: {obs_dict['aqi']} | Target: {env.task['target_aqi']} | Score: {score:.4f}")

    return "<br>".join(output)

def main():
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    main()