import os
import json
import random
from flask import Flask, request
from openai import OpenAI
from env import AirQualityEnv, Action, VALID_ACTIONS

app = Flask(__name__)

# ─── OpenAI Agent ─────────────────────────────────────────────────────────────

def get_agent_action(client: OpenAI, obs: dict, difficulty: str) -> str:
    """Ask the LLM to pick the best action given current observation."""
    prompt = f"""You are an AI agent controlling industrial emissions to reduce air pollution.

Current environment state:
- AQI (Air Quality Index): {obs['aqi']} (lower is better)
- Target AQI to achieve: {obs['target_aqi']}
- Emission level: {obs['emission']}
- Active factories: {obs['factories']}
- Step: {obs['step']}
- Difficulty: {difficulty}

Available actions:
- reduce_emission: reduces emission by 10, lowers AQI by ~15
- shutdown_factory: shuts one factory, lowers AQI by ~25 (use when factories > 0)
- increase_monitoring: small AQI reduction (~5), low impact
- do_nothing: AQI increases by 10 (bad)

Choose the single best action to bring AQI below {obs['target_aqi']}.
Reply with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower()
        # Validate — fall back to reduce_emission if LLM hallucinates
        if action not in VALID_ACTIONS:
            action = "reduce_emission"
        return action
    except Exception:
        # Fallback rule-based agent if API fails
        if obs_dict["aqi"] > 200 and obs_dict["factories"] > 0:
            action_str = "shutdown_factory"
        elif obs_dict["aqi"] > 120:
            action_str = "reduce_emission"
        elif obs_dict["aqi"] > 80:
            action_str = "increase_monitoring"
        else:
            action_str = "do_nothing"


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(difficulty: str, use_llm: bool) -> list[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client  = OpenAI(api_key=api_key) if api_key else None

    env  = AirQualityEnv(difficulty=difficulty)
    obs  = env.reset()
    obs_dict = obs.model_dump()

    output  = []
    rewards = []
    done    = False
    step    = 0

    model_name = "gpt-3.5-turbo" if (client and use_llm) else "rule-based"
    output.append(f"[START] task=air_quality env=custom model={model_name} difficulty={difficulty}")

    while not done and step < env.task["max_steps"]:
        if client and use_llm:
            action_str = get_agent_action(client, obs_dict, difficulty)
        else:
            # Fallback rule-based
            if obs_dict["factories"] > 0 and obs_dict["aqi"] > 150:
                action_str = "shutdown_factory"
            elif obs_dict["aqi"] > 105:
                action_str = "reduce_emission"
            elif obs_dict["aqi"] > 80:
                action_str = "increase_monitoring"
            else:
                action_str = "do_nothing"

        action = Action(action=action_str)
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()

        output.append(
            f"[STEP] step={obs.step} action={action_str} "
            f"aqi={obs.aqi} reward={reward.value:.2f} "
            f"done={str(done).lower()} error={info['error'] or 'null'}"
        )
        rewards.append(reward.value)
        step += 1

    score   = env.get_score()
    success = obs_dict["aqi"] < env.task["target_aqi"]

    output.append(
        f"[END] success={str(success).lower()} steps={step} "
        f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
    )
    output.append(
        f"Final AQI: {obs_dict['aqi']} | Target: {env.task['target_aqi']} | Score: {score:.4f}"
    )
    return output


# ─── Flask App ────────────────────────────────────────────────────────────────

@app.route("/")
def run_env():
    difficulty = request.args.get("difficulty", "easy")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "easy"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    use_llm = bool(api_key)

    output = run_episode(difficulty, use_llm)
    return "<br>".join(output)


@app.route("/score")
def score_all():
    """Run all 3 tasks and return JSON scores — used by grader."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    use_llm = bool(api_key)

    results = {}
    for diff in ["easy", "medium", "hard"]:
        env  = AirQualityEnv(difficulty=diff)
        obs  = env.reset()
        done = False

        while not done:
            obs_dict = obs.model_dump()
            if use_llm:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                action_str = get_agent_action(client, obs_dict, diff)
            else:
                if obs_dict["factories"] > 0 and obs_dict["aqi"] > 150:
                    action_str = "shutdown_factory"
                elif obs_dict["aqi"] > 105:
                    action_str = "reduce_emission"
                elif obs_dict["aqi"] > 80:
                    action_str = "increase_monitoring"
                else:
                    action_str = "do_nothing"

            obs, reward, done, _ = env.step(Action(action=action_str))

        results[diff] = {
            "score":   env.get_score(),
            "success": obs.aqi < env.task["target_aqi"],
            "final_aqi": obs.aqi,
            "target_aqi": env.task["target_aqi"],
        }

    return results

# ─── OpenEnv Required Endpoints ───────────────────────────────────────────────

env_instance = None

@app.route("/reset", methods=["POST"])
def reset_env():
    global env_instance
    data = request.get_json(silent=True) or {}
    difficulty = data.get("difficulty", "easy")

    env_instance = AirQualityEnv(difficulty=difficulty)
    obs = env_instance.reset()

    return obs.model_dump()


@app.route("/step", methods=["POST"])
def step_env():
    global env_instance

    if env_instance is None:
        return {"error": "Environment not initialized. Call /reset first."}, 400

    data = request.get_json(silent=True) or {}
    action_str = data.get("action", "reduce_emission")
   

    obs, reward, done, info = env_instance.step(Action(action=action_str))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.route("/state", methods=["GET"])
def get_state():
    global env_instance

    if env_instance is None:
        return {"error": "Environment not initialized."}, 400

    return env_instance.state().model_dump()


if __name__ == "__main__":
    import sys
    
    # If called directly - run episode and print output (for validator)
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "easy"
    
    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    obs_dict = obs.model_dump()
    initial_aqi = obs_dict["aqi"]
    
    print(f"[START] task=air_quality env=custom model=rule-based difficulty={difficulty}")
    
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
        
        action = Action(action=action_str)
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        rewards.append(reward.value)
        
        print(f"[STEP] step={obs.step} action={action_str} aqi={obs.aqi} reward={reward.value:.2f} done={str(done).lower()} error={info['error'] or 'null'}")
    
    score = env.get_score()
    success = obs_dict["aqi"] < env.task["target_aqi"]
    
    print(f"[END] success={str(success).lower()} steps={obs.step} rewards={','.join([f'{r:.2f}' for r in rewards])}")
    print(f"Final AQI: {obs_dict['aqi']} | Target: {env.task['target_aqi']} | Score: {score:.4f}")

else:
    # When imported by server/app.py - run as Flask server
    pass