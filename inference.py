import sys
import os
from env import AirQualityEnv, Action

def run_episode(difficulty: str = "easy"):
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

if __name__ == "__main__":
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_episode(difficulty)
    print("---")
    run_episode("medium")
    print("---")
    run_episode("hard")