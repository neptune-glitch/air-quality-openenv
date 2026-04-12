def run_episode(client, difficulty="easy"):
    env = AirQualityEnv(difficulty=difficulty)
    obs = env.reset()
    obs_dict = obs.model_dump()

    print(f"[START] task=air_quality env=custom model=gpt-3.5-turbo difficulty={difficulty}")

    done = False
    rewards = []

    while not done:
        action_str = get_agent_action(client, obs_dict, difficulty)
        obs, reward, done, info = env.step(Action(action=action_str))
        obs_dict = obs.model_dump()
        # Clamp reward strictly between 0 and 1
        r = max(0.001, min(0.999, reward.value))
        rewards.append(r)  # ← store clamped value
        print(f"[STEP] step={obs.step} action={action_str} aqi={obs.aqi} reward={r:.4f} done={str(done).lower()} error={info['error'] or 'null'}")

    raw_score = env.get_score()
    score = max(0.001, min(0.999, raw_score))
    success = obs_dict["aqi"] < env.task["target_aqi"]
    print(f"[END] success={str(success).lower()} steps={obs.step} rewards={','.join([f'{r:.4f}' for r in rewards])}")
    print(f"Final AQI: {obs_dict['aqi']} | Target: {env.task['target_aqi']} | Score: {score:.4f}")