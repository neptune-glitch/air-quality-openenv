---
title: Air Quality Env
emoji: 🌫️
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# 🌫️ Air Quality Control — OpenEnv Environment

An **OpenEnv-compliant** Reinforcement Learning environment where an AI agent manages industrial emissions to bring a city's Air Quality Index (AQI) within safe limits.

Simulates real decisions made by city pollution control boards — particularly relevant to Delhi-NCR, one of the world's most polluted urban regions.

---

## 🎯 Objective

The agent observes the current AQI and pollution state, then takes actions (shutting factories, reducing emissions) to bring AQI below a target threshold before running out of steps.

---

## ⚙️ OpenEnv Interface

| Method | Description |
|---|---|
| `reset()` | Returns initial `Observation` |
| `step(action)` | Returns `(Observation, Reward, done, info)` |
| `state()` | Returns current `Observation` |
| `get_score()` | Returns float `0.0–1.0` |

All inputs/outputs use **Pydantic typed models** (`Observation`, `Action`, `Reward`).

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `aqi` | float | Current Air Quality Index |
| `emission` | float | Industrial emission level (0–100) |
| `factories` | int | Active factories (0–5) |
| `step` | int | Current timestep |
| `difficulty` | str | Task level (easy/medium/hard) |
| `target_aqi` | float | Goal AQI to reach |

## ⚡ Action Space (Discrete)

| Action | Effect |
|---|---|
| `reduce_emission` | Lowers emission by 10, AQI by ~15 |
| `shutdown_factory` | Shuts one factory, AQI by ~25 |
| `increase_monitoring` | Minor AQI reduction (~5) |
| `do_nothing` | AQI rises by 10 (penalty) |

---

## 🏆 Tasks & Grading

| Task | Target AQI | Start AQI Range | Max Steps |
|---|---|---|---|
| Easy | < 150 | 180–250 | 10 |
| Medium | < 100 | 200–280 | 10 |
| Hard | < 80 | 220–300 | 10 |

Score = `(initial_aqi - final_aqi) / initial_aqi`, clamped to `[0.0, 1.0]`.

---

## 💰 Reward Function

- **Per-step base reward**: based on action quality (`+0.7` for shutdown, `+0.5` for reduce, etc.)
- **Proportional bonus**: `+0.01 × aqi_drop` — rewards actual improvement
- **Goal bonus**: `+0.3` when target AQI is reached
- **Penalties**: `-0.5` for invalid actions, `-0.3` for do_nothing
- All rewards normalised to `[0.0, 1.0]`

---

## 📋 Sample Output

```
[START] task=air_quality env=custom model=gpt-3.5-turbo difficulty=medium
[STEP] step=1 action=shutdown_factory aqi=221 reward=0.93 done=false error=null
[STEP] step=2 action=shutdown_factory aqi=198 reward=0.90 done=false error=null
[STEP] step=3 action=reduce_emission aqi=182 reward=0.63 done=false error=null
[STEP] step=4 action=reduce_emission aqi=166 reward=0.66 done=false error=null
[STEP] step=5 action=reduce_emission aqi=152 reward=0.64 done=false error=null
[STEP] step=6 action=reduce_emission aqi=138 reward=0.64 done=false error=null
[STEP] step=7 action=reduce_emission aqi=122 reward=0.66 done=false error=null
[STEP] step=8 action=reduce_emission aqi=97 reward=1.00 done=true error=null
[END] success=true steps=8 rewards=0.93,0.90,0.63,0.66,0.64,0.64,0.66,1.00
Final AQI: 97 | Target: 100 | Score: 0.5832
```

---

## 🚀 Setup & Run

### Local
```bash
pip install -r requirements.txt
python inference.py
```
Visit: `http://localhost:7860/?difficulty=medium`

### With OpenAI Agent
```bash
export OPENAI_API_KEY=your_key_here
python inference.py
```

### Score all 3 tasks
```
GET /score
```
Returns JSON with scores for easy, medium, hard.

### Docker
```bash
docker build -t air-quality-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key air-quality-env
```

---

## 🗂️ File Structure

```
├── env.py            # Core RL environment (Pydantic models + logic)
├── inference.py      # Flask server + OpenAI agent runner
├── openenv.yaml      # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔮 Future Work

- Train PPO/DQN agent via Stable-Baselines3
- Multi-zone city grid observation
- Wind/weather as dynamic state variables
- Publish trained weights to Hugging Face Hub
