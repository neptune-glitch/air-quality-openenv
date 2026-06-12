---
title: Air Quality Env
emoji: 🌫️
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# 🌫️ Air Quality Control Environment

An OpenEnv-compatible Reinforcement Learning environment where an AI agent manages industrial emissions to bring a city's AQI within safe limits.

## 🎯 Objective

The agent controls factory operations and emission strategies to reduce Air Quality Index (AQI) below a target threshold — simulating real-world urban pollution management.

## 🏙️ Motivation

Delhi-NCR regularly records AQI above 300 (hazardous). This environment trains agents to take optimal intervention decisions under pollution constraints — a critical real-world problem.

---

## ⚙️ Environment Details

### Observation Space
| Key | Description |
|---|---|
| `aqi` | Current Air Quality Index (0–500+) |
| `emission` | Industrial emission level (0–100) |
| `factories` | Number of active factories (0–5) |
| `difficulty` | Current task difficulty |

### Action Space (Discrete)
| Action | Effect |
|---|---|
| `reduce_emission` | Lowers emission by 10, AQI by ~15 |
| `shutdown_factory` | Shuts one factory, AQI by ~25 |
| `increase_monitoring` | Minor AQI reduction (~5) |
| `do_nothing` | AQI rises by 10 (penalty) |

### Reward Function
- Base reward per action (e.g. `+0.7` for shutdown)
- Proportional bonus: `+0.01 × (aqi_before − aqi_after)`
- Goal bonus: `+2.0` when target AQI is reached
- Penalty: `−0.5` for invalid/wasteful actions

---

## 🎮 Difficulty Levels

| Level | Target AQI | Description |
|---|---|---|
| `easy` | < 150 | Moderate reduction required |
| `medium` | < 100 | Significant intervention needed |
| `hard` | < 80 | Near-clean air — maximum challenge |

---

## 📋 Sample Output

```
[START] task=air_quality env=custom model=dummy difficulty=medium
[STEP] step=1 action=shutdown_factory aqi=230 reward=1.45 done=false error=null
[STEP] step=2 action=reduce_emission aqi=208 reward=0.72 done=false error=null
[STEP] step=3 action=reduce_emission aqi=184 reward=0.74 done=false error=null
[STEP] step=4 action=shutdown_factory aqi=152 reward=1.52 done=false error=null
[STEP] step=5 action=increase_monitoring aqi=140 reward=0.32 done=false error=null
[STEP] step=6 action=reduce_emission aqi=118 reward=0.72 done=false error=null
[STEP] step=7 action=reduce_emission aqi=96 reward=2.72 done=true error=null
[END] success=true steps=7 rewards=1.45,0.72,0.74,1.52,0.32,0.72,2.72
Final AQI: 96 | Target: 100 | Score: 0.6214
```

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python inference.py
```

Then visit: `http://localhost:7860`

Optional difficulty param: `http://localhost:7860/?difficulty=hard`

---

## 🐳 Docker

```bash
docker build -t air-quality-env .
docker run -p 7860:7860 air-quality-env
```

---

## 🗂️ File Structure

```
├── env.py          # Core RL environment
├── inference.py    # Flask server + episode runner
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔮 Future Work

- Integrate trained PPO/DQN agent from Stable-Baselines3
- Add multi-city grid observation
- Add wind/weather as dynamic state variables
- Publish trained model weights to Hugging Face Hub
