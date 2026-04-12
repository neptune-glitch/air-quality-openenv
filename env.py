import random
from pydantic import BaseModel

# ─── Typed Models ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    aqi: float
    emission: float
    factories: int
    step: int
    difficulty: str
    target_aqi: float

class Action(BaseModel):
    action: str  # reduce_emission | shutdown_factory | increase_monitoring | do_nothing

class Reward(BaseModel):
    value: float
    done: bool
    success: bool
    info: dict

# ─── Task Definitions ─────────────────────────────────────────────────────────

TASKS = {
    "easy":   {"target_aqi": 150, "max_steps": 10, "start_aqi": (180, 250)},
    "medium": {"target_aqi": 100, "max_steps": 10, "start_aqi": (200, 280)},
    "hard":   {"target_aqi": 80,  "max_steps": 10, "start_aqi": (220, 300)},
}

VALID_ACTIONS = ["reduce_emission", "shutdown_factory", "increase_monitoring", "do_nothing"]

# ─── Environment ──────────────────────────────────────────────────────────────

class AirQualityEnv:
    def __init__(self, difficulty: str = "easy"):
        if difficulty not in TASKS:
            raise ValueError(f"difficulty must be one of {list(TASKS.keys())}")
        self.difficulty = difficulty
        self.task = TASKS[difficulty]
        self._state: dict = {}
        self._steps: int = 0
        self._initial_aqi: float = 0.0

    def reset(self) -> Observation:
        lo, hi = self.task["start_aqi"]
        self._state = {
            "aqi":       float(random.randint(lo, hi)),
            "emission":  float(random.randint(60, 100)),
            "factories": 5,
        }
        self._steps = 0
        self._initial_aqi = self._state["aqi"]
        return self._make_obs()

    def state(self) -> Observation:
        return self._make_obs()

    def step(self, action: Action):
        act = action.action
        aqi_before = self._state["aqi"]
        reward_val = 0.0
        error = None

        if act == "reduce_emission":
            self._state["emission"] = max(0.0, self._state["emission"] - 10)
            self._state["aqi"] -= 15
            reward_val = 0.5

        elif act == "shutdown_factory":
            if self._state["factories"] > 0:
                self._state["factories"] -= 1
                self._state["aqi"] -= 25
                reward_val = 0.7
            else:
                reward_val = -0.5
                error = "no_factories_left"

        elif act == "increase_monitoring":
            self._state["aqi"] -= 5
            reward_val = 0.2

        elif act == "do_nothing":
            self._state["aqi"] += 10
            reward_val = -0.3

        else:
            reward_val = -0.5
            error = f"invalid_action:{act}"

        self._state["aqi"] += 2


        # Clamp values
        self._state["aqi"]      = max(0.0, self._state["aqi"])
        self._state["emission"] = max(0.0, self._state["emission"])

        # Proportional bonus for actual AQI drop
        aqi_drop = aqi_before - self._state["aqi"]
        reward_val += aqi_drop * 0.01

        # Normalise to 0.0–1.0
        reward_val = round(max(0.0, min(1.0, reward_val)), 4)

        self._steps += 1
        target  = self.task["target_aqi"]
        success = self._state["aqi"] < target
        done    = success or self._steps >= self.task["max_steps"]

        if success:
            reward_val = min(1.0, reward_val + 0.3)

        obs    = self._make_obs()
        reward = Reward(
            value   = reward_val,
            done    = done,
            success = success,
            info    = {"error": error, "aqi_drop": round(aqi_drop, 2)},
        )
        return obs, reward, done, {"error": error}

    def get_score(self) -> float:
        reduction = (self._initial_aqi - self._state["aqi"]) / max(1, self._initial_aqi)
        return round(max(0.0, min(1.0, reduction)), 4)

    def _make_obs(self) -> Observation:
        return Observation(
            aqi        = round(self._state["aqi"], 2),
            emission   = round(self._state["emission"], 2),
            factories  = self._state["factories"],
            step       = self._steps,
            difficulty = self.difficulty,
            target_aqi = self.task["target_aqi"],
        )
