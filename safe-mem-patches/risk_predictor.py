import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI


DEFAULT_RISK_PROMPT_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "prompts" / "risk.txt"
)
EMPTY_SCENE_GRAPH_STATE = (
    "No explicit scene-graph relations were extracted for the current frame."
)


class RiskPredictorError(RuntimeError):
    pass


class RiskReplanState:
    """Tracks one rejected, unexecuted proposal without polluting history."""

    def __init__(self):
        self.risk_source = None

    @property
    def pending(self):
        return self.risk_source is not None

    def reject(self, tracker, current_step, plan, risk_source):
        if not tracker.plans:
            raise RuntimeError("Cannot reject a plan that was not tracked")
        tracked = tracker.plans[-1]
        if tracked['plan']['action'] != plan['action']:
            raise RuntimeError("Rejected plan does not match the latest proposal")

        rejected_step = tracked['step']
        tracker.plans.pop()
        if (
            tracker.raw_outputs
            and tracker.raw_outputs[-1].get('step') == rejected_step
        ):
            tracker.raw_outputs.pop()
        self.risk_source = risk_source
        return max(0, current_step - 1)

    def accept(self):
        self.risk_source = None

    def forbids_done(self, operator):
        return self.pending and str(operator).lower() == 'done'

    def append_prompt(self, prompt):
        if not self.pending:
            return prompt
        return prompt + "\n\n" + f"""
[MANDATORY RISK REPLAN]
The previous proposed action was rejected by the risk predictor and was NOT
executed. The environment state and completed-action history therefore did not
advance. Propose a different applicable atomic action that avoids the stated
risk.

Risk predictor reason: {self.risk_source}

You MUST NOT return DONE in this replan because the rejected action did not
complete the task. Return DONE only after a later action has actually been
accepted and executed and the task goal is satisfied.
""".strip()


@dataclass(frozen=True)
class RiskPrediction:
    checked: bool
    safe: bool
    risk_source: str
    probability: float
    random_value: float
    model: str
    raw_output: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def parse_risk_response(raw_output: str):
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise RiskPredictorError("Risk predictor returned an empty response")

    candidate = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        json_object = re.search(r"\{.*\}", candidate, re.DOTALL)
        if json_object:
            candidate = json_object.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise RiskPredictorError(
            f"Risk predictor did not return valid JSON: {raw_output!r}"
        ) from error

    if not isinstance(payload, dict):
        raise RiskPredictorError("Risk predictor JSON must be an object")
    if type(payload.get("safe")) is not bool:
        raise RiskPredictorError("Risk predictor JSON field 'safe' must be boolean")
    risk_source = payload.get("risk_source")
    if not isinstance(risk_source, str) or not risk_source.strip():
        raise RiskPredictorError(
            "Risk predictor JSON field 'risk_source' must be a non-empty string"
        )
    return payload["safe"], risk_source.strip()


class RiskPredictor:
    def __init__(
        self,
        model="gpt-4o",
        probability=0.8,
        prompt_path=DEFAULT_RISK_PROMPT_PATH,
        random_seed=None,
        client=None,
        random_source: Optional[Callable[[], float]] = None,
        retries=3,
        gate_mode="off",
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("risk check probability must be between 0 and 1")
        if retries < 1:
            raise ValueError("risk predictor retries must be at least 1")
        if gate_mode not in ("off", "pure", "hybrid"):
            raise ValueError("gate_mode must be one of 'off', 'pure', 'hybrid'")

        self.model = model
        self.probability = float(probability)
        self.prompt_path = Path(prompt_path).expanduser().resolve()
        self._client = client
        self._random = random_source or random.Random(random_seed).random
        self.retries = retries
        self.gate_mode = gate_mode

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE")
            if not api_key:
                raise RiskPredictorError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=api_key, base_url=api_base or None)
        return self._client

    def _render_prompt(self, state_desc, action_line):
        if not self.prompt_path.is_file():
            raise RiskPredictorError(
                f"Risk predictor prompt not found: {self.prompt_path}"
            )
        template = self.prompt_path.read_text(encoding="utf-8")
        try:
            return template.format(
                state_desc=state_desc.strip(), action_line=action_line.strip()
            )
        except (KeyError, ValueError) as error:
            raise RiskPredictorError(
                f"Invalid risk predictor prompt template: {self.prompt_path}"
            ) from error

    def _call_model(self, prompt):
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                completion = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=4096,
                )
                output = completion.choices[0].message.content
                if output:
                    return output
                last_error = RiskPredictorError(
                    "Risk predictor returned an empty response"
                )
            except Exception as error:
                last_error = error
            if attempt < self.retries:
                time.sleep(attempt)
        raise RiskPredictorError(
            f"Risk predictor model call failed after {self.retries} attempts: "
            f"{last_error}"
        ) from last_error

    def _apply_symbolic_gate(self, scene_graph_path, action, random_value):
        """Direction A: deterministic symbolic gate over scene-graph text + action.

        Runs before the stochastic η skip. Returns a final ``RiskPrediction`` when
        the gate makes a decision, else ``None`` so the hybrid wrapper falls through
        to the LLM risk predictor.
        """
        from og_ego_prim.symbolic_gate import evaluate

        state_desc = ""
        try:
            path = Path(scene_graph_path).expanduser().resolve()
            if path.is_file():
                state_desc = path.read_text(encoding="utf-8").strip()
        except OSError:
            state_desc = ""

        decision = evaluate(state_desc, action)
        if decision.confident:
            return RiskPrediction(
                checked=True,
                safe=decision.safe,
                risk_source=(
                    f"[symbolic-gate:{decision.rule or 'default'}] "
                    f"{decision.risk_source}"
                ),
                probability=1.0,
                random_value=random_value,
                model="symbolic-gate",
            )
        if self.gate_mode == "pure":
            return RiskPrediction(
                checked=True,
                safe=True,
                risk_source=(
                    "[symbolic-gate] action outside the covered 17-primitive domain; "
                    "no explicit relation chain -> SAFE"
                ),
                probability=1.0,
                random_value=random_value,
                model="symbolic-gate",
            )
        return None

    def predict(self, scene_graph_path, action):
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")

        random_value = float(self._random())
        if not 0.0 <= random_value < 1.0:
            raise ValueError("random source must return a value in [0, 1)")
        if action.strip().upper().startswith("DONE"):
            return RiskPrediction(
                checked=False,
                safe=True,
                risk_source="Terminal DONE action bypasses physical risk prediction",
                probability=self.probability,
                random_value=random_value,
                model=self.model,
            )
        if self.gate_mode in ("pure", "hybrid"):
            gate_result = self._apply_symbolic_gate(
                scene_graph_path, action, random_value
            )
            if gate_result is not None:
                return gate_result
        if random_value >= self.probability:
            return RiskPrediction(
                checked=False,
                safe=True,
                risk_source=(
                    "Probability bypass: risk check was skipped "
                    f"({random_value:.6f} >= {self.probability:.6f})"
                ),
                probability=self.probability,
                random_value=random_value,
                model=self.model,
            )

        scene_graph_path = Path(scene_graph_path).expanduser().resolve()
        if not scene_graph_path.is_file():
            raise RiskPredictorError(
                f"Current scene graph not found: {scene_graph_path}"
            )
        state_desc = scene_graph_path.read_text(encoding="utf-8").strip()
        if not state_desc:
            state_desc = EMPTY_SCENE_GRAPH_STATE

        prompt = self._render_prompt(state_desc, action)
        raw_output = self._call_model(prompt)
        safe, risk_source = parse_risk_response(raw_output)
        return RiskPrediction(
            checked=True,
            safe=safe,
            risk_source=risk_source,
            probability=self.probability,
            random_value=random_value,
            model=self.model,
            raw_output=raw_output,
        )
