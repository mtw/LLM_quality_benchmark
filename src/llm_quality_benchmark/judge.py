from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

JUDGE_RUBRIC = """You are evaluating the quality of an LLM response.

You will be given:
1. The original task
2. A candidate response

Evaluate the response strictly and conservatively.

Scoring scale:
1 = very poor
2 = weak
3 = acceptable
4 = strong
5 = excellent

Score the candidate on:
- instruction_following
- correctness
- completeness
- clarity
- actionability
- hallucination_risk

Interpretation of hallucination_risk:
5 = very low hallucination risk
1 = very high hallucination risk

Also provide:
- overall: integer 1-5
- verdict: one of ["excellent", "strong", "acceptable", "weak", "poor"]
- pass_fail: one of ["pass", "fail"]
- notes: short explanation, max 120 words

Important rules:
- Penalize failure to follow requested format.
- Penalize invented facts or unjustified claims.
- Penalize evasiveness if the task requested a concrete answer.
- For coding tasks, correctness and completeness matter most.
- For scientific summaries, correctness and hallucination_risk matter most.
- For shell/debugging tasks, actionability and correctness matter most.
- Be strict.

Return ONLY valid JSON matching exactly this schema:
{
  "instruction_following": 1,
  "correctness": 1,
  "completeness": 1,
  "clarity": 1,
  "actionability": 1,
  "hallucination_risk": 1,
  "overall": 1,
  "verdict": "acceptable",
  "pass_fail": "pass",
  "notes": "..."
}
"""

TASK_TYPE_HINTS = {
    "coding": "This is a coding task. Correctness, completeness, and constraint adherence are especially important.",
    "summary": "This is a scientific or technical summary task. Fidelity, precision, and non-hallucination are especially important.",
    "reasoning": "This is a reasoning task. Logical structure, prioritization, and correctness are especially important.",
    "shell": "This is a shell/debugging task. Actionable next steps and correct interpretation are especially important.",
    "general": "This is a general task. Evaluate normally.",
}

_KNOWN_TASK_TYPES = set(TASK_TYPE_HINTS.keys())

_VERDICTS = {"excellent", "strong", "acceptable", "weak", "poor"}
_PASS_FAIL = {"pass", "fail"}


@dataclass(frozen=True)
class JudgeScore:
    instruction_following: int
    correctness: int
    completeness: int
    clarity: int
    actionability: int
    hallucination_risk: int
    overall: int
    verdict: str
    pass_fail: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_following": self.instruction_following,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "actionability": self.actionability,
            "hallucination_risk": self.hallucination_risk,
            "overall": self.overall,
            "verdict": self.verdict,
            "pass_fail": self.pass_fail,
            "notes": self.notes,
        }


def parse_prompt_front_matter(prompt_text: str) -> dict[str, str]:
    """
    Very small front-matter parser.

    Supports Markdown files starting with:
    ---
    task_type: coding
    ---

    Only parses simple `key: value` lines; everything else is ignored.
    """
    if not prompt_text.startswith("---\n"):
        return {}

    end = prompt_text.find("\n---\n", 4)
    if end == -1:
        return {}

    block = prompt_text[4:end]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key and value:
            meta[key] = value
    return meta


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def detect_task_type(prompt_path: Path, prompt_text: str) -> str:
    meta = parse_prompt_front_matter(prompt_text)
    declared = (meta.get("task_type") or meta.get("type") or "").strip().lower()
    if declared in _KNOWN_TASK_TYPES:
        return declared

    name = prompt_path.name.lower()
    text = prompt_text.lower()

    if "coding" in name or "return only code" in text or "python function" in text:
        return "coding"
    if "summary" in name or "summarize" in text or "main findings" in text:
        return "summary"
    if "shell" in name or "log output" in text or "next 5 shell commands" in text:
        return "shell"
    if "reasoning" in name or "rank them" in text or "most likely causes" in text:
        return "reasoning"
    return "general"


def build_judge_prompt(task_text: str, candidate_text: str, task_type: str) -> str:
    hint = TASK_TYPE_HINTS.get(task_type, TASK_TYPE_HINTS["general"])
    return (
        f"{JUDGE_RUBRIC}\n\n"
        f"Task-type hint:\n{hint}\n\n"
        f"Original task:\n{task_text}\n\n"
        f"Candidate response:\n{candidate_text}\n"
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first JSON object embedded in surrounding text, without using a
    # greedy regex that may capture too much (or too little) when braces appear
    # in model chatter.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not find JSON object in judge output:\n{text}")


def validate_score(data: dict[str, Any]) -> JudgeScore:
    required_ints = [
        "instruction_following",
        "correctness",
        "completeness",
        "clarity",
        "actionability",
        "hallucination_risk",
        "overall",
    ]
    required_strs = ["verdict", "pass_fail", "notes"]

    for key in required_ints:
        if key not in data:
            raise ValueError(f"Judge JSON missing key: {key}")
        if not isinstance(data[key], int):
            raise ValueError(f"Judge JSON key '{key}' must be int, got {type(data[key])}")
        if not (1 <= data[key] <= 5):
            raise ValueError(f"Judge JSON key '{key}' must be between 1 and 5")

    for key in required_strs:
        if key not in data:
            raise ValueError(f"Judge JSON missing key: {key}")
        if not isinstance(data[key], str):
            raise ValueError(f"Judge JSON key '{key}' must be str, got {type(data[key])}")

    if data["verdict"] not in _VERDICTS:
        raise ValueError(f"Invalid verdict: {data['verdict']}")

    if data["pass_fail"] not in _PASS_FAIL:
        raise ValueError(f"Invalid pass_fail: {data['pass_fail']}")

    return JudgeScore(
        instruction_following=data["instruction_following"],
        correctness=data["correctness"],
        completeness=data["completeness"],
        clarity=data["clarity"],
        actionability=data["actionability"],
        hallucination_risk=data["hallucination_risk"],
        overall=data["overall"],
        verdict=data["verdict"],
        pass_fail=data["pass_fail"],
        notes=data["notes"],
    )
