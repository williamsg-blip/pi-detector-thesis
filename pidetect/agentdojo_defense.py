"""Week 11: the thesis detector as an AgentDojo defense.

AgentDojo calls `detect()` on every tool output (an email list, a file, a
bank statement) BEFORE the agent sees it. If it returns True, AgentDojo
replaces that output with "<Data omitted because a prompt injection was
detected>", so the agent never reads the injection (and also loses that
output's legitimate content, which is how false alarms hurt utility).

Scoring is identical to weeks 8-10: optional normalization, 200-word chunks,
document score = highest chunk score, flagged if score >= threshold (or if
any rule fires, when rules are on).

Build one from the week 10 operating point:
    detector = ThesisDetector.from_operating_point("deberta-v3-small")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pidetect.chunking import chunk_text  # noqa: E402
from pidetect.normalize import normalize as normalize_text  # noqa: E402
from pidetect.rules import fired_rules  # noqa: E402

OPERATING_POINTS = ROOT / "results" / "week10_operating_points.json"


class ThesisDetector(PromptInjectionDetector):
    def __init__(self, scorer, threshold: float, name: str, normalize: bool = False, rules: bool = False,
                 no_chunking: bool = False):
        super().__init__(mode="message", raise_on_injection=False)
        self.scorer = scorer
        self.threshold = threshold
        self.name = name
        self.normalize = normalize
        self.rules = rules
        self.no_chunking = no_chunking
        self.log: list[dict] = []  # one entry per tool output checked; reset for every agent run

    def score(self, text: str) -> tuple[float, list[str]]:
        cleaned = normalize_text(text) if self.normalize else text
        pieces = [cleaned] if self.no_chunking else chunk_text(cleaned)
        score = float(np.max(self.scorer(pieces))) if pieces else 0.0
        fired = fired_rules(text) if self.rules else []
        return score, fired

    def detect(self, tool_output: str) -> tuple[bool, float]:
        score, fired = self.score(tool_output)
        flagged = score >= self.threshold or bool(fired)
        self.log.append({"score": score, "flagged": flagged, "rules": fired})
        return flagged, score

    @classmethod
    def from_spec(cls, spec: str, threshold: float, normalize: bool = False, rules: bool = False,
                  no_chunking: bool = False, name: str | None = None) -> "ThesisDetector":
        from score_sets import make_scorer  # scripts/score_sets.py: keyword | tfidf | hf:<path>

        default_name, scorer = make_scorer(spec)
        return cls(scorer, threshold, name or default_name, normalize, rules, no_chunking)

    @classmethod
    def from_operating_point(cls, key: str, path: Path = OPERATING_POINTS) -> "ThesisDetector":
        points = json.loads(Path(path).read_text())
        if key not in points:
            raise SystemExit(f"No operating point '{key}' in {path}. Available: {', '.join(points)}. "
                             "Run scripts/analyze_week10.py after scoring that detector.")
        p = points[key]
        print(f"Detector '{key}': {p['detector']}, threshold {p['threshold']:.4g}, variant {p.get('variant')}")
        return cls.from_spec(p["detector"], p["threshold"], p["normalize"], p["rules"], p.get("no_chunking", False), key)
