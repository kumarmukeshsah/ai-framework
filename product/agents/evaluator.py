"""Multi-stage candidate evaluation agent.

Pipeline stages:
1. **Parse** — Extract structured info from raw transcript
2. **Evaluate** — Score candidate against rubric using LLM
3. **Report** — Generate final structured evaluation

Each stage can operate in LLM mode or rule-based mode.
"""
from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from product.core.errors import AgentException
from product.core.telemetry import track_agent_execution, track_llm_call, span
from product.providers.base import LLMProvider, Message
from product.agents.base import BaseAgent
from product.models.candidate import (
    CandidateEvaluation,
    EvaluationPipelineResult,
    EvaluationStageResult,
    RubricBreakdown,
)


class EvaluationStage(str, Enum):
    """Stages in the evaluation pipeline."""

    PARSE = "parse"
    EVALUATE = "evaluate"
    REPORT = "report"


# ── Keyword-based fallback constants ──────────────────────────────────────

SENIORITY_KEYWORDS: Dict[str, List[str]] = {
    "Senior": ["lead", "architect", "8+ years", "10+ years", "mentored", "team lead", "staff"],
    "Mid": ["4-7 years", "experienced", "independent", "contribute", "mid-level"],
    "Junior": ["junior", "0-3 years", "entry", "new", "learning", "fresh", "intern"],
}

SKILL_PATTERNS: Dict[str, str] = {
    "python": r"\bpython\b",
    "typescript": r"\btypescript\b|ts\b(?!\w)",
    "javascript": r"\bjavascript\b|js\b(?!\w)",
    "react": r"\breact\b",
    "fastapi": r"\bfastapi\b",
    "django": r"\bdjango\b",
    "kubernetes": r"\bkubernetes\b|k8s",
    "docker": r"\bdocker\b",
    "aws": r"\baws\b|amazon\s+web\s+services",
    "gcp": r"\bgcp\b|google\s+cloud",
    "azure": r"\bazure\b",
    "sql": r"\bsql\b|postgresql|mysql",
    "mongodb": r"\bmongodb\b|mongo",
    "redis": r"\bredis\b",
    "graphql": r"\bgraphql\b",
    "machine learning": r"\bmachine\s+learning\b|ml\b(?!\w)",
}

RECOMMENDATION_THRESHOLDS = [
    (7.5, "Hire", "Strong candidate who meets or exceeds requirements."),
    (5.0, "Consider", "Candidate has potential but may need additional support."),
    (0.0, "Reject", "Candidate does not meet minimum requirements."),
]

SYSTEM_PROMPT = """You are an expert technical interviewer evaluating candidates.
Analyze transcripts carefully and provide structured, fair evaluations."""


class EvaluatorAgent(BaseAgent):
    """Multi-stage candidate evaluation agent.

    Supports:
    - LLM-powered evaluation (when provider is set)
    - Rule-based fallback evaluation (for testing/offline use)
    - Multi-stage pipeline with intermediate results
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        use_llm: bool = False,
    ) -> None:
        super().__init__(
            name="EvaluatorAgent",
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
        )
        self._use_llm = use_llm and provider is not None

    @track_agent_execution(agent_name="EvaluatorAgent")
    async def process(self, transcript: str, context: Optional[str] = None) -> EvaluationPipelineResult:
        """Run the full evaluation pipeline.

        Args:
            transcript: The interview transcript to evaluate.
            context: Optional job context / requirements.

        Returns:
            EvaluationPipelineResult with the full pipeline trace.
        """
        start = time.monotonic()
        stages: List[EvaluationStageResult] = []

        try:
            # Stage 1: Parse
            with span("evaluator.parse"):
                parse_result = await self._run_parse_stage(transcript, context)
                stages.append(parse_result)

            # Stage 2: Evaluate
            with span("evaluator.evaluate"):
                eval_result = await self._run_evaluate_stage(
                    transcript, context, parse_result
                )
                stages.append(eval_result)

            # Stage 3: Report
            with span("evaluator.report"):
                report_result = await self._run_report_stage(
                    transcript, context, parse_result, eval_result
                )
                stages.append(report_result)

            duration = (time.monotonic() - start) * 1000
            return EvaluationPipelineResult(
                success=True,
                evaluation=report_result.data.get("evaluation"),
                stages=stages,
                duration_ms=round(duration, 1),
            )

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return EvaluationPipelineResult(
                success=False,
                error=str(e),
                stages=stages,
                duration_ms=round(duration, 1),
            )

    async def _run_parse_stage(
        self, transcript: str, context: Optional[str]
    ) -> EvaluationStageResult:
        try:
            if self._use_llm and self.provider:
                data = await self._llm_parse(transcript, context)
            else:
                data = self._rule_parse(transcript)
            return EvaluationStageResult(stage_name="parse", success=True, data=data)
        except Exception as e:
            return EvaluationStageResult(stage_name="parse", success=False, data={"transcript": transcript}, error=str(e))

    async def _run_evaluate_stage(
        self, transcript: str, context: Optional[str], parse_result: EvaluationStageResult
    ) -> EvaluationStageResult:
        try:
            parsed = parse_result.data
            if self._use_llm and self.provider:
                data = await self._llm_evaluate(transcript, parsed, context)
            else:
                data = self._rule_evaluate(parsed)
            return EvaluationStageResult(stage_name="evaluate", success=True, data=data)
        except Exception as e:
            return EvaluationStageResult(stage_name="evaluate", success=False, error=str(e))

    async def _run_report_stage(
        self,
        transcript: str,
        context: Optional[str],
        parse_result: EvaluationStageResult,
        eval_result: EvaluationStageResult,
    ) -> EvaluationStageResult:
        try:
            if self._use_llm and self.provider:
                evaluation = await self._llm_report(transcript, eval_result.data, context)
            else:
                evaluation = self._rule_report(eval_result.data)
            return EvaluationStageResult(
                stage_name="report", success=True, data={"evaluation": evaluation}
            )
        except Exception as e:
            return EvaluationStageResult(stage_name="report", success=False, error=str(e))

    # ── LLM-powered methods ──────────────────────────────────────────────

    async def _llm_parse(self, transcript: str, context: Optional[str]) -> Dict[str, Any]:
        prompt = f"""Extract structured information from this interview transcript:

TRANSCRIPT:
{transcript}

CONTEXT: {context or "None provided"}

Return a JSON object with:
- "skills": list of technical skills mentioned
- "experience_years": estimated years of experience (number or null)
- "seniority_hints": list of seniority-indicating phrases found
- "keywords_matched": list of relevant keywords
"""
        msgs = [Message(role="user", content=prompt)]
        import json
        result = await self.provider.generate(msgs, temperature=0.1)  # type: ignore[union-attr]
        return json.loads(result.content)

    async def _llm_evaluate(
        self, transcript: str, parsed: Dict[str, Any], context: Optional[str]
    ) -> Dict[str, Any]:
        prompt = f"""Evaluate this candidate based on the transcript and parsed data.

TRANSCRIPT:
{transcript}

SKILLS FOUND: {parsed.get('skills', [])}
EXPERIENCE: {parsed.get('experience_years', 'unknown')}

CONTEXT: {context or "None"}

Return a JSON object with:
- "rubric": {{"technical_depth": 0-3, "problem_solving": 0-3, "communication": 0-2, "experience_relevance": 0-2}}
- "score": overall score 0-10
- "level": "Junior", "Mid", "Senior", or "Lead"
- "strengths": list of key strengths
- "weaknesses": list of areas for improvement
- "reasoning": brief chain-of-thought
"""
        msgs = [Message(role="user", content=prompt)]
        import json
        result = await self.provider.generate(msgs, temperature=0.2)  # type: ignore[union-attr]
        return json.loads(result.content)

    async def _llm_report(
        self, transcript: str, eval_data: Dict[str, Any], context: Optional[str]
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate_level=eval_data.get("level", "Mid"),
            score=eval_data.get("score", 5.0),
            recommendation=self._recommend_from_score(eval_data.get("score", 5.0)),
            skills=eval_data.get("skills", []),
            rubric=RubricBreakdown(**eval_data.get("rubric", {})),
            feedback=eval_data.get("feedback", ""),
            strengths=eval_data.get("strengths", []),
            weaknesses=eval_data.get("weaknesses", []),
            chain_of_thought=eval_data.get("reasoning"),
        )

    # ── Rule-based methods ───────────────────────────────────────────────

    def _rule_parse(self, transcript: str) -> Dict[str, Any]:
        lower = transcript.lower()
        keywords_matched = self._extract_keywords(lower)
        skills = self._identify_skills(keywords_matched)
        years = self._extract_years(lower)
        seniority_hints = self._find_seniority_hints(lower)

        return {
            "keywords_matched": keywords_matched,
            "skills": skills,
            "experience_years": years,
            "seniority_hints": seniority_hints,
        }

    def _rule_evaluate(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        skills = parsed.get("skills", [])
        level = self._determine_level(parsed.get("seniority_hints", []))

        rubric = RubricBreakdown(
            technical_depth=min(3.0, len(skills) * 0.4),
            problem_solving=2.0 if "lead" in str(parsed.get("seniority_hints", [])).lower() else 1.5,
            communication=1.5,
            experience_relevance=min(2.0, (parsed.get("experience_years") or 0) / 5.0),
        )

        score = self._calculate_score(rubric, level)
        strengths = skills or ["General technical ability"]
        weaknesses = ["No specific weaknesses identified"]

        return {
            "rubric": rubric.model_dump(),
            "score": score,
            "level": level,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "reasoning": f"Rule-based evaluation: {len(skills)} skills, level={level}, score={score:.1f}",
        }

    def _rule_report(self, eval_data: Dict[str, Any]) -> CandidateEvaluation:
        score = eval_data.get("score", 5.0)
        level = eval_data.get("level", "Mid")
        strengths = eval_data.get("strengths", [])
        weaknesses = eval_data.get("weaknesses", [])
        rubric_data = eval_data.get("rubric", {})

        feedback_parts = [
            f"Candidate is at {level.title()} level.",
            f"Overall evaluation score: {score:.1f}/10.",
        ]
        if strengths:
            feedback_parts.append(f"Strengths: {', '.join(strengths)}.")
        feedback_parts.append(self._recommendation_text(score))

        return CandidateEvaluation(
            candidate_level=level,
            score=score,
            recommendation=self._recommend_from_score(score),
            skills=strengths,
            rubric=RubricBreakdown(**rubric_data),
            feedback=" ".join(feedback_parts),
            strengths=strengths,
            weaknesses=weaknesses,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract recognised skill keywords plus a structured years-of-experience token.

        Adds an ``"{N}_years_experience"`` token to the returned list when an
        explicit years-of-experience phrase (e.g. "8 years", "5+ years", "3 yrs")
        is found in *text*. This makes years-of-experience first-class so callers
        (and tests) can match on ``"<N>_years_experience"``.
        """
        keywords: List[str] = []
        for name, pattern in SKILL_PATTERNS.items():
            if re.search(pattern, text):
                keywords.append(name)
        # Detect "N years" / "N+ years" / "N yrs" experience phrases
        years_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", text)
        if years_match:
            keywords.append(f"{years_match.group(1)}_years_experience")
        return keywords

    def _identify_skills(self, keywords: List[str]) -> List[str]:
        return [kw.title() for kw in keywords if kw in SKILL_PATTERNS]

    def _extract_years(self, text: str) -> Optional[float]:
        match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", text)
        if match:
            return float(match.group(1))
        return None

    def _find_seniority_hints(self, text: str) -> List[str]:
        hints = []
        for level, keywords in SENIORITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    hints.append(kw)
        return hints

    def _determine_level(self, hints: List[str]) -> str:
        scores = {"Junior": 0, "Mid": 0, "Senior": 0}
        for hint in hints:
            for level, keywords in SENIORITY_KEYWORDS.items():
                if hint in keywords:
                    scores[level] += 1
        return max(scores, key=scores.get)  # type: ignore[return-value]

    def _calculate_score(self, rubric: RubricBreakdown, level: str) -> float:
        base = rubric.technical_depth + rubric.problem_solving + rubric.communication + rubric.experience_relevance
        if level == "Senior":
            base += 2.0
        elif level == "Mid":
            base += 1.0
        return min(10.0, base)

    def _recommend_from_score(self, score: float) -> str:
        for threshold, rec, _ in RECOMMENDATION_THRESHOLDS:
            if score >= threshold:
                return rec
        return "Reject"

    def _recommendation_text(self, score: float) -> str:
        for threshold, _, text in RECOMMENDATION_THRESHOLDS:
            if score >= threshold:
                return text
        return RECOMMENDATION_THRESHOLDS[-1][2]
