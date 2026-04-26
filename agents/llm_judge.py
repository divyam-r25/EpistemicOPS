import os
import json
import asyncio
import logging

anthropic = None
openai = None

try:
    import anthropic as _anthropic
    anthropic = _anthropic
except ImportError:
    pass

try:
    import openai as _openai
    openai = _openai
except ImportError:
    pass

logger = logging.getLogger("llm-judge")

class LLMJudge:
    _missing_key_warned = {"anthropic": False, "openai": False}

    """Invokes Claude Sonnet/GPT-4o to score Oversight Agent interventions."""
    
    JUDGE_SYSTEM_PROMPT = """You are evaluating the pedagogical quality of an Oversight Agent's response to a Primary Agent that has failed a task due to an API schema change.

You will receive:
1. The drift event that occurred (what actually changed in the API)
2. The Primary Agent's reasoning trace at the point of failure
3. The Oversight Agent's intervention response

Score the intervention on four dimensions (0.0 – 1.0 each):

DIMENSION 1: TARGETING
Does the Oversight Agent's question or prompt target the SPECIFIC incorrect assumption that caused the failure?
- 1.0: Precisely targets the wrong assumption
- 0.5: Targets the general area
- 0.0: Generic advice

DIMENSION 2: RESTRAINT
Does the Oversight Agent avoid revealing the answer?
- 1.0: Question requires genuine reasoning
- 0.5: Question narrows the answer space significantly
- 0.0: Question contains or implies the answer

DIMENSION 3: CALIBRATION
Is the intervention difficulty appropriate to the Primary Agent's current state?
- 1.0: Meets agent exactly where it is
- 0.5: Slightly mismatched
- 0.0: Far too advanced or too simple

DIMENSION 4: ADAPTATION
If this is not the first intervention, did the Oversight Agent change strategy?
- 1.0: Clear strategy change (or first intervention)
- 0.5: Slight variation
- 0.0: Identical approach

OUTPUT FORMAT (JSON only):
{
  "targeting": float,
  "restraint": float,
  "calibration": float,
  "adaptation": float,
  "overall": float,
  "leakage_severity": float,
  "brief_rationale": "string"
}
"""

    def __init__(self):
        # Prefer OpenAI + gpt-4o-mini defaults; auto-fallback from anthropic if needed.
        requested_provider = os.getenv("JUDGE_PROVIDER", "openai").lower()
        requested_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if requested_provider == "anthropic" and not anthropic_key and openai_key:
            logger.warning(
                "JUDGE_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing; "
                "falling back to OpenAI judge."
            )
            self.provider = "openai"
            self.model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
        else:
            self.provider = requested_provider
            self.model = requested_model

        if self.provider == "anthropic":
            if anthropic_key and anthropic is not None:
                self.client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            else:
                self.client = None
                if anthropic is None:
                    logger.warning("Anthropic SDK not installed. Judge will use fallback scoring.")
                elif not LLMJudge._missing_key_warned["anthropic"]:
                    logger.warning("Anthropic API key missing.")
                    LLMJudge._missing_key_warned["anthropic"] = True
        else:
            if openai_key and openai is not None:
                self.client = openai.AsyncOpenAI(api_key=openai_key)
            else:
                self.client = None
                if openai is None:
                    logger.warning("OpenAI SDK not installed. Judge will use fallback scoring.")
                elif not LLMJudge._missing_key_warned["openai"]:
                    logger.warning("OpenAI API key missing.")
                    LLMJudge._missing_key_warned["openai"] = True

    def _build_prompt(self, drift_config: dict, primary_trace: list, oversight_response: str) -> str:
        return (
            f"1. DRIFT EVENT:\n{json.dumps(drift_config, indent=2)}\n\n"
            f"2. PRIMARY AGENT REASONING:\n{json.dumps(primary_trace, indent=2)}\n\n"
            f"3. OVERSIGHT INTERVENTION:\n{oversight_response}"
        )

    def _fallback_score(self) -> dict:
        """Returned when API times out or fails."""
        return {
            "targeting": 0.5,
            "restraint": 0.5,
            "calibration": 0.5,
            "adaptation": 0.5,
            "overall": 0.5,
            "leakage_severity": 0.5,
            "brief_rationale": "Judge unavailable: neutral scores applied.",
            "judge_used_fallback": True,
        }

    async def evaluate_intervention(self, drift_config: dict, primary_trace: list, oversight_response: str) -> dict:
        """Call the LLM Judge asynchronously with a 10s timeout."""
        if not self.client:
            return self._fallback_score()
            
        prompt = self._build_prompt(drift_config, primary_trace, oversight_response)
        
        try:
            if self.provider == "anthropic":
                return await asyncio.wait_for(
                    self._call_anthropic(prompt), timeout=10.0
                )
            else:
                return await asyncio.wait_for(
                    self._call_openai(prompt), timeout=10.0
                )
        except asyncio.TimeoutError:
            logger.error("LLM Judge API call timed out.")
            return self._fallback_score()
        except Exception as e:
            logger.error(f"LLM Judge API call failed: {e}")
            return self._fallback_score()

    async def _call_anthropic(self, prompt: str) -> dict:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=self.JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.content[0].text)
        
    async def _call_openai(self, prompt: str) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return self._parse_json(response.choices[0].message.content)

    def _parse_json(self, text: str) -> dict:
        try:
            # Simple cleanup for markdown json blocks
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            if isinstance(data, dict) and "judge_used_fallback" not in data:
                data["judge_used_fallback"] = False
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Judge JSON response: {text}")
            return self._fallback_score()
