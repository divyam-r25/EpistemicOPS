import re
from typing import Dict, Tuple, List

class LegacyParser:
    """Parses, scores, and truncates Legacy Documents written by the Primary Agent."""
    
    REQUIRED_SECTIONS = [
        "SECTION 1: WORLD STATE AT ERA END",
        "SECTION 2: TRUST RATINGS",
        "SECTION 3: DRIFT EVENTS DETECTED",
        "SECTION 4: KEY DECISIONS & RATIONALE",
        "SECTION 5: OPEN ISSUES & TECHNICAL DEBT",
        "SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1"
    ]
    
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        # Zero-dependency token estimator: ~3 characters per token
        # Conservative estimate (overestimates token count) to guarantee
        # we never exceed the 2048-token hard limit from the spec.
        # Real tokenizers average ~3.5-4 chars/token for English, so
        # using 3 provides a safe margin.
        self._chars_per_token = 3

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. ~3 chars per token (conservative)."""
        return max(1, len(text) // self._chars_per_token)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens tokens."""
        max_chars = max_tokens * self._chars_per_token
        return text[:max_chars]

    def parse_and_truncate(self, doc_text: str) -> Tuple[str, bool, dict]:
        """
        Enforce token limit and evaluate structural compliance.
        Returns: (truncated_doc, was_truncated, compliance_stats)
        """
        token_count = self._count_tokens(doc_text)
        was_truncated = False
        
        if token_count > self.max_tokens:
            was_truncated = True
            doc_text = self._truncate_to_tokens(doc_text, self.max_tokens)
            doc_text += "\n\n[TRUNCATED BY ENVIRONMENT ENGINE: 2048 TOKEN LIMIT REACHED]"
            
        stats = self._evaluate_structure(doc_text)
        return doc_text, was_truncated, stats

    def _evaluate_structure(self, doc_text: str) -> dict:
        """Check if all required sections are present."""
        found = []
        missing = []

        for section in self.REQUIRED_SECTIONS:
            # Primary: full section title after optional markdown hashes
            pattern = re.compile(rf"#{{0,4}}\s*{re.escape(section)}", re.IGNORECASE)
            if pattern.search(doc_text):
                found.append(section)
                continue

            # Fallback: same section number line with colon (## Section 1: ..., SECTION 1:, etc.)
            head = section.split(":", 1)[0].strip()
            loose = re.compile(
                rf"(?i)(^|[\n\r])\s*#{{0,4}}\s*{re.escape(head)}\b\s*:",
                re.MULTILINE,
            )
            if loose.search(doc_text):
                found.append(section)
            else:
                missing.append(section)

        compliance_score = len(found) / len(self.REQUIRED_SECTIONS)
        
        return {
            "compliance_score": compliance_score,
            "sections_found": found,
            "sections_missing": missing
        }

    def score_drift_capture(self, doc_text: str, actual_drifts: List[dict]) -> float:
        """Score how well the document captured the drift events that occurred."""
        if not actual_drifts:
            return 1.0  # Perfect score if no drifts occurred
            
        # Basic keyword matching based on drift target service and type
        doc_lower = doc_text.lower()
        captured = 0
        
        for drift in actual_drifts:
            service = drift.get("target_service", "").lower()
            behaviour = drift.get("drifted_behaviour") or {}
            field = str(behaviour.get("field", "")).lower()
            
            # Simple heuristic: if the service name and the affected field are mentioned, count it
            if service in doc_lower and (field in doc_lower or drift.get("type", "").lower() in doc_lower):
                captured += 1
                
        return captured / len(actual_drifts)
