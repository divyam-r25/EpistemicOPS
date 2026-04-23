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
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def parse_and_truncate(self, doc_text: str) -> Tuple[str, bool, dict]:
        """
        Enforce token limit and evaluate structural compliance.
        Returns: (truncated_doc, was_truncated, compliance_stats)
        """
        tokens = self.tokenizer.encode(doc_text)
        was_truncated = False
        
        if len(tokens) > self.max_tokens:
            was_truncated = True
            tokens = tokens[:self.max_tokens]
            doc_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            doc_text += "\n\n[TRUNCATED BY ENVIRONMENT ENGINE: 2048 TOKEN LIMIT REACHED]"
            
        stats = self._evaluate_structure(doc_text)
        return doc_text, was_truncated, stats

    def _evaluate_structure(self, doc_text: str) -> dict:
        """Check if all required sections are present."""
        found = []
        missing = []
        
        for section in self.REQUIRED_SECTIONS:
            # Flexible matching allowing markdown headers (##, ###) and slight whitespace differences
            pattern = re.compile(rf"#{{0,4}}\s*{re.escape(section)}", re.IGNORECASE)
            if pattern.search(doc_text):
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
            field = drift.get("drifted_behaviour", {}).get("field", "").lower()
            
            # Simple heuristic: if the service name and the affected field are mentioned, count it
            if service in doc_lower and (field in doc_lower or drift.get("type", "").lower() in doc_lower):
                captured += 1
                
        return captured / len(actual_drifts)
