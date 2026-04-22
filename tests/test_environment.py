import pytest
from environment.world_engine import WorldEngine, Phase
from environment.legacy_parser import LegacyParser

def test_world_engine_initialization():
    engine = WorldEngine()
    engine.initialize_era({"eras": [{"era_id": 1}]}, 1)
    
    assert engine.state.era_id == 1
    assert engine.state.phase == Phase.AWAKENING
    assert engine.state.step == 0

def test_legacy_parser_truncation():
    parser = LegacyParser(max_tokens=10) # tiny limit for testing
    
    doc = "# SECTION 1: WORLD STATE AT ERA END\n This is a very long document that will definitely exceed the ten token limit."
    truncated_doc, was_truncated, stats = parser.parse_and_truncate(doc)
    
    assert was_truncated is True
    assert "[TRUNCATED" in truncated_doc

def test_legacy_parser_compliance():
    parser = LegacyParser()
    doc = """
    # SECTION 1: WORLD STATE AT ERA END
    data
    # SECTION 2: TRUST RATINGS
    data
    # SECTION 3: DRIFT EVENTS DETECTED
    data
    # SECTION 4: KEY DECISIONS & RATIONALE
    data
    # SECTION 5: OPEN ISSUES & TECHNICAL DEBT
    data
    # SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1
    data
    """
    _, _, stats = parser.parse_and_truncate(doc)
    assert stats["compliance_score"] == 1.0
