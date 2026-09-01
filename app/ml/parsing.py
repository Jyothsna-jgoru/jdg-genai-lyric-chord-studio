from __future__ import annotations

import ast
import json
import re

from app.schemas.studio import SectionGeneration


def parse_structured_output(raw: str) -> tuple[SectionGeneration | None, bool, str | None]:
    try:
        return SectionGeneration.model_validate_json(raw.strip()), False, None
    except Exception as first_error:
        repaired = raw.strip()
        start, end = repaired.find("{"), repaired.rfind("}")
        if start >= 0 and end > start:
            repaired = repaired[start:end + 1]
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            if "'" in repaired and '"' not in repaired:
                repaired = json.dumps(ast.literal_eval(repaired))
            return SectionGeneration.model_validate_json(repaired), True, str(first_error)
        except Exception as repair_error:
            return None, True, f"Initial parse: {first_error}; repair: {repair_error}"

