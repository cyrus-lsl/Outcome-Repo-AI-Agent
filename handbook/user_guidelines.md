# Measurement Instrument Assistant — User Guidelines

## What This Assistant Can Do

This AI assistant helps you explore the HKJC measurement instrument repository. You can:

1. **Find instruments** — describe your measurement need and get ranked recommendations
2. **Look up project usage** — see which instruments a project used and how
3. **Compare instruments** — side-by-side comparison of two or more tools
4. **Ask why** — after a search, ask why specific instruments were recommended
5. **Get help** — ask how to use this system (you're reading part of it now)

## Example Questions

### Find instruments
- "mental health assessment for elderly"
- "physical activity questionnaire for youth"
- "quality of life scale validated in Hong Kong"
- "programme-level metric for depression, maximum 10 items"

### Project usage (Details)
- "What instruments did P2024-001 use?"
- "Which projects used PHQ-9?"

### Compare
- "Compare PHQ-9 and GAD-7"
- "What's the difference between DASS-21 and BDI?"

### Why (follow-up after search)
- "Why did you recommend the first one?"
- "Why these instruments?"

## Search Tips

- Be specific about **target group** (e.g. youth, elderly, adults)
- Mention **outcome domain** (e.g. mental health, physical activity, quality of life)
- Add filters in natural language:
  - **HK validated**: "validated in Hong Kong" or "HK-validated"
  - **Programme-level**: "programme-level metric"
  - **Item count**: "maximum 10 items" or "at least 20 questions"

## Key Terms

### HK Validated
An instrument that has been validated or adapted for use in Hong Kong. The database records validation status in the "Validated in Hong Kong" column.

### Programme-level Metric
An instrument flagged as suitable for programme-level outcome reporting. Not all instruments carry this designation.

### Project No
Project identifiers follow the format `PYYYY-NNN` (e.g. `P2024-001`). Use these when asking about which instruments a specific project used.

## Data Sources

- **Instrument catalogue**: `measurement_instruments.xlsx` — full instrument metadata
- **Project usage**: `project_usage.xlsx` — which projects used which instruments and how

## Limitations

- Recommendations are AI-assisted and should be reviewed by subject matter experts
- Project usage data may not cover all historical projects
- The assistant explains prior results for "why" questions — it does not re-search unless you ask a new find/compare question
