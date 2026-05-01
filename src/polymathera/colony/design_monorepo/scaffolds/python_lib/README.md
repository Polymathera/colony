# ${name}

${description}

This tool was scaffolded into ``tools/${purpose}/${name}/`` of a Polymathera
Colony design monorepo. See
``colony_docs/markdown/apps/design_automation_architecture.md`` (§9) for the
discipline that governs how it is built, validated, and registered.

## Layout

```
${name_dash}/
├── pyproject.toml
├── src/${name_snake}/
│   ├── __init__.py
│   └── core.py
└── tests/
    └── test_smoke.py
```

## Running tests

```
pip install -e .
pytest
```

## Provenance

Scaffolded ${iso_date} by ${author}. Licence: ${license}.
