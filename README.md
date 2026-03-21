DermAgent is a dermatology-agent research workspace organized around four main layers:

- `agent/`: orchestration, planning, routing, aggregation, reflection
- `skills/`: perception, comparison, metadata checks, specialists, reporting
- `memory/`: experience bank, retrieval, compression, writeback
- `scripts/`: evaluation, training, review, and workflow entry points

Useful entry points:

- [WORKFLOW.md](/g:/0-newResearch/derm_agent/docs/WORKFLOW.md): mainline experiment workflow
- [docs/ROOT_ORGANIZATION.md](/g:/0-newResearch/derm_agent/docs/ROOT_ORGANIZATION.md): what belongs in root vs `outputs/`
- [OUTPUT_LAYOUT.md](/g:/0-newResearch/derm_agent/docs/OUTPUT_LAYOUT.md): artifact and checkpoint layout
- [PROJECT_TREE.txt](/g:/0-newResearch/derm_agent/docs/PROJECT_TREE.txt): compact module map
- [RUN_CATALOG.md](/g:/0-newResearch/derm_agent/outputs/train_runs/RUN_CATALOG.md): retained train-run notes and baseline recommendations

Generated artifacts should go under `outputs/`, especially:

- `outputs/checkpoints/seed/`
- `outputs/train_runs/`
- `outputs/archive/`
- `outputs/logs/`
- `outputs/test_artifacts/`
