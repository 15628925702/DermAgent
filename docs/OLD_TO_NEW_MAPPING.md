# 旧项目到新骨架的映射

## 你当前项目里能继续用的

- `datasets/pad_ufes20.py` -> 继续保留到 `datasets/pad_ufes20.py`
- `agent/state.py` -> 迁移思路到新的 `agent/state.py`
- `skills/perception.py` -> 迁移到新的 `skills/perception.py`
- `skills/reporter.py` -> 迁移到新的 `skills/reporter.py`
- `run_eval.py` -> 迁移到 `scripts/run_eval.py`

## 你当前项目里要拆掉的

- `skills/verifier.py`
  - 原本的大量规则，拆到：
  - `agent/planner.py`
  - `skills/compare.py`
  - `skills/malignancy.py`
  - `skills/metadata_consistency.py`
  - `agent/aggregator.py`

- `skills/retrieval.py`
  - 检索逻辑迁到 `memory/retriever.py`
  - 如果还想保留 skill 形式，`skills/retrieval.py` 只保留薄封装

- `agent/run_agent.py`
  - 原本固定流程，拆成：
  - `agent/planner.py`
  - `agent/router.py`
  - `agent/aggregator.py`
  - `agent/reflection.py`

## 旧文件里建议停更

- `pipeline/build_memory.py`
- `pipeline/infer.py`
- `retrieval/faiss_index.py`（除非你要复用底层索引）
- 重复的 `evaluate.py` / `evaluation.py`
- 全部 `__pycache__/`
