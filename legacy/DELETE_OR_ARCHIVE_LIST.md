# 建议删除或归档的旧文件

## 可直接删除
- 所有 `__pycache__/`
- 数据目录里的测试残留图片（如果不是正式实验数据）
- 重复评测脚本中的废弃版本

## 先归档，不要继续开发
- `pipeline/build_memory.py`
- `pipeline/infer.py`
- `retrieval/faiss_index.py`
- 旧版 `skills/verifier.py`

## 删除原则
1. 同一职责只保留一个入口
2. 不再让 `pipeline/` 成为主干
3. 不再把复杂路由逻辑堆进 verifier
