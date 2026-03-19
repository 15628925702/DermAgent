# DermAgent Scaffold v2

这是一个**基础版可落地开发骨架**，目标不是一次性把所有逻辑写完，而是把项目拆成：

- `agent/`：中控层，负责规划、路由、聚合、反思
- `skills/`：能力层，负责感知、比较、恶性判断、专病判断、报告
- `memory/`：经验层，负责经验 schema、检索、写回、压缩
- `integrations/langchain/`：可选的 LangChain 接口层，只做工具封装和 planner 接口，不承载论文创新
- `datasets/` / `models/` / `llm/`：数据与模型适配层
- `docs/`：迁移说明和开发表
- `legacy/`：旧文件归档建议

## 你现在应该怎么用

1. 解压这个 scaffold
2. 把你当前项目中还能用的逻辑，逐步迁移到对应文件
3. 每次只开发一个文件
4. 以后你只要把某个文件发给我，我就能按这个接口继续补全

## 开发顺序

1. `agent/state.py`
2. `skills/perception.py`
3. `memory/schema.py`
4. `memory/experience_bank.py`
5. `memory/retriever.py`
6. `agent/planner.py`
7. `agent/router.py`
8. `skills/compare.py`
9. `skills/malignancy.py`
10. `skills/specialists/ack_scc_specialist.py`
11. `skills/specialists/mel_nev_specialist.py`
12. `agent/aggregator.py`
13. `skills/reporter.py`
14. `agent/reflection.py`
15. `memory/writer.py`

## LangChain 放哪里

只放在：

- `integrations/langchain/tools.py`：把 skills 封装成 tools
- `integrations/langchain/planner_chain.py`：可选的 LLM planner 接口

不要把核心逻辑写进 LangChain chain 里。
论文核心还是你自己的：

- experience bank
- planner / router
- skill library
- reflection / writeback

## 旧项目里建议停更或删除的内容

详见：
- `docs/OLD_TO_NEW_MAPPING.md`
- `legacy/DELETE_OR_ARCHIVE_LIST.md`
