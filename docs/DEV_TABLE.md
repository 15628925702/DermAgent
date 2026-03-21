# 基础版开发总表

## agent/state.py
职责：定义共享病例状态对象
输入：原始 case dict
输出：CaseState
依赖：无
被这些文件使用：几乎全项目

## skills/perception.py
职责：从 image + metadata 提取初始诊断线索
输入：CaseState
输出：perception result
依赖：models/vlm_backend.py, integrations/openai_client.py
被这些文件使用：agent/run_agent.py, agent/planner.py

## memory/schema.py
职责：定义 experience 的标准字段
输入：结构化字典
输出：标准化 experience item
依赖：无
被这些文件使用：memory/writer.py, memory/experience_bank.py

## memory/experience_bank.py
职责：经验存储、加载、按类型返回
输入：experience item
输出：经验列表 / 统计信息
依赖：memory/schema.py
被这些文件使用：memory/retriever.py, memory/writer.py

## memory/retriever.py
职责：经验检索
输入：CaseState + bank
输出：raw_case_hits / prototype_hits / confusion_hits / rule_hits
依赖：memory/experience_bank.py
被这些文件使用：agent/run_agent.py, agent/planner.py

## agent/planner.py
职责：根据 perception + retrieval 选择 skill
输入：CaseState
输出：selected_skills, planner_reason
依赖：skills/uncertainty.py
被这些文件使用：agent/run_agent.py, agent/router.py

## agent/router.py
职责：执行 selected_skills
输入：CaseState + registry
输出：各 skill result 写入 state
依赖：agent/registry.py
被这些文件使用：agent/run_agent.py

## skills/compare.py
职责：比较候选病种
输入：CaseState
输出：compare result
依赖：memory/retriever.py 的结果
被这些文件使用：agent/router.py, agent/aggregator.py

## skills/malignancy.py
职责：恶性风险判断
输入：CaseState
输出：risk result
依赖：skills/perception.py
被这些文件使用：agent/router.py, agent/aggregator.py

## skills/specialists/ack_scc_specialist.py
职责：ACK / SCC 混淆组精细判断
输入：CaseState
输出：specialist result
依赖：skills/compare.py, memory/retriever.py
被这些文件使用：agent/router.py, agent/aggregator.py

## skills/specialists/mel_nev_specialist.py
职责：MEL / NEV 混淆组精细判断
输入：CaseState
输出：specialist result
依赖：skills/compare.py, memory/retriever.py
被这些文件使用：agent/router.py, agent/aggregator.py

## agent/aggregator.py
职责：汇总 perception / retrieval / compare / risk / specialist 证据
输入：CaseState
输出：final diagnosis bundle
依赖：所有 skills
被这些文件使用：agent/run_agent.py

## skills/reporter.py
职责：把最终结果转成报告文本
输入：CaseState
输出：report dict + report text
依赖：agent/aggregator.py
被这些文件使用：agent/run_agent.py

## agent/reflection.py
职责：生成本病例经验总结
输入：CaseState
输出：experience draft
依赖：memory/schema.py
被这些文件使用：agent/run_agent.py, memory/writer.py

## memory/writer.py
职责：把 reflection 结果写回经验库
输入：CaseState / experience item
输出：bank updated
依赖：memory/schema.py, memory/experience_bank.py
被这些文件使用：agent/run_agent.py
