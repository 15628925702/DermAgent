"""
全局配置。

这里只放：
- 默认模型名
- top_k
- planner 阈值
- experience bank 路径

不要把业务逻辑写进配置文件。
"""

DEFAULT_PERCEPTION_MODEL = "gpt-4o-mini"
DEFAULT_RETRIEVAL_TOP_K = 5
UNCERTAINTY_COMPARE_THRESHOLD = 0.5
EXPERIENCE_BANK_PATH = "./artifacts/experience_bank.json"
ENABLE_LANGCHAIN_PLANNER = False
