"""会话记忆模块。

多轮对话时，用户常说"它呢？""上面那个制度怎么规定的？"——
没有上下文就不知道"它"是什么。我们按 session_id（一次浏览器会话一个）
在内存中保存最近若干轮对话，用于问题改写时补全指代词。

注意：本项目回答内容始终"只基于检索到的资料"，
历史对话主要用于改写问题，而非直接当作答案依据，这样可以避免旧对话误导回答。
"""

from app.config import settings


class ConversationMemory:
    """内存版会话历史（重启服务后清空，符合本期非目标）。"""

    def __init__(self) -> None:
        # session_id -> [{"role": "user"/"assistant", "content": ...}, ...]
        self._store: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取某会话的历史（返回列表引用，调用方不应修改）。"""
        return self._store.get(session_id, [])

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮对话（一问一答算 1 轮），并裁剪到最大轮数。"""
        history = self._store.setdefault(session_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # 只保留最近 N 轮（1 轮 = 2 条消息），防止历史越来越长、token 费用增加
        max_messages = settings.max_history_turns * 2
        if len(history) > max_messages:
            del history[:-max_messages]

    def clear(self, session_id: str) -> None:
        """清空某会话历史（前端"新对话"按钮可用）。"""
        self._store.pop(session_id, None)
