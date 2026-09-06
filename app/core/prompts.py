"""Prompt（提示词）模板集中管理。

把提示词从业务代码里抽出来集中管理，方便以后调优。
本项目有两个提示词：
1. 问题改写：多轮对话时，把"它的适用范围？"这种追问补全成完整问题；
2. 问答生成：要求大模型严格基于检索到的资料回答，并标注引用 [n]。
"""

# ---------- 1. 问题改写 ----------
REWRITE_SYSTEM_PROMPT = """你是一个问题改写助手。用户正在与一个"企业知识库问答机器人"进行多轮对话。
请根据对话历史，把用户最新提出的问题改写成一个【语义完整、可以独立用于检索】的问题。

规则：
1. 补全指代词（如"它""这个""上面那条"）和被省略的主语/宾语；
2. 不得改变原意，不要回答问题，不要输出任何解释或寒暄；
3. 如果最新问题本身已经语义完整，直接原样返回。

只输出改写后的问题本身。"""


def build_rewrite_messages(
    history: list[dict[str, str]],
    question: str,
) -> list[dict[str, str]]:
    """组装问题改写的消息列表。

    :param history: 历史对话，OpenAI 格式 [{"role": "user"/"assistant", "content": ...}]
    :param question: 用户最新提问
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"用户最新问题：{question}\n请输出改写后的独立问题：",
        }
    )
    return messages


# ---------- 2. 基于资料回答 ----------
ANSWER_SYSTEM_PROMPT = """你是企业知识库问答助手。请严格根据用户提供的【参考资料】回答问题。

规则：
1. 只能依据参考资料作答，绝对不能编造资料中没有的信息；
2. 回答中引用资料的位置，要用 [序号] 标注，例如：员工年假为 5 天[1]；
3. 如果参考资料不足以回答问题，直接回复"根据现有知识库资料，未找到与该问题相关的内容"，不要编造；
4. 使用简体中文，条理清晰，内容较长时分点作答；
5. 不要提及"参考资料""片段"等检索系统内部概念，自然地组织答案。"""


def build_answer_messages(
    question: str,
    contexts: list[str],
) -> list[dict[str, str]]:
    """组装问答的消息列表。

    :param question: 改写后的独立问题
    :param contexts: 检索到的片段文本（已按相关性排序），会被编号为 [1][2]...
    """
    # 给每个片段前面加上 [1] [2] ... 编号，供大模型引用
    numbered_blocks = [f"[{i}] {text}" for i, text in enumerate(contexts, start=1)]
    context_text = "\n\n".join(numbered_blocks)

    user_content = f"【参考资料】\n{context_text}\n\n【用户问题】\n{question}"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------- 3. 兜底话术（检索不到相关内容时直接返回，不调用大模型）----------
FALLBACK_ANSWER = (
    "根据现有知识库资料，未找到与该问题相关的内容。\n"
    "请换个问法试试，或确认相关文档是否已上传到知识库。"
)
