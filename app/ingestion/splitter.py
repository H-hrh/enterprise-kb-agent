"""文本切分模块：把长文档切成一个个带重叠的小片段（chunk）。

【为什么要切分？】
1. 向量模型 / 大模型都有输入长度限制，整篇文档塞不进去；
2. 片段越小，检索越精准——用户问"年假几天"，不希望把整本员工手册都喂给大模型；
3. 片段之间保留一点重叠（overlap），防止一句话正好从中间被切断、上下文丢失。

【切分策略】
1. Markdown 标题感知（首选边界）：文档含 # / ## 标题时，先按标题拆成"小节"，
   每个小节主题单一、独立成片（超长的小节再递归细分）。
   —— 好处：检索片段主题纯净，重排序（Rerank）分数更有区分度，
      避免"一个片段混了年假/病假/婚假多个主题"导致相关度被打分稀释。
2. 普通文本：递归降级，优先在"最自然的边界"切开，边界级别从大到小：
   段落(空行) → 换行 → 句号/问号/叹号/分号 → 英文句点 → 最后才按字符硬切
3. 超长小节 / 普通文本合并时，相邻片段保留重叠（overlap），
   防止一句话正好从中间被切断、上下文丢失。
"""

import re

# 分隔符优先级：从"最宽泛"到"最细粒度"
# 注意顺序很重要：先尝试段落，切不开再降级到句子，最后硬切
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", ""]

# Markdown 标题行：1~6 个 # + 空格 + 标题文字（匹配行首）
_MD_HEADER_RE = re.compile(r"(?m)^(#{1,6} .+)$")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """把长文本切分为带重叠的片段列表。

    :param text: 整篇文档的纯文本
    :param chunk_size: 每个片段的目标最大字符数
    :param chunk_overlap: 相邻片段的重叠字符数
    :return: 片段文本列表（已去除空白片段）
    """
    # 统一换行符，避免 Windows 的 \\r\\n 影响切分
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 首选：Markdown 标题感知切分（每个小节主题单一）
    sections = _split_markdown_sections(text)
    if sections is not None:
        chunks: list[str] = []
        for section in sections:
            if len(section) <= chunk_size:
                # 小节不超长 → 整节作为一个片段（主题完整，检索最精准）
                chunks.append(section)
            else:
                # 小节超长 → 在小节内部递归切分（保留重叠）
                pieces = _recursive_split(section, chunk_size, _SEPARATORS)
                chunks.extend(_merge_with_overlap(pieces, chunk_size, chunk_overlap))
        return [c for c in chunks if c.strip()]

    # 兜底：非 Markdown 文本 → 递归切出小单元，再贪心合并成带重叠的片段
    pieces = _recursive_split(text, chunk_size, _SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _split_markdown_sections(text: str) -> list[str] | None:
    """按 Markdown 标题把文档拆成小节列表；文档不含标题时返回 None。

    每个小节以标题行开头，包含标题下的全部内容；
    第一个标题之前的内容（如文档大标题、前言）也单独成一小节。
    """
    matches = list(_MD_HEADER_RE.finditer(text))
    if not matches:
        return None

    sections: list[str] = []

    # 第一个标题之前的内容（文档开头部分）
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(pre)

    # 每个标题到下一个标题之间的内容
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """按分隔符层级递归切分。

    思路：用当前级别的分隔符把文本切开；
    如果切出来的某一段还是太长，就用下一级分隔符继续切那段；
    直到所有片段都不超过 size，或者降级到字符硬切。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sep = separators[0]
    next_separators = separators[1:]

    # 最后一级：没有更细的分隔符了，按字符硬切
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    # 用当前分隔符切开（把分隔符保留在后一段的开头，避免标点丢失）
    units = text.split(sep)

    pieces: list[str] = []
    for i, unit in enumerate(units):
        if i > 0:
            unit = sep + unit
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) <= size:
            pieces.append(unit)
        else:
            # 这个单元还是太长 → 用下一级分隔符递归切
            pieces.extend(_recursive_split(unit, size, next_separators))
    return pieces


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """把小片段贪心合并到接近 size，并让相邻片段共享 overlap 个字符。

    例：size=500, overlap=50
    第 1 片：[第 1~500 字]
    第 2 片：[第 451~950 字]  ← 前 50 字与上一片末尾重叠，保证上下文连续
    """
    if not pieces:
        return []

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= size:
            # 还放得下，继续往当前片段追加
            current += piece
            continue

        # 当前片段已满 → 保存
        if current.strip():
            chunks.append(current.strip())

        # 从上一片末尾取 overlap 个字符作为新片段的开头（重叠区）
        current = (current[-overlap:] if overlap > 0 else "") + piece

        # 兜底：单个 piece 本身就超长（理论上递归切分后不会，双保险）
        while len(current) > size:
            chunks.append(current[:size].strip())
            current = current[size - overlap :] if overlap > 0 else current[size:]

    if current.strip():
        chunks.append(current.strip())

    return chunks
