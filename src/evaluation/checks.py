def question_preserved(src: str, mt: str) -> bool:
    return src.count("?") == mt.count("?")


def exclamation_preserved(src: str, mt: str) -> bool:
    return src.count("!") == mt.count("!")


def token_ratio(src_tokens: list[str], mt_tokens: list[str]) -> float:
    return (len(src_tokens) - len(mt_tokens)) / len(src_tokens)


def length_ratio(src: str, mt: str) -> float:
    return len(src) / len(mt)
