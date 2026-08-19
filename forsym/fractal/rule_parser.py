import re


def tokenize(string: str) -> list[str]:
    """'F(c)[+X]F(de)[-X]+X' => ['F(c)', '[', '+', 'X', ']', 'F(de)', '[', '-', 'X', ']', '+', 'X']"""
    pattern = r"[A-Z,+,/,-,&,^]\([^()]+\)|."
    matches = re.findall(pattern, string)

    if not matches:
        return [string]

    return matches


def extract_operand(succ: str) -> str:
    """'F(c+1, w*2)' => 'F({})'"""
    pattern = r"\([^()]+\)"
    replaced = re.sub(pattern, "({})", succ)
    return replaced


def extract_values(token: str) -> list[float] | None:
    """F(12,200) => [12, 200]"""
    pattern = r"\(([-?\d\s.,]+)\)"
    match = re.search(pattern, token)
    if match:
        values = match.group(1).split(",")
        return [float(value.strip()) for value in values]
    return None


def extract_idents(pred: str) -> list[str] | None:
    """F(c,w) => [c, w]"""
    pattern = r"\(([\w\s,]+)\)"
    match = re.search(pattern, pred)
    if match:
        idents = match.group(1).split(",")
        return [ident.strip() for ident in idents]
    return None


def check_format(pred:str, token:str) -> tuple[bool, dict]:
    """Match a parameterized L-system token against a predecessor.

    Parameters
    ----------
    pred : str
        Rule predecessor, such as ``"F(c,w)"``.
    token : str
        Expanded token, such as ``"F(12.23,200.43)"``.

    Returns
    -------
    matched : bool
        Whether the operand names and parameter counts match.
    parameters : dict
        Mapping from predecessor identifiers to token values.

    Raises
    ------
    ValueError
        If matching operands contain different numbers of parameters.
    """

    idents = extract_idents(pred)
    values = extract_values(token)
    params = {}
    if idents is not None and values is not None:
        token_without_values = re.sub(r"\([\d\s.,]+\)", "", token)
        pred_without_idents = re.sub(r"\([\w\s,]+\)", "", pred)
        if pred_without_idents == token_without_values:
            if len(idents) != len(values):
                raise ValueError(f"Mismatch in the number of identifiers and values idents{idents}, values{values}")

            params = {ident: value for ident, value in zip(idents, values)}
            return True, params

    return False, params


def extract_evals(succ:str) -> list[str]:
    """Extract the evaluation string from the rule list
    E.g. F(c+1, w*2)" => ['c+1', 'w*2']"""
    pattern = r"\(([^()]+)\)"
    matches = re.findall(pattern, succ)
    evals = [expr.strip() for expr in matches[0].split(",")] if matches else []
    return evals
