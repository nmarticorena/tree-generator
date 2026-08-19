from forsym.tree.config import LSystemConfig
from forsym.fractal import rule_parser as parser





def expand_lsystem(l_config:LSystemConfig):
    """Expand a configured L-system to its final generation.

    Parameters
    ----------
    l_config : forsym.tree.config.LSystemConfig
        Axiom, rewrite rules, free parameters, and generation count.

    Returns
    -------
    str
        Fully expanded L-system string.
    """
    def _evaluate(successor, parameters):
        tokens = []
        for token in parser.tokenize(successor):
            operand = parser.extract_operand(token)
            if "(" in operand and ")" in operand:
                values = [eval(expression, parameters) for expression in parser.extract_evals(token)]
                operand = operand.format(",".join(map(str, values)))
            tokens.append(operand)
        return "".join(tokens)
    
    l_string = l_config.axiom

    for _ in range(l_config.generations):
        l_tokens = parser.tokenize(l_string)
        new_l_string = ""
        for token in l_tokens:
            rule_matched = False
            for rule in l_config.rules:
                partial_match, match_params = parser.check_format(rule.pred, token)
                whole_params = {**match_params, **l_config.free_params}

                if token == rule.pred:
                    new_l_string += _evaluate(rule.succ, whole_params)
                    rule_matched = True
                    break

                if partial_match:
                    new_l_string += _evaluate(rule.succ, whole_params)
                    rule_matched = True
                    break

            if not rule_matched:
                new_l_string += token

        l_string = new_l_string

    return l_string
