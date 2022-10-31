import re
import ctypes
import inspect

bin_operator = ['add', 'sub', 'mul', 'div']
unary_operator = ["neg"]


def nxt_token(res_str, idx=0):
    if not res_str:
        return None, None
    while res_str[0] in [",", " "]:
        res_str = res_str[1:]
        idx += 1
        if not res_str:
            return None, None
    if res_str[0] in ["(", ")"]:
        idx += 1
        return res_str[0], idx
    if res_str[:3] in bin_operator + unary_operator:
        idx += 3
        return res_str[:3], idx
    # lst = re.findall(r"^([a-zA-Z_]+)\)", res_str)
    lst = re.findall(r"^[a-zA-Z_]+", res_str)

    # print(res_str, lst)
    if lst:
        idx += len(lst[0])
        return lst[0], idx
    return None, None


def lexical_analysis(parse_str='add(add(add(neg(en), pressure_log_p_abs), neg(en)), neg(ionization_energy_sqrt_p))',
                     ):
    """
    :param parse_str: Prefix expression for gplearn output
    :return: token_lst
    """
    res_str = parse_str
    token_lst = []
    while True:
        token, idx = nxt_token(res_str)
        if token is None:
            break
        res_str = res_str[idx:]
        token_lst.append(token)
    # print(token_lst)
    return token_lst


def calculate(token_lst, par_array):
    """
    :param token_lst: prefix token list
    :param par_array:
    :return:
    """
    stack = []
    for i in range(len(token_lst) - 1, -1, -1):
        if token_lst[i] in ["(", ")"]:
            continue
        if token_lst[i] in unary_operator:
            val = stack.pop()
            stack.append(-val)
        elif token_lst[i] in bin_operator:
            val_1 = stack.pop()
            val_2 = stack.pop()
            if token_lst[i] == "add":
                stack.append(val_1 + val_2)
            elif token_lst[i] == "sub":
                stack.append(val_1 - val_2)
            elif token_lst[i] == "div":
                stack.append(val_1 / val_2)
            elif token_lst[i] == "mul":
                stack.append(val_1 * val_2)
        else:
            stack.append(par_array[token_lst[i]])
    return stack[0]


if __name__ == "__main__":
    dict_ = {"en":1.0, "pressure_log_p_abs":2.0, "ionization_energy_sqrt_p":3.0}
    token_lst = lexical_analysis()
    print(token_lst)
    res = calculate(token_lst, dict_)
    print(res)
    # str_ = "add(add(pressure_inv_p, en_sqrt_p), add(atomic_number_inv_p, add(add(pressure_inv_p, en_sqrt_p), en_sqrt_p)))"
    # token_lst = lexical_analysis(str_)
    # token_lst = lexical_analysis()
    # print(token_lst)
