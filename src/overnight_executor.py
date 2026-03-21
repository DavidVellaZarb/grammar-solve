import os
import subprocess
import tempfile

from nltk import tree


def _to_lisp_string(node) -> str:
    if isinstance(node, tree.Tree):
        return f"({node.label()} {' '.join(_to_lisp_string(c) for c in node)})"
    return node


def denormalize_lf(raw_lf: str) -> str:
    function_names = [
        "listValue", "filter", "ensureNumericProperty", "ensureNumericEntity",
        "superlative", "countSuperlative", "countComparative", "_size",
        "aggregate", "getProperty", "singleton", "concat",
    ]
    string_symbols = [
        "shape", "color", "length", "is_special", "width", "height",
        "left", "right", "above", "below",
        "=", ">", "<", ">=", "<=", "!=",
        "sum", "max", "min", "avg", "!type",
    ]
    number_symbols = ["3 en.inch", "6 en.inch", "2"]
    padded_number_symbols = [s.replace(" ", "#") for s in number_symbols]

    for symbol in number_symbols:
        if symbol in raw_lf:
            raw_lf = raw_lf.replace(symbol, symbol.replace(" ", "#"))

    lf_tree = tree.Tree.fromstring(raw_lf)

    def denormalize(node):
        if isinstance(node, tree.Tree) and node.label() in function_names:
            if node.label().startswith("_"):
                real_label = "." + node.label()[1:]
            else:
                real_label = "SW." + node.label()
            node.set_label("call")
            node.insert(0, real_label)

            for index, child in enumerate(node):
                if index == 0:
                    continue
                if not isinstance(child, tree.Tree):
                    if child in string_symbols:
                        node[index] = f"(string {child})"
                    elif child in padded_number_symbols:
                        child = child.replace("#", " ")
                        node[index] = f"(number {child})"
                else:
                    denormalize(child)

    def to_spaced_lisp(node):
        if isinstance(node, tree.Tree):
            return f"( {node.label()} {' '.join(to_spaced_lisp(c) for c in node)} )"
        return node

    denormalize(lf_tree)
    return to_spaced_lisp(lf_tree)


_EVAL_PATH = "third_party/overnight"


def execute(programs: list[str], domain: str = "blocks") -> list[str | None]:
    def post_process(lf: str) -> str:
        if lf is None:
            lf = "None"
        return lf.replace("SW", "edu.stanford.nlp.sempre.overnight.SimpleWorld")

    cur_dir = os.getcwd()
    os.chdir(_EVAL_PATH)
    eval_script = "./evaluator/overnight"

    tf = tempfile.NamedTemporaryFile(suffix=".examples", mode="w", delete=False)
    for lf in programs:
        tf.write(post_process(lf) + "\n")
    tf.close()

    try:
        msg = subprocess.check_output(
            [eval_script, domain, tf.name],
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        msg = msg.decode("utf-8")
        denotations = [
            line.split("\t")[1]
            for line in msg.split("\n")
            if line.startswith("targetValue\t")
        ]
        denotations = [
            None if "FAILED" in d or "Exception" in d else d
            for d in denotations
        ]
    except Exception:
        denotations = [None] * len(programs)
    finally:
        os.unlink(tf.name)
        os.chdir(cur_dir)

    return denotations


def execute_single(program: str, domain: str = "blocks") -> str | None:
    results = execute([program], domain)
    return results[0] if results else None


def is_available() -> bool:
    eval_script = os.path.join(_EVAL_PATH, "evaluator", "overnight")
    return os.path.isfile(eval_script) and os.access(eval_script, os.X_OK)
