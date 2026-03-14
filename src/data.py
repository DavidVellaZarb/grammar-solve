import json

from datasets import Dataset

SYSTEM_PROMPT_WITH_GRAMMAR = (
    "You are a semantic parser. Given a user query and a grammar, produce the "
    "program that parses the query according to the grammar. Output only the "
    "program, nothing else."
)

SYSTEM_PROMPT_WITHOUT_GRAMMAR = (
    "You are a semantic parser. Given a user query, produce the program that "
    "parses the query. Output only the program, nothing else."
)

SYSTEM_PROMPT_GRAMMAR = (
    "You are a semantic parser. Given a user query, produce the minimal grammar "
    "for the query. Output only the grammar, nothing else."
)

SYSTEM_PROMPT_GRAMMAR_PROGRAM = (
    "You are a semantic parser. Given a user query, first produce the minimal "
    "grammar for the query, then produce the corresponding program. Separate "
    "the grammar and program with a blank line and the label 'Program:'. "
    "Output only the grammar and program, nothing else."
)

SYSTEM_PROMPT_GRAMMAR_COT = (
    "You are a semantic parser. Given a user query, reason step-by-step about "
    "which grammar rules are needed to parse the query, then output the minimal "
    "grammar wrapped in <grammar> tags.\n\n"
    "First, explain your reasoning about why specific rules are needed. "
    "Then output the grammar inside <grammar>...</grammar> tags.\n"
    "Output only the reasoning and grammar, nothing else."
)


def format_prompt_messages(
    example: dict, include_grammar: bool = True, task: str = "program"
) -> list[dict]:
    if task == "grammar":
        system_prompt = SYSTEM_PROMPT_GRAMMAR
        user_content = f"Query: {example['query']}"
    elif task == "grammar_cot":
        system_prompt = SYSTEM_PROMPT_GRAMMAR_COT
        user_content = f"Query: {example['query']}"
    elif task == "grammar_program":
        system_prompt = SYSTEM_PROMPT_GRAMMAR_PROGRAM
        user_content = f"Query: {example['query']}"
    elif include_grammar:
        system_prompt = SYSTEM_PROMPT_WITH_GRAMMAR
        user_content = (
            f"Query: {example['query']}\n\n"
            f"Grammar:\n{example['minimal_grammar']}"
        )
    else:
        system_prompt = SYSTEM_PROMPT_WITHOUT_GRAMMAR
        user_content = f"Query: {example['query']}"

    module_header = example.get("module_header")
    if module_header:
        user_content = f"Query: {example['query']}\n\nModule header:\n{module_header}"
        if include_grammar and task not in ("grammar", "grammar_cot", "grammar_program"):
            user_content += f"\n\nGrammar:\n{example['minimal_grammar']}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_raw_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["data"]


def _extract_module_header(prompt: str) -> str:
    lines = prompt.split("\n")
    rest_lines = []
    found_module = False
    for line in lines:
        if not found_module and line.strip().startswith("//"):
            continue
        else:
            found_module = True
            rest_lines.append(line)
    return "\n".join(rest_lines).strip()


def load_test_data(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path) as f:
            examples = [json.loads(line) for line in f if line.strip()]
    else:
        examples = load_raw_data(path)

    for i, ex in enumerate(examples):
        if "query" not in ex and "description" in ex:
            ex["query"] = ex["description"]
        if "program" not in ex and "canonical_solution" in ex:
            ex["program"] = ex["canonical_solution"]
        if "module_header" not in ex and "prompt" in ex:
            ex["module_header"] = _extract_module_header(ex["prompt"])

        if "query" not in ex:
            raise ValueError(
                f"Example {i} in {path} has no 'query' or 'description' field. "
                f"Available fields: {list(ex.keys())}"
            )
        if "program" not in ex:
            raise ValueError(
                f"Example {i} in {path} has no 'program' or 'canonical_solution' field. "
                f"Available fields: {list(ex.keys())}"
            )

    return examples


def load_data(
    path: str, include_grammar: bool = True, task: str = "program"
) -> Dataset:
    raw = load_raw_data(path)

    records = []
    for ex in raw:
        if task == "grammar_program":
            completion = f"{ex['minimal_grammar']}\n\nProgram:\n{ex['program']}"
        elif task == "grammar":
            completion = ex["minimal_grammar"]
        elif task == "grammar_cot":
            completion = ex["grammar_cot"]
        else:
            completion = ex["program"]

        records.append(
            {
                "prompt": format_prompt_messages(
                    ex, include_grammar=include_grammar, task=task
                ),
                "completion": [
                    {"role": "assistant", "content": completion},
                ],
            }
        )
    return Dataset.from_list(records)
