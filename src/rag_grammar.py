import asyncio
import sys
import time

import fire
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data import load_raw_data, load_test_data
from grammar_utils import parse_lark_grammar, reconstruct_minimal_grammar
from knn import _find_knn, _load_or_compute_embeddings
from llm_client import LLMClient, cache_key, find_latest_metadata, load_cache, save_cache
from predict_utils import write_output

load_dotenv()


def _load_grammar_as_bnf(grammar_path: str) -> str:
    with open(grammar_path) as f:
        lark_text = f.read()
    rules = parse_lark_grammar(lark_text)
    return reconstruct_minimal_grammar(rules)


SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {
    "smiles": (
        "You are a SMILES grammar prediction assistant. Given a natural language description of a\n"
        "molecule and similar examples with their minimal grammars and SMILES strings, predict the\n"
        "minimal grammar needed to generate the SMILES string for the new molecule.\n\n"
        "CRITICAL — be specific to THIS molecule, not generic:\n"
        "Each rule must list only the alternatives that actually appear in this molecule's SMILES.\n"
        "The similar examples show molecules with DIFFERENT atoms, bonds, and ring counts — do NOT\n"
        "blindly copy their grammar. Instead, reason about what THIS specific molecule requires.\n\n"
        "KEY RULES TO GET RIGHT:\n"
        "- `organic_symbol`: list only the element symbols this molecule contains.\n"
        '  A simple steroid with only carbon and oxygen → organic_symbol ::= "C" | "O"\n'
        "  Do NOT add \"N\", \"S\", \"P\", etc. just because a neighbor example has them.\n"
        "- `aromatic_symbol`: include only if the molecule has aromatic rings (e.g., benzene,\n"
        "  pyridine, furan). List only the aromatic atoms present.\n"
        "- `bond`: include only bond types the molecule uses. Most molecules need only \"=\".\n"
        '  Add "/" or "\\\\" only for cis/trans isomerism (E/Z double bonds).\n'
        '  Add "-" only for explicit single bonds (rare in SMILES). Do NOT include by default.\n'
        "- `ring_closure`: count the distinct ring-closure digits the SMILES needs.\n"
        "  A single ring → \"1\". A steroid (4 fused rings) → \"1\" | \"2\" | \"3\" | \"4\".\n"
        "- `branch`: use concrete enumerated alternatives from the reference grammar,\n"
        '  e.g., "(" smiles ")" | "(" bond smiles ")". Do NOT use quantifiers like ?, +, *.\n'
        "- `atom_spec`: include only the bracket-atom forms actually needed — e.g., for chiral\n"
        "  centers like [C@@H], charged atoms like [O-], or aromatic atoms with H like [nH].\n\n"
        "WHAT TO INCLUDE:\n"
        "- Only rules whose alternatives are actually used in the SMILES for this molecule\n"
        "- Enumerate concrete alternatives (specific symbols, digits, bond characters)\n\n"
        "WHAT TO EXCLUDE:\n"
        "- Alternatives from neighbor examples that don't apply to this molecule\n"
        "- Rules for features the molecule doesn't have (e.g., no charge rule if no charged atoms)\n\n"
        "Think step-by-step: (1) what atoms does this molecule contain, (2) what bond types,\n"
        "(3) how many rings, (4) any special features (chirality, charges, aromaticity).\n"
        "Then output the grammar inside <grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for rule names and structure only):\n{full_grammar}"
    ),
    "default": (
        "You are a grammar prediction assistant for semantic parsing. You are given:\n"
        "1. A reference grammar defining all valid rules for a formal language\n"
        "2. Similar example queries with their minimal grammars and programs\n\n"
        "Your task: given a new query, predict the minimal grammar needed to parse it.\n\n"
        "Guidelines:\n"
        '- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n'
        '- Include concrete string values (e.g., "\\\"Meeting\\\"") and numbers (e.g., 4L) '
        "inferred from the query\n"
        "- The provided examples are retrieved by similarity — not all needed rules may "
        "appear in them\n"
        "- Use the full reference grammar to identify any additional rules beyond what "
        "the examples show\n"
        "- Include only necessary rules and alternatives; do not add extras\n\n"
        "Think step-by-step about which rules are needed, then output the grammar inside "
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar:\n{full_grammar}"
    ),
    "default_cot": (
        "You are a grammar prediction assistant for semantic parsing. You are given:\n"
        "1. A reference grammar defining all valid rules for a formal language\n"
        "2. Similar example queries with step-by-step reasoning about which grammar rules\n"
        "   are needed and why, followed by their minimal grammars and programs\n\n"
        "Your task: given a new query, reason step-by-step about which grammar rules are\n"
        "needed (and why), then predict the minimal grammar.\n\n"
        "Guidelines:\n"
        '- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n'
        '- Include concrete string values (e.g., "\\\"Meeting\\\"") and numbers (e.g., 4L) '
        "inferred from the query\n"
        "- The provided examples are retrieved by similarity — not all needed rules may "
        "appear in them\n"
        "- Use the full reference grammar to identify any additional rules beyond what "
        "the examples show\n"
        "- Include only necessary rules and alternatives; do not add extras\n\n"
        "Follow the same reasoning pattern shown in the examples: explain why each rule\n"
        "is needed for this specific query, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar:\n{full_grammar}"
    ),
    "spice": (
        "You are a SPICE circuit design assistant. Given a natural language description of a circuit\n"
        "and similar example circuits with their SPICE netlists, predict the minimal grammar\n"
        "that describes the netlist for the new circuit.\n\n"
        "CRITICAL FORMAT REQUIREMENT:\n"
        "Each grammar rule alternative must be a CONCRETE component declaration — the exact text\n"
        "that would appear in the SPICE netlist, written as a quoted string literal.\n\n"
        "Correct example:\n"
        '  resistor ::= "R1 3 2 10k" | "R2 6 3 1k"\n'
        '  voltage_source ::= "Vin 3 0" dc_spec | "VCC 9 0" dc_spec\n'
        '  dc_spec ::= "DC 12V" | "DC 0V"\n'
        '  bjt ::= "Q1 5 6 4 NPN_MODEL"\n'
        '  model_def ::= ".model NPN_MODEL NPN (" param_assignment param_assignment ")"\n'
        '  param_assignment ::= "IS=1E-15" | "BF=100"\n\n'
        "WRONG — do NOT output abstract rules like:\n"
        "  resistor ::= COMP_R node node value?\n"
        "  voltage_source ::= COMP_V node node source_spec*\n\n"
        "Guidelines:\n"
        "- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n"
        "- Each component (resistor, voltage_source, bjt, etc.) must have alternatives that are\n"
        "  concrete declarations with specific names, node numbers, and values as quoted strings\n"
        "- Assign node numbers to maintain correct connectivity (node 0 is always ground)\n"
        "- Include .model definitions and analysis commands (.op, .tran, .ac, .dc) as needed\n"
        "- Study the similar examples carefully — they show the exact format expected\n"
        "- Use the reference grammar below only to understand what component types and analysis\n"
        "  commands are available, NOT as an output format template\n"
        "- Include only the rules needed for this specific circuit\n\n"
        "Think step-by-step about what components are needed and how they connect,\n"
        "then output the grammar inside <grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for component types only — do NOT copy this format):\n{full_grammar}"
    ),
    "openscad": (
        "You are an OpenSCAD grammar prediction assistant. Given a natural language description of a\n"
        "3D model and similar examples with their minimal grammars and programs, predict the minimal\n"
        "grammar needed to generate the OpenSCAD code for the new design.\n\n"
        "CRITICAL FORMAT REQUIREMENT:\n"
        "Each grammar rule must enumerate CONCRETE alternatives — the specific variable names, values,\n"
        "module names, and expressions that will appear in the program. Do NOT output abstract/generic\n"
        "syntax rules.\n\n"
        "Correct example:\n"
        '  assignment ::= "diameter=30;" | "thickness=2;" | "$fn=90;"\n'
        '  module_call ::= module_name "(" arg_list ");" | module_name "(" arg_list "){" module_call "}"\n'
        '  module_name ::= "cylinder" | "difference" | "translate"\n'
        '  arg_list ::= arg "," arg\n'
        '  arg ::= "r=" mul_expr | "h=thickness"\n'
        '  mul_expr ::= "diameter/2"\n\n'
        "WRONG — do NOT output abstract rules like:\n"
        '  assignment ::= name "=" expr ";"\n'
        "  ?expr ::= ternary_expr\n"
        "  ?ternary_expr ::= or_expr \"?\" ternary_expr \":\" ternary_expr | or_expr\n"
        '  module_call ::= modifier? module_name "(" arg_list? ")" "{" statement* "}" ";"\n\n'
        "KEY PRINCIPLES:\n"
        "- The `assignment` rule must list every specific variable assignment as a quoted string\n"
        "  alternative (e.g., \"width=10;\"), not a generic pattern like name \"=\" expr \";\"\n"
        "- The `module_name` rule must list only the specific OpenSCAD modules used\n"
        "  (e.g., \"cube\" | \"translate\" | \"difference\"), not a generic name reference\n"
        "- Expression rules (add_expr, mul_expr, etc.) must list the specific expressions that\n"
        "  appear in the program, not generic recursive patterns\n"
        "- Study the similar examples carefully — they show the exact format expected\n"
        "- Use the reference grammar below only to understand what rule names exist,\n"
        "  NOT as an output format template — do NOT copy its abstract structure\n"
        "- Include only the rules needed for this specific design\n\n"
        "Think step-by-step about what the design requires, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for rule names only — do NOT copy this format):\n{full_grammar}"
    ),
    "verilog": (
        "You are a Verilog grammar prediction assistant. Given a natural language description of a\n"
        "hardware module and similar examples with their minimal grammars and programs, predict the\n"
        "minimal grammar needed to generate the module body.\n\n"
        "SCOPE — what the grammar should cover:\n"
        "The grammar describes only the functional body of the module: continuous assignments,\n"
        "always blocks, generate blocks, and any declarations they require (e.g., reg, wire).\n"
        "Do NOT include rules for:\n"
        "- Port declarations (port_decl_stmt, port_variable, list_of_port_variables, port_dir)\n"
        "- The module_item wrapper rule\n"
        "Study the similar examples carefully — their grammars show exactly which rules are in scope.\n\n"
        "CHOOSING AN IMPLEMENTATION APPROACH:\n"
        "Many Verilog tasks can be implemented in multiple structurally different ways (e.g.,\n"
        "boolean expressions via continuous assign vs. case statements in an always block).\n"
        "Follow the implementation pattern shown by the majority of the similar examples.\n"
        "If the examples use always blocks with case statements, predict grammar for that approach.\n"
        "If the examples use continuous assign with boolean expressions, predict that instead.\n\n"
        "TERMINAL VALUES (number, hier_identifier):\n"
        "- For hier_identifier, include only signal names that are clearly required by the query\n"
        "  (e.g., port names mentioned in the description). Do not guess internal signal names.\n"
        "- For number, include only numeric literals that are directly stated or clearly implied\n"
        "  by the query (bit widths, constants, range bounds). Use the Verilog literal format\n"
        "  shown in the examples (e.g., 4'hF, 2'b01, 8'd255).\n"
        "- It is better to under-predict terminal values than to hallucinate extras.\n\n"
        "RECURSIVE RULES:\n"
        "Use proper recursive forms for list-like rules rather than flattening them.\n"
        "Correct:  expression_list ::= expression_list \",\" expression | expression\n"
        "Wrong:    expression_list ::= identifier_ref \",\" identifier_ref\n\n"
        "Guidelines:\n"
        "- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n"
        "- Keep the grammar minimal — fewer incorrect rules is better than broader coverage\n"
        "- Use the reference grammar to verify rule names and structure, not as a source of\n"
        "  extra rules to include\n\n"
        "Think step-by-step about which rules are needed, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar:\n{full_grammar}"
    ),
    "smiles_cot": (
        "You are a SMILES grammar prediction assistant. Given a natural language description of a\n"
        "molecule and similar examples with step-by-step reasoning about which grammar rules are\n"
        "needed and why, followed by their minimal grammars and SMILES strings, predict the\n"
        "minimal grammar needed to generate the SMILES string for the new molecule.\n\n"
        "CRITICAL — be specific to THIS molecule, not generic:\n"
        "Each rule must list only the alternatives that actually appear in this molecule's SMILES.\n"
        "The similar examples show molecules with DIFFERENT atoms, bonds, and ring counts — do NOT\n"
        "blindly copy their grammar. Instead, reason about what THIS specific molecule requires.\n\n"
        "KEY RULES TO GET RIGHT:\n"
        "- `organic_symbol`: list only the element symbols this molecule contains.\n"
        '  A simple steroid with only carbon and oxygen → organic_symbol ::= "C" | "O"\n'
        "  Do NOT add \"N\", \"S\", \"P\", etc. just because a neighbor example has them.\n"
        "- `aromatic_symbol`: include only if the molecule has aromatic rings (e.g., benzene,\n"
        "  pyridine, furan). List only the aromatic atoms present.\n"
        "- `bond`: include only bond types the molecule uses. Most molecules need only \"=\".\n"
        '  Add "/" or "\\\\" only for cis/trans isomerism (E/Z double bonds).\n'
        '  Add "-" only for explicit single bonds (rare in SMILES). Do NOT include by default.\n'
        "- `ring_closure`: count the distinct ring-closure digits the SMILES needs.\n"
        "  A single ring → \"1\". A steroid (4 fused rings) → \"1\" | \"2\" | \"3\" | \"4\".\n"
        "- `branch`: use concrete enumerated alternatives from the reference grammar,\n"
        '  e.g., "(" smiles ")" | "(" bond smiles ")". Do NOT use quantifiers like ?, +, *.\n'
        "- `atom_spec`: include only the bracket-atom forms actually needed — e.g., for chiral\n"
        "  centers like [C@@H], charged atoms like [O-], or aromatic atoms with H like [nH].\n\n"
        "WHAT TO INCLUDE:\n"
        "- Only rules whose alternatives are actually used in the SMILES for this molecule\n"
        "- Enumerate concrete alternatives (specific symbols, digits, bond characters)\n\n"
        "WHAT TO EXCLUDE:\n"
        "- Alternatives from neighbor examples that don't apply to this molecule\n"
        "- Rules for features the molecule doesn't have (e.g., no charge rule if no charged atoms)\n\n"
        "Your task: reason step-by-step about which grammar rules are needed for this molecule\n"
        "(and why), then predict the minimal grammar.\n\n"
        "Follow the same reasoning pattern shown in the examples: explain why each rule\n"
        "is needed for this specific molecule, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for rule names and structure only):\n{full_grammar}"
    ),
    "spice_cot": (
        "You are a SPICE circuit design assistant. Given a natural language description of a circuit\n"
        "and similar example circuits with step-by-step reasoning about which grammar rules are\n"
        "needed and why, followed by their minimal grammars and SPICE netlists, predict the minimal\n"
        "grammar that describes the netlist for the new circuit.\n\n"
        "CRITICAL FORMAT REQUIREMENT:\n"
        "Each grammar rule alternative must be a CONCRETE component declaration — the exact text\n"
        "that would appear in the SPICE netlist, written as a quoted string literal.\n\n"
        "Correct example:\n"
        '  resistor ::= "R1 3 2 10k" | "R2 6 3 1k"\n'
        '  voltage_source ::= "Vin 3 0" dc_spec | "VCC 9 0" dc_spec\n'
        '  dc_spec ::= "DC 12V" | "DC 0V"\n'
        '  bjt ::= "Q1 5 6 4 NPN_MODEL"\n'
        '  model_def ::= ".model NPN_MODEL NPN (" param_assignment param_assignment ")"\n'
        '  param_assignment ::= "IS=1E-15" | "BF=100"\n\n'
        "WRONG — do NOT output abstract rules like:\n"
        "  resistor ::= COMP_R node node value?\n"
        "  voltage_source ::= COMP_V node node source_spec*\n\n"
        "Guidelines:\n"
        "- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n"
        "- Each component (resistor, voltage_source, bjt, etc.) must have alternatives that are\n"
        "  concrete declarations with specific names, node numbers, and values as quoted strings\n"
        "- Assign node numbers to maintain correct connectivity (node 0 is always ground)\n"
        "- Include .model definitions and analysis commands (.op, .tran, .ac, .dc) as needed\n"
        "- Study the similar examples carefully — they show the exact format expected\n"
        "- Use the reference grammar below only to understand what component types and analysis\n"
        "  commands are available, NOT as an output format template\n"
        "- Include only the rules needed for this specific circuit\n\n"
        "Your task: reason step-by-step about which grammar rules are needed for this circuit\n"
        "(and why), then predict the minimal grammar.\n\n"
        "Follow the same reasoning pattern shown in the examples: explain why each rule\n"
        "is needed for this specific circuit, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for component types only — do NOT copy this format):\n{full_grammar}"
    ),
    "openscad_cot": (
        "You are an OpenSCAD grammar prediction assistant. Given a natural language description of a\n"
        "3D model and similar examples with step-by-step reasoning about which grammar rules are\n"
        "needed and why, followed by their minimal grammars and programs, predict the minimal\n"
        "grammar needed to generate the OpenSCAD code for the new design.\n\n"
        "CRITICAL FORMAT REQUIREMENT:\n"
        "Each grammar rule must enumerate CONCRETE alternatives — the specific variable names, values,\n"
        "module names, and expressions that will appear in the program. Do NOT output abstract/generic\n"
        "syntax rules.\n\n"
        "Correct example:\n"
        '  assignment ::= "diameter=30;" | "thickness=2;" | "$fn=90;"\n'
        '  module_call ::= module_name "(" arg_list ");" | module_name "(" arg_list "){" module_call "}"\n'
        '  module_name ::= "cylinder" | "difference" | "translate"\n'
        '  arg_list ::= arg "," arg\n'
        '  arg ::= "r=" mul_expr | "h=thickness"\n'
        '  mul_expr ::= "diameter/2"\n\n'
        "WRONG — do NOT output abstract rules like:\n"
        '  assignment ::= name "=" expr ";"\n'
        "  ?expr ::= ternary_expr\n"
        "  ?ternary_expr ::= or_expr \"?\" ternary_expr \":\" ternary_expr | or_expr\n"
        '  module_call ::= modifier? module_name "(" arg_list? ")" "{" statement* "}" ";"\n\n'
        "KEY PRINCIPLES:\n"
        "- The `assignment` rule must list every specific variable assignment as a quoted string\n"
        "  alternative (e.g., \"width=10;\"), not a generic pattern like name \"=\" expr \";\"\n"
        "- The `module_name` rule must list only the specific OpenSCAD modules used\n"
        "  (e.g., \"cube\" | \"translate\" | \"difference\"), not a generic name reference\n"
        "- Expression rules (add_expr, mul_expr, etc.) must list the specific expressions that\n"
        "  appear in the program, not generic recursive patterns\n"
        "- Study the similar examples carefully — they show the exact format expected\n"
        "- Use the reference grammar below only to understand what rule names exist,\n"
        "  NOT as an output format template — do NOT copy its abstract structure\n"
        "- Include only the rules needed for this specific design\n\n"
        "Your task: reason step-by-step about which grammar rules are needed for this design\n"
        "(and why), then predict the minimal grammar.\n\n"
        "Follow the same reasoning pattern shown in the examples: explain why each rule\n"
        "is needed for this specific design, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar (for rule names only — do NOT copy this format):\n{full_grammar}"
    ),
    "verilog_cot": (
        "You are a Verilog grammar prediction assistant. Given a natural language description of a\n"
        "hardware module and similar examples with step-by-step reasoning about which grammar rules\n"
        "are needed and why, followed by their minimal grammars and programs, predict the\n"
        "minimal grammar needed to generate the module body.\n\n"
        "SCOPE — what the grammar should cover:\n"
        "The grammar describes only the functional body of the module: continuous assignments,\n"
        "always blocks, generate blocks, and any declarations they require (e.g., reg, wire).\n"
        "Do NOT include rules for:\n"
        "- Port declarations (port_decl_stmt, port_variable, list_of_port_variables, port_dir)\n"
        "- The module_item wrapper rule\n"
        "Study the similar examples carefully — their grammars show exactly which rules are in scope.\n\n"
        "CHOOSING AN IMPLEMENTATION APPROACH:\n"
        "Many Verilog tasks can be implemented in multiple structurally different ways (e.g.,\n"
        "boolean expressions via continuous assign vs. case statements in an always block).\n"
        "Follow the implementation pattern shown by the majority of the similar examples.\n"
        "If the examples use always blocks with case statements, predict grammar for that approach.\n"
        "If the examples use continuous assign with boolean expressions, predict that instead.\n\n"
        "TERMINAL VALUES (number, hier_identifier):\n"
        "- For hier_identifier, include only signal names that are clearly required by the query\n"
        "  (e.g., port names mentioned in the description). Do not guess internal signal names.\n"
        "- For number, include only numeric literals that are directly stated or clearly implied\n"
        "  by the query (bit widths, constants, range bounds). Use the Verilog literal format\n"
        "  shown in the examples (e.g., 4'hF, 2'b01, 8'd255).\n"
        "- It is better to under-predict terminal values than to hallucinate extras.\n\n"
        "RECURSIVE RULES:\n"
        "Use proper recursive forms for list-like rules rather than flattening them.\n"
        "Correct:  expression_list ::= expression_list \",\" expression | expression\n"
        "Wrong:    expression_list ::= identifier_ref \",\" identifier_ref\n\n"
        "Guidelines:\n"
        "- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n"
        "- Keep the grammar minimal — fewer incorrect rules is better than broader coverage\n"
        "- Use the reference grammar to verify rule names and structure, not as a source of\n"
        "  extra rules to include\n\n"
        "Your task: reason step-by-step about which grammar rules are needed for this module\n"
        "(and why), then predict the minimal grammar.\n\n"
        "Follow the same reasoning pattern shown in the examples: explain why each rule\n"
        "is needed for this specific module, then output the grammar inside\n"
        "<grammar>...</grammar> tags.\n\n"
        "Reference Grammar:\n{full_grammar}"
    ),
}


def _get_system_prompt(grammar_path: str, full_grammar: str, prompt_style: str = "default") -> str:
    stem = grammar_path.rsplit("/", 1)[-1].split(".")[0].lower()
    if prompt_style == "cot":
        template = SYSTEM_PROMPT_TEMPLATES.get(f"{stem}_cot", SYSTEM_PROMPT_TEMPLATES.get("default_cot"))
    else:
        template = SYSTEM_PROMPT_TEMPLATES.get(stem, SYSTEM_PROMPT_TEMPLATES["default"])
    assert "{full_grammar}" in template, f"No {{full_grammar}} placeholder in template for '{stem}'"
    return template.replace("{full_grammar}", full_grammar)


def _build_user_message(
    test_query: str,
    neighbors: list[dict],
    prompt_style: str = "default",
) -> str:
    parts = ["Similar examples:\n"]
    for i, ex in enumerate(neighbors, 1):
        if prompt_style == "cot" and "grammar_cot" in ex:
            grammar_label = "Reasoning and Grammar"
            grammar_value = ex["grammar_cot"]
        else:
            grammar_label = "Grammar"
            grammar_value = ex["minimal_grammar"]
        parts.append(
            f"--- Example {i} ---\n"
            f"Query: {ex['query']}\n"
            f"{grammar_label}:\n{grammar_value}\n"
            f"Program:\n{ex['program']}\n"
        )
    parts.append(f"--- Your Task ---\nQuery: {test_query}")
    return "\n".join(parts)


def _build_messages(
    test_query: str, neighbors: list[dict], system_prompt: str, prompt_style: str = "default",
) -> list[dict]:
    user_message = _build_user_message(test_query, neighbors, prompt_style=prompt_style)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def _process_example(
    ex: dict,
    neighbors: list[dict],
    system_prompt: str,
    llm: LLMClient,
    cache: dict,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    prompt_style: str = "default",
) -> dict:
    messages = _build_messages(ex["query"], neighbors, system_prompt, prompt_style=prompt_style)
    response = await llm.call(messages, cache, semaphore)
    pbar.update(1)
    return {**ex, "minimal_grammar": response}


async def _predict_async(
    test_data: list[dict],
    train_data: list[dict],
    knn_indices,
    system_prompt: str,
    llm: LLMClient,
    cache: dict,
    max_concurrent: int,
    prompt_style: str = "default",
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc="RAG predict")

    tasks = []
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        tasks.append(
            _process_example(
                ex, neighbors, system_prompt, llm, cache, semaphore, pbar,
                prompt_style=prompt_style,
            )
        )
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def _load_knn(
    test_path: str,
    train_path: str,
    embedding_model: str,
    cache_dir: str,
    k: int,
    batch_size: int,
):
    train_data = load_raw_data(train_path)
    test_data = load_test_data(test_path)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, k={k}")

    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_data]

    encoder = SentenceTransformer(embedding_model)
    train_embeddings = _load_or_compute_embeddings(
        train_queries, encoder, cache_dir, embedding_model, batch_size
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, encoder, cache_dir, embedding_model, batch_size
    )
    del encoder

    knn_indices = _find_knn(test_embeddings, train_embeddings, k)
    print(f"Found {k}-NN for {len(test_queries)} test queries")

    return train_data, test_data, knn_indices


def _write_from_cache(
    test_data, train_data, knn_indices, system_prompt,
    model, cache, output_path, prompt_style: str = "default",
):
    results = []
    n_missing = 0
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        messages = _build_messages(ex["query"], neighbors, system_prompt, prompt_style=prompt_style)
        key = cache_key(messages, model)
        if key in cache:
            results.append({**ex, "minimal_grammar": cache[key]})
        else:
            results.append({**ex, "minimal_grammar": None})
            n_missing += 1
    if n_missing:
        print(f"Warning: {n_missing} examples missing from cache")
    write_output(results, output_path)


def predict(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    grammar_path: str = "grammars/smcalflow.lark",
    output_path: str = "outputs/predicted_grammars/rag/test_k8.json",
    model: str = "claude-opus-4-6",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    k: int = 8,
    cache_path: str = "cache/rag_cache.json",
    cache_dir: str = "cache/knn",
    max_concurrent: int = 5,
    max_tokens: int = 2048,
    batch_size: int = 256,
    api: str = "anthropic",
    mode: str = "async",
    poll_interval: int = 60,
    prompt_style: str = "default",
):
    print(f"Model: {model}, Embedding: {embedding_model}")

    llm = LLMClient(api=api, model=model, max_tokens=max_tokens)

    if mode == "batch":
        task_name = output_path.replace("/", "_").replace(".", "_")


        meta_path = None
        try:
            meta_path = find_latest_metadata(task_name)
            status = LLMClient.check(metadata_path=meta_path)
            if status == "failed":
                print("Previous batch failed, resubmitting...")
                meta_path = None
            elif status == "in_progress":
                print(f"Resuming existing batch from {meta_path}")
        except FileNotFoundError:
            pass

        if meta_path is None:
            full_grammar = _load_grammar_as_bnf(grammar_path)
            system_prompt = _get_system_prompt(grammar_path, full_grammar, prompt_style=prompt_style)

            train_data, test_data, knn_indices = _load_knn(
                test_path, train_path, embedding_model, cache_dir, k, batch_size
            )
            cache = load_cache(cache_path)
            print(f"Loaded cache with {len(cache)} entries")

            requests = []
            for i, ex in enumerate(test_data):
                neighbors = [train_data[idx] for idx in knn_indices[i]]
                messages = _build_messages(
                    ex["query"], neighbors, system_prompt, prompt_style=prompt_style,
                )
                requests.append((f"req-{i}", messages))

            meta_path = llm.submit(requests, cache, task_name)
            save_cache(cache, cache_path)

            if not meta_path:
                _write_from_cache(
                    test_data, train_data, knn_indices, system_prompt,
                    model, cache, output_path, prompt_style=prompt_style,
                )
                return

        print(f"\nPolling every {poll_interval}s...")
        while True:
            status = LLMClient.check(metadata_path=meta_path)
            if status == "completed":
                break
            if status == "failed":
                print("One or more batches failed.")
                sys.exit(1)
            time.sleep(poll_interval)

        full_grammar = _load_grammar_as_bnf(grammar_path)
        system_prompt = _get_system_prompt(grammar_path, full_grammar, prompt_style=prompt_style)

        train_data, test_data, knn_indices = _load_knn(
            test_path, train_path, embedding_model, cache_dir, k, batch_size
        )
        cache = load_cache(cache_path)
        LLMClient.collect(metadata_path=meta_path, cache=cache, cache_path=cache_path)
        _write_from_cache(
            test_data, train_data, knn_indices, system_prompt,
            model, cache, output_path, prompt_style=prompt_style,
        )
        return

    full_grammar = _load_grammar_as_bnf(grammar_path)
    system_prompt = _get_system_prompt(grammar_path, full_grammar, prompt_style=prompt_style)

    train_data, test_data, knn_indices = _load_knn(
        test_path, train_path, embedding_model, cache_dir, k, batch_size
    )
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    results = asyncio.run(
        _predict_async(
            test_data, train_data, knn_indices, system_prompt, llm, cache,
            max_concurrent, prompt_style=prompt_style,
        )
    )

    save_cache(cache, cache_path)
    write_output(results, output_path)


def check(
    metadata_path: str | None = None,
    task_name: str | None = None,
):
    status = LLMClient.check(metadata_path=metadata_path, task_name=task_name)
    print(f"Status: {status}")
    return status


def collect(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    grammar_path: str = "grammars/smcalflow.lark",
    output_path: str = "outputs/predicted_grammars/rag/test_k8.json",
    model: str = "claude-opus-4-6",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    k: int = 8,
    cache_path: str = "cache/rag_cache.json",
    cache_dir: str = "cache/knn",
    batch_size: int = 256,
    metadata_path: str | None = None,
    task_name: str | None = None,
    prompt_style: str = "default",
):
    cache = load_cache(cache_path)
    LLMClient.collect(
        metadata_path=metadata_path, task_name=task_name,
        cache=cache, cache_path=cache_path,
    )

    full_grammar = _load_grammar_as_bnf(grammar_path)
    system_prompt = _get_system_prompt(grammar_path, full_grammar, prompt_style=prompt_style)

    train_data, test_data, knn_indices = _load_knn(
        test_path, train_path, embedding_model, cache_dir, k, batch_size
    )

    _write_from_cache(
        test_data, train_data, knn_indices, system_prompt,
        model, cache, output_path, prompt_style=prompt_style,
    )


if __name__ == "__main__":
    fire.Fire({"predict": predict, "check": check, "collect": collect})
