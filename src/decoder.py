import json
from .models import FunctionDefinition, FunctionCall
from typing import List, Dict, Any, Optional


def load_function_definitions(path: str) -> List[FunctionDefinition]:
    """
    Read a JSON file and return a list of FunctionDefinition objects.
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)

    functions = []
    for obj in data:
        functions.append(
            FunctionDefinition(
                name=obj.get("name", ""),
                description=obj.get("description", ""),
                parameters=obj.get("parameters", {}),
                returns=obj.get("returns", {})
            )
        )
    return functions


def load_prompts(path: str) -> List[str]:
    """
    Read a JSON file and return a list of prompts (strings).
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return [obj["prompt"] for obj in data if "prompt" in obj]


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary from the JSON file provided by the SDK.
    """
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if ("model" in data and isinstance(data["model"], dict) and
                    "vocab" in data["model"]):
                return data["model"]["vocab"]
            return data

        raise ValueError("Le format du fichier vocabulaire JSON "
                         "n'est pas reconnu.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier de vocabulaire '{vocab_path}' "
              "est introuvable.")
        raise
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier '{vocab_path}' n'est pas un JSON valide.")
        raise


def build_templates(definitions: List[FunctionDefinition]) -> List[List[str]]:
    """
    Build validity templates for each function definition.
    """
    templates = []
    for d in definitions:
        tpl = []
        tpl.append(f'{{"name": "{d.name}", "parameters": {{')

        param_items = list(d.parameters.items())
        for i, (p_name, p_spec) in enumerate(param_items):
            p_type = p_spec.get("type", "string")
            tpl.append(f'"{p_name}": ')

            if p_type == "number":
                tpl.append("NUMBER")
            elif p_type == "boolean":
                tpl.append("BOOLEAN")
            else:
                tpl.append('"')
                tpl.append("STRING_CONTENT")
                tpl.append('"')

            if i < len(param_items) - 1:
                tpl.append(", ")

        tpl.append("}}")
        templates.append(tpl)
    return templates


def is_prefix_match(candidate: str, template: List[str]) -> bool:
    """
    Check if the string is a valid prefix for the given template.
    """
    s_idx = 0
    t_idx = 0

    while s_idx < len(candidate) and t_idx < len(template):
        part = template[t_idx]
        rem = candidate[s_idx:]

        if part == "NUMBER":
            i = 0
            if i < len(rem) and rem[i] == '-':
                i += 1

            has_dot = False

            while i < len(rem):
                char = rem[i]
                if char.isdigit():
                    i += 1
                elif char == '.' and not has_dot:
                    has_dot = True
                    i += 1
                else:
                    break

            if i == 0 and rem.startswith("-"):
                return True

            if i > 0:
                s_idx += i
                if s_idx == len(candidate):
                    return True
                t_idx += 1
            else:
                return False

        elif part == "BOOLEAN":
            if "true".startswith(rem) or "false".startswith(rem):
                return True
            elif rem.startswith("true"):
                s_idx += 4
                t_idx += 1
            elif rem.startswith("false"):
                s_idx += 5
                t_idx += 1
            else:
                return False

        elif part == "STRING_CONTENT":
            i = 0
            while i < len(rem) and rem[i] != '"':
                i += 1

            s_idx += i
            if s_idx == len(candidate):
                return True
            t_idx += 1

        else:
            if part.startswith(rem):
                return True
            elif rem.startswith(part):
                s_idx += len(part)
                t_idx += 1
            else:
                return False

    return s_idx == len(candidate)


def get_allowed_tokens(
    current_generation: str,
    vocab: Dict[str, int],
    definitions: List[FunctionDefinition],
    clean_vocab_map: Optional[Dict[int, str]] = None
) -> List[int]:
    """
    Determine allowed next tokens to ensure valid JSON.
    """
    if clean_vocab_map is None:
        clean_vocab_map = {v: k for k, v in vocab.items()}

    templates = build_templates(definitions)
    allowed = []

    for token_id, token_str in clean_vocab_map.items():
        if not token_str:
            continue

        candidate = current_generation + token_str

        for tpl in templates:
            if is_prefix_match(candidate, tpl):
                allowed.append(token_id)
                break

    return allowed


def apply_logit_bias(logits: Any, allowed_tokens: List[int]) -> Any:
    """
    Mask forbidden token probabilities in logits by setting them to -inf.
    """
    allowed_set = set(allowed_tokens)
    for i in range(len(logits)):
        if i not in allowed_set:
            logits[i] = -float('inf')
    return logits


def generate_structured_call(
    prompt: str,
    definitions: List[FunctionDefinition],
    model: Any,
    vocab: Dict[str, int],
    max_tokens: int = 256
) -> FunctionCall:
    """
    Generate a structured function call using an LLM and constrained decoding.
    """
    clean_vocab_map = {}
    for token_str, token_id in vocab.items():
        s = token_str
        s = (s.replace('Ġ', ' ')
             .replace('Ċ', '\n')
             .replace('ĉ', '\n')
             .replace('ċ', '\n'))
        clean_vocab_map[token_id] = s

    sys_prompt = "You are an assistant. Output ONLY a valid JSON object " \
                 "matching exactly one of the definitions.\n"
    for d in definitions:
        sys_prompt += f"- Function: '{d.name}' | Parameters: {d.parameters}\n"
    sys_prompt += f"Request: {prompt}\nJSON:\n"

    input_ids = model.encode(sys_prompt + "{").tolist()[0]  # tokenisation
    current_generation = "{"  # "{" to force JSON starting

    # Constrained Decoding
    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_tokens = get_allowed_tokens(current_generation,
                                            vocab,
                                            definitions,
                                            clean_vocab_map
                                            )

        if not allowed_tokens:
            print(f"Warning: Blocked at step {step} for prompt, "
                  f"no valid token found.")
            break

        logits = apply_logit_bias(logits, allowed_tokens)

        best_token_id = max(range(len(logits)), key=lambda x: logits[x])

        input_ids.append(best_token_id)
        current_generation += clean_vocab_map[best_token_id]

        # Stop genereation as soon as the principal breace close
        if current_generation.endswith('}'):
            try:
                parsed = json.loads(current_generation)
                if "name" in parsed and "parameters" in parsed:
                    return FunctionCall(
                        prompt=prompt,
                        name=parsed["name"],
                        parameters=parsed["parameters"]
                    )
            except ValueError:
                continue

    # Fallback mecanisme in case of loop error
    try:
        parsed = json.loads(current_generation)
        return FunctionCall(prompt=prompt,
                            name=parsed.get("name", "error"),
                            parameters=parsed.get("parameters", {})
                            )
    except Exception:
        return FunctionCall(prompt=prompt, name="error", parameters={})
