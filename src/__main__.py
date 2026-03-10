"""
derouler du programme :

__main__.py
   ↓
decoder.py
   ↓
models.py (création d’objets)
   ↓
state_machine.py
   ↓
résultat final

"""
from .decoder import (
    load_function_definitions,
    load_prompts,
    generate_structured_call,
    load_vocabulary
)
import json
import sys
import os
from llm_sdk import Small_LLM_Model  # type: ignore


def main() -> None:
    """
    Point d'entrée principal du programme.

    Il parse les arguments en ligne de commande, charge les définitions de
    fonctions et les requêtes, puis sauvegarde les appels structurés générés
    dans le fichier de sortie spécifié. Gère les erreurs lors du chargement des
    fichiers ou lors de la génération des dossiers de sortie.
    """
    func_def_path = "data/input/functions_definition.json"
    input_path = "data/input/function_calling_tests.json"
    output_path = "data/output/function_calling_results.json"

    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == "--functions_definition" and i + 1 < len(args):
            func_def_path = args[i+1]
        elif args[i] == "--input" and i + 1 < len(args):
            input_path = args[i+1]
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i+1]

    try:
        definitions = load_function_definitions(func_def_path)
        prompts = load_prompts(input_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return

    results = []

    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    vocab_path = model.get_path_to_vocab_file()
    vocab = load_vocabulary(vocab_path)

    for prompt in prompts:
        call = generate_structured_call(prompt, definitions, model, vocab)
        results.append(call.model_dump())
        print(f"Processing prompt (needs LLM implementation): {prompt}")

    try:
        with open(output_path, "w") as f:
            json.dump(
                results,  # Remplacer par vrais outputs de decoder.py
                f,
                indent=4,
                ensure_ascii=False
            )
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    main()
