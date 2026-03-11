*This project has been created as part of the 42 curriculum by mlorenzo*

## Description

Call-Me-Maybe is a LLM (Large Language Model) project wich goal is to return all the information needed based on the prompt that is given.

We should return the prompt that was given, the function that is needed to use based on the given prompt, as well as the parameters requiered for this specific function.

The goal of this project is to receive a "Natural language", and to change it into something that a computer can understand, and use.

## Instructions

- **Compilation:** This is a Python project, so there is no formal compilation step. However, dependencies need to be managed through `uv`.
- **Installation:** Run `make install` to synchronize `uv` and install the required dependencies (`pydantic`, `numpy`, and the local `llm_sdk`).
- **Execution:** Run `make run` to execute the project with the default paths. You can also manually specify files using:
  `uv run python -m src --functions_definition <definition_file> --input <input_file> --output <output_file>`

## Resources

- https://arxiv.org/pdf/2302.07919
- https://huggingface.co/docs/transformers/tokenizer_summary
- https://mypy.readthedocs.io/en/stable/
- https://docs.pydantic.dev/latest/concepts/models/
- https://docs.python.org/3/library/json.html

- **AI usage:** 
	AI has been used as a tool to search for efficient teaching websites, and to anwser questions while learning that wheren't clear in the documentation found. It has also been used for the structure of the readme, and to do repetitive task, such has type hints.

## Algorithm Explanation

The core of this project relies on **Constrained Decoding** directly applied to the LLM's raw logits.
- **Prefix Matching Check:** Before generating each new token, the current partially generated string + the new possible token is tested against a set of valid JSON templates representing the available functions.
- **Logit Biasing:** Any token that breaks the JSON structure or the expected parameter schema (types like `NUMBER`, `BOOLEAN`, or `STRING_CONTENT`) is strictly prohibited. The algorithm achieves this by setting the model's output logits for invalid tokens to negative infinity `-inf`.
- **Token Selection:** The token with the highest remaining logit score is always selected, fundamentally guaranteeing that the model cannot deviate from a valid path, ensuring a 100% structured JSON formulation at the end of the inference.

## Design Decisions

- **Dictionaries for JSON Templates:** Building static templates representing exactly what the target JSON should look like ensures faster validation versus parsing random JSON attempts dynamically at every token step.
- **Pydantic Validation:** Using `Pydantic` in `models.py` and dynamic models in `state_machine.py` helps centralize the strict validation logic of nested dictionaries and typing, leaving the main pipeline clean and free of excessive type-checking logic.
- **Fallback Avoidance:** Rather than using prompting alone, the logit-masking enforces the schema token by token. Any fallback logic remains minimal since mathematical certainty prevents syntax errors via the constrained prefix path.

## Performance Analysis

- **Accuracy:** The solution achieves near 100% accuracy in structuring the output. Even with the small 0.6B parameter model, parsing issues strictly vanish. Correct function routing depends heavily on the model's semantic grasp, but the constrained environment ensures parameters type-hints are strictly enforced.
- **Speed:** The prefix-matching algorithm per token limits the speed of text generation slightly since every vocabulary token needs validation against templates per frame. However, on standard machines, it easily processes all prompt loops under the 5-minute requirement.
- **Reliability:** By gracefully catching and ignoring missing tokens and handling malformed input files directly at the startup via `try-except`, the program ensures that it crashes neither during load time nor during LLM inferences.

## Challenges Faced

- **Tokenizer Quirks:** Managing the space allocations and special identifier behaviors (like the `Ġ` characters representing spaces in the tokenizer vocabulary) presented challenges around token strings reconstruction, initially causing prefix match false negatives. I addressed this by systematically stripping and replacing these control characters before running the logic map.
- **Building Precise Prefix Matches:** Making sure floats, decimals, negative signs, and strings terminated at the right quota character required careful string index alignment. Solving this meant building custom state validators like tracking dot bounds (`.`) inside `NUMBER` types.

## Testing Strategy

- **Manual Testing:** Multiple edge case prompts like multi-parameter functions, negative numbers, decimals, and heavily capitalized instructions were thrown directly within the JSON `data/input/`.
- **Output Sanity Check:** Relying on `Pydantic` models essentially functioned as my testing gateway. I continually loaded the generated `function_calling_results.json` directly back into Pydantic validators to assert 100% data integrity post-inference.

## Example Usage

With the default files located in `data/input/`:
```bash
make run
```
Which essentially translates to:
```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

**Expected terminal output (sample):**
```
Processing prompt (needs LLM implementation): What is the sum of 2 and 3?
```

**Resulting `function_calling_results.json` example snippet:**
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```