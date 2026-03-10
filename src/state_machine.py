from typing import List, Dict, Optional
from pydantic import BaseModel, PrivateAttr, create_model, ValidationError
from .models import FunctionCall, FunctionDefinition


class StateMachine(BaseModel):
    """
    Handle the differents states of the machine and transition of the system

    States:
    - "idle": initial state
    - "valid": valid call
    - "invalid_function": unknown function name
    - "invalid_parameters": invalid parameters
    """
    definitions: List[FunctionDefinition]

    _definitions_by_name: Dict[str, FunctionDefinition] = PrivateAttr(
        default_factory=dict
    )
    _state: str = PrivateAttr(default="idle")
    _last_error: str = PrivateAttr(default="")

    def __init__(self,
                 definitions: Optional[List[FunctionDefinition]] = None,
                 **data
                 ):
        if definitions is not None:
            data['definitions'] = definitions
        super().__init__(**data)
        self._definitions_by_name = {d.name: d for d in self.definitions}

    def process_call(self, call: FunctionCall) -> bool:
        """
        Vérifie la validité d'un appel (nom, paramètres, types).
        """
        definition = self._definitions_by_name.get(call.name)
        if not definition:
            self._state = "invalid_function"
            self._last_error = f"Unknown function: {call.name}"
            return False

        try:
            fields = {}
            for name, spec in definition.parameters.items():
                p_type = self._map_type(spec.get("type", "string"))
                fields[name] = (p_type, ...)

            DynamicValidator = create_model(f"Validator_{call.name}", **fields)
            DynamicValidator(**call.parameters)

            self._state = "valid"
            self._last_error = ""
            return True

        except ValidationError as e:
            self._state = "invalid_parameters"
            self._last_error = str(e.errors()[0]['msg'])
            return False
        except Exception as e:
            self._state = "invalid_parameters"
            self._last_error = f"Unexpected error: {str(e)}"
            return False

    def _map_type(self, json_type: str) -> type:
        """
        Convertit le type JSON (string) en type Python pour Pydantic.
        """
        mapping = {
            "number": float,
            "string": str,
            "boolean": bool,
        }
        return mapping.get(json_type, str)

    def get_state(self) -> str:
        return self._state

    def get_last_error(self) -> str:
        return self._last_error
