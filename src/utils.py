def get_expected_type(parameter_spec) -> str:
    """
    Get the expected type of a parameter
    """
    try:
        return str(parameter_spec.type)
    except AttributeError:
        pass

    if isinstance(parameter_spec, dict) and "type" in parameter_spec:
        return str(parameter_spec["type"])

    return ""
