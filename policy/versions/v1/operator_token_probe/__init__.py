__all__ = ["OperatorTokenProbePolicyFolder", "run_operator_token_probe"]


def __getattr__(name):
    if name in __all__:
        from .operator_token_probe_policy_impl import OperatorTokenProbePolicyFolder, run_operator_token_probe

        mapping = {
            "OperatorTokenProbePolicyFolder": OperatorTokenProbePolicyFolder,
            "run_operator_token_probe": run_operator_token_probe,
        }
        return mapping[name]
    raise AttributeError(name)
