import ast
import types
from pathlib import Path
import numpy as np


def load_assign_cluster():
    """Extract the assign_cluster function from Experiment/CDT.py without
    executing the entire script."""
    path = Path(__file__).resolve().parents[1] / "Experiment" / "CDT.py"
    source = path.read_text()
    module_ast = ast.parse(source)
    assign_def = None
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "assign_cluster":
            assign_def = node
            break
    if assign_def is None:
        raise RuntimeError("assign_cluster not found")
    ctx = {"np": np}
    exec(compile(ast.Module([assign_def], []), filename=str(path), mode="exec"), ctx)
    return ctx["assign_cluster"], ctx


def setup_test_context():
    """Return assign_cluster bound to a context with fake centroids and scaler."""
    assign_cluster, ctx = load_assign_cluster()
    ctx["scaler_mean"] = np.zeros(6, dtype=np.float32)
    ctx["scaler_std"] = np.ones(6, dtype=np.float32)
    ctx["CLUSTER_CENTROIDS"] = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [2, 0, 2, 0, 2, 0],
        [3, 0, 3, 0, 3, 0],
        [4, 0, 4, 0, 4, 0],
        [5, 0, 5, 0, 5, 0],
    ], dtype=np.float32)
    return assign_cluster


def test_assign_cluster_returns_expected_indices():
    ac = setup_test_context()
    assert ac(0, 0, 0) == 0
    assert ac(3, 3, 3) == 3
    assert ac(5, 5, 5) == 5
