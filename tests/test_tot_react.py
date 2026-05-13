from latent_kv.benchmarks import load_game24, verify_output
from latent_kv.react_baseline import REACT_TASKS, run_react_task
from latent_kv.tot_baseline import solve_game24_tot


def test_tot_solver_solves_builtin_game24_example():
    example = load_game24(1)[0]
    expression, diagnostics = solve_game24_tot(example.metadata["numbers"], breadth=5)
    parsed, correct = verify_output(expression, example)
    assert parsed is not None
    assert correct
    assert diagnostics["expanded"] > 0


def test_react_task_finishes_with_item():
    text, success, transcript = run_react_task(REACT_TASKS[0])
    assert success
    assert "Thought" in text
    assert transcript[-1]["action"] == "finish"

