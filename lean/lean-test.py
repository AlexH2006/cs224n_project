import modal
import subprocess

app = modal.App(name="lean-test")
image = modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13.2")

TESTS = [
    ("trivial true", "import Mathlib\n\nexample : True := trivial\n", True),
    ("1 + 1 = 2", "import Mathlib\n\nexample : 1 + 1 = 2 := by norm_num\n", True),
    ("bad proof", "import Mathlib\n\nexample : 1 + 1 = 3 := by norm_num\n", False),
]

@app.function(image=image)
def check_lean(lean_code: str) -> tuple[bool, str]:
    with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
        f.write(lean_code)
    result = subprocess.run(
        ["lake", "env", "lean", "LeanChecker/Test.lean"],
        cwd="/lean-checker",
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stderr.strip()

@app.local_entrypoint()
def main():
    for name, code, expected in TESTS:
        passed, stderr = check_lean.remote(code)
        status = "PASS" if passed else "FAIL"
        correct = passed == expected
        print(f"[{'OK' if correct else 'WRONG'}] {name}: {status}")
        if stderr and not passed:
            print(f"       {stderr[:200]}")
