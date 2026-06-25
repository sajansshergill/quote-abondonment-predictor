"""
run_all.py
----------
Runs the full insurance-funnel-analytics pipeline end-to-end.

Usage:
    python run_all.py              # full run
    python run_all.py --skip-dash  # skip launching Streamlit at the end
"""

import argparse
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MPLCONFIGDIR = os.path.join(ROOT, ".matplotlib")

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def log(msg, colour=GREEN):
    print(f"{colour}{BOLD}[pipeline]{RESET} {msg}")

def run(cmd: list[str], cwd: str = ROOT) -> None:
    log(f"▶  {' '.join(cmd)}", YELLOW)
    t0 = time.time()
    env = os.environ.copy()
    os.makedirs(MPLCONFIGDIR, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"✗  Failed (exit {result.returncode}) after {elapsed:.1f}s", RED)
        sys.exit(result.returncode)
    log(f"✓  Done in {elapsed:.1f}s")

# ── Steps ─────────────────────────────────────────────────────────────────────

def step_simulate():
    log("── Step 1: Generate synthetic funnel dataset ──────────────────────")
    run([sys.executable, "data/simulate_funnel.py"])

def step_notebooks():
    log("── Step 2: Run analysis notebooks ────────────────────────────────")
    notebooks = [
        "notebooks/01_eda.py",
        "notebooks/02_survival_analysis.py",
        "notebooks/03_abandonment_model.py",
        "notebooks/04_intervention_sim.py",
    ]
    # Create app/assets dir if missing
    os.makedirs(os.path.join(ROOT, "app", "assets"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    for nb in notebooks:
        log(f"  Running {nb}", YELLOW)
        run([sys.executable, nb])

def step_dashboard():
    log("── Step 3: Launch Streamlit dashboard ────────────────────────────")
    log("  Dashboard will open at http://localhost:8501", GREEN)
    log("  Press Ctrl+C to stop.", GREEN)
    env = os.environ.copy()
    os.makedirs(MPLCONFIGDIR, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/dashboard.py",
            "--server.fileWatcherType",
            "none",
        ],
        cwd=ROOT,
        env=env,
    )

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the full funnel analytics pipeline.")
    parser.add_argument("--skip-dash",      action="store_true", help="Skip launching Streamlit.")
    parser.add_argument("--skip-notebooks", action="store_true", help="Skip notebook execution.")
    parser.add_argument("--only-dash",      action="store_true", help="Only launch Streamlit (assumes data already generated).")
    args = parser.parse_args()

    print(f"\n{BOLD}{'─'*60}")
    print("  Insurance Funnel Analytics — Full Pipeline")
    print(f"{'─'*60}{RESET}\n")

    if args.only_dash:
        step_dashboard()
        return

    step_simulate()

    if not args.skip_notebooks:
        step_notebooks()

    if not args.skip_dash:
        step_dashboard()

    print(f"\n{GREEN}{BOLD}Pipeline complete.{RESET}\n")

if __name__ == "__main__":
    main()