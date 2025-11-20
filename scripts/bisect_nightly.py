#!/usr/bin/env python3
import subprocess
import re
from textwrap import dedent
from tqdm import tqdm

PYTORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cu128/torch/"
CUDA_VERSION = "cu128"
MANYLINUX_VERSION = "manylinux_2_28_x86_64"


def run_in_conda(cmd, conda_env):
    full_cmd = f"source ~/.zshrc && conda activate {conda_env} && {cmd}"
    return subprocess.run(
        full_cmd, shell=True, executable="/bin/zsh", capture_output=True, text=True
    )


def get_python_version(conda_env):
    result = run_in_conda("python --version", conda_env)
    match = re.search(r"Python (\d+)\.(\d+)", result.stdout)
    if match:
        return f"cp{match.group(1)}{match.group(2)}"
    raise RuntimeError(f"Could not detect Python version in {conda_env}")


def get_nightly_dates(conda_env, start_date=None, end_date=None):
    result = subprocess.run(["curl", "-s", PYTORCH_NIGHTLY_URL], capture_output=True, text=True)

    py_ver = get_python_version(conda_env)
    date_version_map = {}
    pattern = rf"torch-(\d+\.\d+\.\d+)\.dev(\d{{8}})\+{CUDA_VERSION}-{py_ver}-{py_ver}-{MANYLINUX_VERSION}\.whl"

    for match in re.finditer(pattern, result.stdout):
        version, date = match.groups()
        if date not in date_version_map:
            date_version_map[date] = version

    all_dates = sorted(date_version_map.keys())

    if start_date and start_date not in all_dates:
        raise ValueError(
            f"Start date {start_date} not found in available nightly builds for {py_ver}"
        )
    if end_date and end_date not in all_dates:
        raise ValueError(f"End date {end_date} not found in available nightly builds for {py_ver}")

    if start_date or end_date:
        start_idx = all_dates.index(start_date) if start_date else 0
        end_idx = all_dates.index(end_date) + 1 if end_date else len(all_dates)
        return (
            all_dates[start_idx:end_idx],
            {d: date_version_map[d] for d in all_dates[start_idx:end_idx]},
            set(all_dates),
        )

    return all_dates, date_version_map, set(all_dates)


def install_nightly(date, version, py_ver, conda_env="nightly"):
    torch_spec = f"torch=={version}.dev{date}+{CUDA_VERSION}"
    index_url = f"https://download.pytorch.org/whl/nightly/{CUDA_VERSION}"
    cmd = f"uv pip install --force-reinstall --index-url {index_url} '{torch_spec}'"
    result = run_in_conda(cmd, conda_env)
    if result.returncode != 0:
        print(f"    Install error: {result.stderr[:200]}")
    return result.returncode == 0


def get_commit_sha(conda_env="nightly"):
    result = run_in_conda("python -c 'import torch; print(torch.version.git_version)'", conda_env)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def test_pytorch(test_script, conda_env="nightly"):
    result = run_in_conda(f"python {test_script}", conda_env)
    return result.returncode == 0, result.stdout, result.stderr


def print_bisect_results(last_good, first_bad, last_good_sha, first_bad_sha, test_script, mode):
    print("\n" + "=" * 60)

    if last_good and first_bad:
        match mode:
            case "fix":
                print(
                    dedent(f"""
                    Bug was FIXED between:
                      Last bad:   {first_bad}
                      First good: {last_good}
                """).strip()
                )
                older_date, older_sha = first_bad, first_bad_sha
                newer_date, newer_sha = last_good, last_good_sha
                bisect_instructions = dedent(f"""
                    To bisect locally (finding when bug was FIXED):
                      cd <pytorch>
                      git bisect start
                      git bisect old <older_main_sha>   # Older commit - bug exists (test fails)
                      git bisect new <newer_main_sha>   # Newer commit - bug fixed (test passes)
                      # Build and test each commit:
                      python setup.py develop && python {test_script}
                      git bisect old   # If test FAILS (bug still exists)
                      git bisect new   # If test PASSES (bug is fixed)
                """).strip()
            case "regression":
                print(
                    dedent(f"""
                    Regression occurred between:
                      Last good: {last_good}
                      First bad: {first_bad}
                """).strip()
                )
                older_date, older_sha = last_good, last_good_sha
                newer_date, newer_sha = first_bad, first_bad_sha
                bisect_instructions = dedent(f"""
                    To bisect locally (finding regression):
                      cd <pytorch>
                      git bisect start
                      git bisect good <older_main_sha>  # Older commit worked (test passes)
                      git bisect bad <newer_main_sha>   # Newer commit is broken (test fails)
                      # Build and test each commit:
                      python setup.py develop && python {test_script}
                      git bisect good  # If test PASSES (working)
                      git bisect bad   # If test FAILS (broken)
                """).strip()

        # pyrefly: ignore  # unbound-name
        if older_sha and newer_sha:
            print(
                dedent(f"""

                Commit SHA range:
                  # pyrefly: ignore  # unbound-name
                  Older commit: {older_sha} ({older_date})
                  # pyrefly: ignore  # unbound-name
                  Newer commit: {newer_sha} ({newer_date})

                Note: These SHAs may be from the nightly release branch.
                Extract the actual main branch SHAs from the nightly release commits:
                  cd <pytorch>
                  # pyrefly: ignore  # unbound-name
                  git log --format='%H %s %b' {older_sha} -1
                  # pyrefly: ignore  # unbound-name
                  git log --format='%H %s %b' {newer_sha} -1
                  # Look for the main branch SHA in parentheses in the commit message

                # pyrefly: ignore  # unbound-name
                {bisect_instructions}
            """).strip()
            )

    elif last_good:
        match mode:
            case "fix":
                print(f"Bug is fixed in all tested versions starting from: {last_good}")
            case "regression":
                print("All tested versions are working (no regression found)")
        if last_good_sha:
            print(f"  Commit SHA: {last_good_sha}")

    elif first_bad:
        match mode:
            case "fix":
                print(f"Bug exists in all tested versions up to: {first_bad}")
            case "regression":
                print("All tested versions are broken")
        if first_bad_sha:
            print(f"  Commit SHA: {first_bad_sha}")

    else:
        match mode:
            case "fix":
                print("Could not determine when bug was fixed")
            case "regression":
                print("Could not determine when regression occurred")

    print("=" * 60)


def binary_search_fix(
    dates, version_map, all_dates_set, test_script, py_ver, conda_env="nightly", mode="fix"
):
    left, right = 0, len(dates) - 1
    last_good = None
    first_bad = None
    last_good_sha = None
    first_bad_sha = None
    max_tests = len(dates).bit_length()

    match mode:
        case "fix":
            mode_desc = "bug fix"
            pass_msg = "Bug is fixed"
            fail_msg = "Bug exists"
        case "regression":
            mode_desc = "regression"
            pass_msg = "Working"
            fail_msg = "Broken"

    print(
        dedent(f"""
        Testing {len(dates)} nightly builds from {dates[0]} to {dates[-1]}
        Using conda environment: {conda_env}
        Python version: {py_ver}
        Running test script: {test_script}
        # pyrefly: ignore  # unbound-name
        Mode: Finding when {mode_desc} occurred
    """).strip()
        + "\n"
    )

    pbar = tqdm(total=max_tests, desc="Bisecting", unit="test")

    while left <= right:
        mid = (left + right) // 2
        date = dates[mid]
        version = version_map[date]

        # pyrefly: ignore  # missing-attribute
        pbar.set_description(f"Testing {date} (torch-{version})")

        if not install_nightly(date, version, py_ver, conda_env):
            tqdm.write(f"  ⚠️  {date}: Skipping - install failed")
            left = mid + 1 if mid < right else left
            right = mid - 1 if mid >= right else right
            continue

        sha = get_commit_sha(conda_env)
        passed, _, _ = test_pytorch(test_script, conda_env)
        # pyrefly: ignore  # missing-attribute
        pbar.update(1)

        sha_display = sha[:8] if sha else "unknown"

        if passed:
            # pyrefly: ignore  # unbound-name
            tqdm.write(f"  ✅ {date}: PASSED - {pass_msg} (commit: {sha_display})")
            last_good = date
            last_good_sha = sha
            match mode:
                case "fix":
                    right = mid - 1
                case "regression":
                    left = mid + 1
        else:
            # pyrefly: ignore  # unbound-name
            tqdm.write(f"  ❌ {date}: FAILED - {fail_msg} (commit: {sha_display})")
            first_bad = date
            first_bad_sha = sha
            match mode:
                case "fix":
                    left = mid + 1
                case "regression":
                    right = mid - 1

    # pyrefly: ignore  # missing-attribute
    pbar.close()
    print_bisect_results(last_good, first_bad, last_good_sha, first_bad_sha, test_script, mode)
    return last_good_sha, first_bad_sha


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Bisect PyTorch nightly builds to find when a bug was fixed or when a regression occurred"
    )
    parser.add_argument(
        "test_script",
        help="Path to test script (exits with 0 if passing, non-zero if failing)",
    )
    parser.add_argument(
        "--conda-env", default="nightly", help="Conda environment to use (default: nightly)"
    )
    parser.add_argument("--start", help="Start date (YYYYMMDD) - must exist in nightly builds")
    parser.add_argument("--end", help="End date (YYYYMMDD) - must exist in nightly builds")
    parser.add_argument(
        "--mode",
        choices=["fix", "regression"],
        default="fix",
        help="Mode: 'fix' to find when bug was fixed, 'regression' to find when it broke (default: fix)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_script):
        print(f"Error: Test script not found: {args.test_script}")
        exit(1)

    print("Detecting Python version in conda environment...")
    py_ver = get_python_version(args.conda_env)
    print(f"Python version: {py_ver}")

    print("\nFetching available nightly builds...")
    dates, version_map, all_dates_set = get_nightly_dates(args.conda_env, args.start, args.end)

    print(
        dedent(f"""
        ✓ Found {len(dates)} nightly builds available for {py_ver}
          Date range: {dates[0]} to {dates[-1]}
          Binary search will require ~{len(dates).bit_length()} tests
    """).strip()
        + "\n"
    )

    binary_search_fix(
        dates,
        version_map,
        all_dates_set,
        os.path.abspath(args.test_script),
        py_ver,
        args.conda_env,
        args.mode,
    )
