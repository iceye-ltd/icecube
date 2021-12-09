"""
Note: Change variables in `deployment_conf.py` before using these tasks
Docker specific tasks are prefixed with a d. docker run > drun
"""

import os
from invoke import task

PYTEST = "python3.8 -m pytest"
PYTEST_COVERAGE = "-m pytest"
PYTEST_FLAGS = "--color yes --durations 3 -ra --failed-first -x"


@task
def setup(c):
    """Setup development/testing environment"""

    c.run("pip install -U pip invoke")
    c.run("pip install -e '.[dev]'")

    if not os.getenv("CI"):
        c.run(
            "(pre-commit install && pre-commit install -t pre-push) || echo Could not install pre-commit"
        )
        c.run("(type pyenv > /dev/null 2>&1 && pyenv rehash) || true")
        print("Setting up pre-commit..")
        c.run("pre-commit autoupdate")
    print("Done. Please commit '.pre-commit-config.yaml' if it has changed!")


@task
def test(c):
    """Run tests"""
    c.run(f"{PYTEST} {PYTEST_FLAGS}")


@task
def test_coverage(c):
    """Run tests"""
    c.run(f"coverage run {PYTEST_COVERAGE} {PYTEST_FLAGS}")
    c.run("coverage html")


@task
def fmt(c):
    """
    Format the code using the pre-commit hooks - blake, autoflake
    """
    c.run("pre-commit run  --color=always --all-file")


@task
def doc_local(c):
    """Build or re-generate the documentation."""
    cmd = "mkdocs serve"
    print(cmd)
    c.run(cmd)


@task
def doc_deploy(c):
    """Build or re-generate the documentation."""
    cmd = "mkdocs gh-deploy"
    print(cmd)
    c.run(cmd)
