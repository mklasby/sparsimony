from subprocess import run


def install():
    run(["pre-commit", "install"], check=True)
