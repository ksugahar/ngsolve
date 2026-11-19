"""Self-test for replay.py on throwaway repositories.

Guards: check accepts a faithful two-repo replay and refuses a replay
that only re-indents a line (same patch-id, different meaning), equal
empty ranges, and ranges that do not start at their stated bases.
Run: python -I tools/radia_fork/test_replay.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = [sys.executable, "-I", os.path.join(HERE, "replay.py")]
SUB = "external_dependencies/netgen"
ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.invalid",
       "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.invalid",
       "GIT_AUTHOR_DATE": "1700000000 +0000", "GIT_COMMITTER_DATE": "1700000000 +0000"}


def git(repo, *args, stdin=None):
    return subprocess.run(["git", "-C", repo, *args], input=stdin, text=True,
                          capture_output=True, check=True, env=ENV).stdout.strip()


def commit(repo, files, msg, link=None):
    for name, text in files.items():
        path = os.path.join(repo, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="\n") as f:
            f.write(text)
        git(repo, "add", name)
    if link:
        git(repo, "update-index", "--add", "--cacheinfo", f"160000,{link},{SUB}")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def run(*args):
    r = subprocess.run([*TOOL, *args], capture_output=True, text=True, env=ENV)
    return r.returncode, r.stdout


def main():
    with tempfile.TemporaryDirectory() as t:
        ng, ngs = os.path.join(t, "netgen"), os.path.join(t, "ngsolve")
        for r in (ng, ngs):
            os.makedirs(r)
            git(r, "init", "-q", "-b", "main")
            git(r, "config", "core.autocrlf", "false")
        # Netgen: old base, one patch; upstream moved elsewhere
        n0 = commit(ng, {"a.txt": "1\n"}, "base")
        n1 = commit(ng, {"p.txt": "patch\n"}, "netgen patch")
        git(ng, "checkout", "-q", "-b", "moved", n0)
        m0 = commit(ng, {"u.txt": "upstream\n"}, "upstream")
        # NGSolve: a two-commit series pinning the Netgen base n0 (kept as is),
        # then the Netgen patch n1 (remapped to its replay)
        s0 = commit(ngs, {"f.py": "def f(c):\n    pass\n"}, "base", link=n0)
        code = "def g(c):\n    if c:\n        return 1\n    return 2\n"
        s1 = commit(ngs, {"g.py": code}, "add g", link=n0)
        s2 = commit(ngs, {"h.txt": "h\n"}, "pin patch", link=n1)
        git(ngs, "checkout", "-q", "-b", "moved", s0)
        t0 = commit(ngs, {"u.txt": "upstream\n"}, "upstream", link=m0)

        rc, out = run("netgen", ng, n0, n1, m0)
        assert rc == 0, out
        nmap = json.loads(out)
        n1r = nmap["tip"]
        rc, out = run("ngsolve", ngs, s0, s2, t0, "--map", f"{n1}={n1r}")
        assert rc == 0, out
        s2r = json.loads(out)["tip"]
        args = ["check", "--netgen", ng, n0, n1, m0, n1r, "--ngsolve", ngs, s0, s2, t0]
        rc, out = run(*args, s2r)
        assert rc == 0 and json.loads(out)["result"] == "pass", ("faithful replay", rc, out)

        # re-indent the last return into the if: patch-id equal, meaning not
        bad = "def g(c):\n    if c:\n        return 1\n        return 2\n"
        git(ngs, "checkout", "-q", "--detach", t0)
        b1 = commit(ngs, {"g.py": bad}, "add g", link=n0)
        git(ngs, "rm", "-q", "--cached", SUB)
        git(ngs, "update-index", "--add", "--cacheinfo", f"160000,{n1r},{SUB}")
        with open(os.path.join(ngs, "h.txt"), "w", newline="\n") as f:
            f.write("h\n")
        git(ngs, "add", "h.txt")
        git(ngs, "commit", "-q", "-m", "pin patch")
        b2 = git(ngs, "rev-parse", "HEAD")
        assert b1
        rc, out = run(*args, b2)
        assert rc == 1 and "tree" in out, ("re-indented replay must fail", rc, out)

        # both ranges empty
        rc, out = run("check", "--netgen", ng, n1, n1, m0, m0, "--ngsolve", ngs, s2, s2, t0, t0)
        assert rc == 1 and "empty range" in out, ("both empty must fail", rc, out)

        # wrong stated base: a range that does not start at it
        rc, out = run("check", "--netgen", ng, n0, n1, m0, n1r, "--ngsolve", ngs, s0, s2, s1, s2r)
        assert rc == 1 and "not a linear series" in out, ("wrong new base must fail", rc, out)
        rc, out = run("check", "--netgen", ng, n0, n1, m0, n1r, "--ngsolve", ngs, t0, s2, t0, s2r)
        assert rc == 1 and "not a linear series" in out, ("wrong old base must fail", rc, out)
    print("test_replay: 5/5 pass")


if __name__ == "__main__":
    main()
