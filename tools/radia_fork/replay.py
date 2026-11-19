"""Replay a pending fork patch series onto moved fork heads, then check it.

A series that was qualified on an older Netgen/NGSolve pair lands on the
current fork heads in two steps, Netgen first:

  replay.py netgen  NETGEN_REPO  OLD_BASE OLD_TIP NEW_BASE
  replay.py ngsolve NGSOLVE_REPO OLD_BASE OLD_TIP NEW_BASE --map OLD=NEW ...
  replay.py check   --netgen REPO OLD_BASE OLD_TIP NEW_BASE NEW_TIP
                    --ngsolve REPO OLD_BASE OLD_TIP NEW_BASE NEW_TIP

`netgen` and `ngsolve` create commits only (no branch, ref, push or
checkout change) and print one JSON object with the old -> new map. For
NGSolve every replayed commit points `external_dependencies/netgen` at the
image of its original gitlink under --map (the netgen step's output);
unmapped gitlinks are kept. Pass --map for each Netgen commit that an
NGSolve commit in the series references.

`check` is independent of how the replay was made. Both ranges must be
nonempty linear chains starting at their stated bases. Per commit it
requires: same count and order, identical message (trailing newlines
ignored) and author, and a tree equal to the original patch (Netgen
gitlink excluded) applied exactly, without merging, onto the replayed
parent; for NGSolve the gitlink must equal the Netgen image of the
original one and be reachable from the new Netgen tip. It exits non-zero
on the first failure.

The committer is the caller's git identity (set user.name/user.email).

Exit codes: 0 success, 1 check failure or invalid range, 2 conflict or
git/usage error.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

SUB = "external_dependencies/netgen"


def git(repo, *args, env=None, stdin=None):
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, *args],
                       input=stdin, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def series(repo, base, tip):
    """Nonempty linear chain base -> ... -> tip; anything else is refused."""
    base_c = git(repo, "rev-parse", f"{base}^{{commit}}")
    tip_c = git(repo, "rev-parse", f"{tip}^{{commit}}")
    out = git(repo, "rev-list", "--reverse", "--topo-order", f"{base_c}..{tip_c}")
    commits = out.split() if out else []
    if not commits:
        raise AssertionError(f"{repo}: empty range {base}..{tip}")
    parent = base_c
    for c in commits:
        if git(repo, "show", "-s", "--format=%P", c) != parent:
            raise AssertionError(f"{repo}: {c} is not a single-parent child of {parent}; "
                                 f"{base}..{tip} is not a linear series from {base}")
        parent = c
    return commits


def expected_tree(repo, commit, new_parent, link):
    """Tree of `commit`'s exact patch applied onto new_parent, gitlink set to link."""
    fd, idx = tempfile.mkstemp(prefix="radia-check-", suffix=".idx")
    os.close(fd)
    env = {"GIT_INDEX_FILE": idx}
    try:
        git(repo, "read-tree", new_parent, env=env)
        body = patch(repo, commit)
        if body:
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "apply", "--cached"],
                               input=body, capture_output=True, env={**os.environ, **env})
            if r.returncode:
                return None
        if link:
            git(repo, "update-index", "--cacheinfo", f"160000,{link},{SUB}", env=env)
        return git(repo, "write-tree", env=env)
    finally:
        os.unlink(idx)


def gitlink(repo, commit):
    line = git(repo, "ls-tree", commit, SUB)
    return line.split()[2] if line.startswith("160000 ") else None


def patch(repo, commit):
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo, "diff", "--binary",
         f"{commit}~1", commit, "--", ".", f":(exclude){SUB}"],
        capture_output=True, check=True).stdout


def author_env(repo, c):
    name, email, date = git(repo, "show", "-s", "--format=%an%x00%ae%x00%ad", "--date=raw", c).split("\x00")
    return {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date}


def replay(repo, old_base, old_tip, new_base, links):
    """Apply each commit's patch (gitlink excluded) onto the new chain."""
    mapping, parent = {}, git(repo, "rev-parse", f"{new_base}^{{commit}}")
    with tempfile.TemporaryDirectory(prefix="radia-replay-") as tmp:
        wt = os.path.join(tmp, "wt")
        git(repo, "worktree", "add", "-q", "--detach", wt, parent)
        try:
            for c in series(repo, old_base, old_tip):
                body = patch(repo, c)
                if body:
                    r = subprocess.run(["git", "-C", wt, "apply", "--index", "--3way", "--whitespace=nowarn"],
                                       input=body, capture_output=True)
                    if r.returncode:
                        sys.stderr.write(r.stderr.decode())
                        raise SystemExit(2)
                old_link = gitlink(repo, c)
                if old_link and links is not None:
                    new_link = links.get(old_link, old_link)
                    git(wt, "update-index", "--cacheinfo", f"160000,{new_link},{SUB}")
                tree = git(wt, "write-tree")
                raw = subprocess.run(["git", "-C", repo, "cat-file", "commit", c],
                                     capture_output=True, check=True).stdout
                msg = raw.split(b"\n\n", 1)[1].decode()  # message bytes as-is
                new = git(wt, "commit-tree", tree, "-p", parent, env=author_env(repo, c), stdin=msg)
                git(wt, "checkout", "-q", "--detach", new)
                mapping[c], parent = new, new
        finally:
            git(repo, "worktree", "remove", "--force", wt)
    return {"base": new_base, "tip": parent, "map": mapping}


def meta(repo, c):
    msg, rest = git(repo, "show", "-s", "--format=%B%x00%an%x00%ae%x00%ad", "--date=raw", c).split("\x00", 1)
    return msg.rstrip("\n"), rest


def check(repo, old_base, old_tip, new_base, new_tip, netgen=None):
    old, new = series(repo, old_base, old_tip), series(repo, new_base, new_tip)
    if len(old) != len(new):
        raise AssertionError(f"{repo}: {len(old)} original commits, {len(new)} replayed")
    parent = git(repo, "rev-parse", f"{new_base}^{{commit}}")
    rows = []
    for a, b in zip(old, new):
        checks = {
            "parent": git(repo, "show", "-s", "--format=%P", b) == parent,
            "message_author": meta(repo, a) == meta(repo, b),
        }
        row = {"old": a, "new": b}
        want = None
        if netgen is not None:
            ng_repo, ng_map, ng_tip = netgen
            la, lb = gitlink(repo, a), gitlink(repo, b)
            want = ng_map.get(la, la)
            checks["gitlink"] = lb == want and subprocess.run(
                ["git", "-C", ng_repo, "merge-base", "--is-ancestor", lb, ng_tip]).returncode == 0
            row.update(gitlink_old=la, gitlink_new=lb)
        checks["tree"] = expected_tree(repo, a, parent, want) == git(repo, "rev-parse", f"{b}^{{tree}}")
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise AssertionError(f"{a} -> {b}: {', '.join(bad)} differ")
        rows.append(row)
        parent = b
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("netgen", "ngsolve"):
        q = sp.add_parser(name)
        q.add_argument("repo"); q.add_argument("old_base"); q.add_argument("old_tip"); q.add_argument("new_base")
        if name == "ngsolve":
            q.add_argument("--map", nargs="*", default=[], metavar="OLD=NEW")
    q = sp.add_parser("check")
    q.add_argument("--netgen", nargs=5, required=True, metavar=("REPO", "OLD_BASE", "OLD_TIP", "NEW_BASE", "NEW_TIP"))
    q.add_argument("--ngsolve", nargs=5, required=True, metavar=("REPO", "OLD_BASE", "OLD_TIP", "NEW_BASE", "NEW_TIP"))
    a = p.parse_args()
    try:
        if a.cmd == "netgen":
            out = replay(a.repo, a.old_base, a.old_tip, a.new_base, None)
        elif a.cmd == "ngsolve":
            links = {}
            for item in a.map:
                o, _, n = item.partition("=")
                if len(o) != 40 or len(n) != 40:
                    raise RuntimeError(f"--map takes full 40-hex SHAs: {item}")
                links[o] = n
            out = replay(a.repo, a.old_base, a.old_tip, a.new_base, links)
        else:
            ng = a.netgen
            ng_rows = check(*ng)
            ng_map = {r["old"]: r["new"] for r in ng_rows}
            ngs_rows = check(*a.ngsolve, netgen=(ng[0], ng_map, ng[4]))
            out = {"netgen": ng_rows, "ngsolve": ngs_rows, "result": "pass"}
    except AssertionError as e:
        print(json.dumps({"result": "fail", "reason": str(e)}))
        return 1
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
