"""Flow2 wheel route bookkeeping.

identity BUILD BUILD_DIR
    Print the identity of the exact wheels a validation installs: build name,
    source heads, wheel names and their observed SHA-256, receipt hash.
report JOB_ROOT --build BUILD --validate RUN [--report PATH]
    Collect every PBS job ID and terminal state, verify the build seal, the
    validated wheel identity and the evidence hashes, and write result.json.
    Exits non-zero (still writing the report) when any gate fails.
"""
import argparse
import hashlib
import json
import pathlib
import sys


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def identity(build, d):
    d = pathlib.Path(d)
    receipt = json.loads((d / "receipt.json").read_text())
    wheels = sorted((d / "out" / "wheelhouse").glob("*.whl"))
    return {"build": build,
            "source": {"ngsolve": receipt["ngsolve"]["head"], "netgen": receipt["netgen"]["head"]},
            "lane": receipt["lane"], "static_occ": receipt.get("static_occ"),
            "wheels": [{"name": w.name, "sha256": sha(w)} for w in wheels],
            "receipt_sha256": sha(d / "receipt.json")}


def terminal(d):
    t = {}
    p = pathlib.Path(d) / "terminal.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            k, _, v = line.partition("=")
            t[k] = v
    return t


def report(root, build, validate):
    root = pathlib.Path(root)
    owner = json.loads((root / "owner.json").read_text())
    failures, jobs = [], []
    b = root / build
    for j in sorted(b.glob("job-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        jobs.append(dict(json.loads(j.read_text()), step=build))
    links = (b / "links.txt").read_text().splitlines() if (b / "links.txt").exists() else []
    tb = terminal(b)
    if tb.get("phase") != "complete" or tb.get("exit_code") != "0" or not (b / "sealed.txt").exists():
        failures.append(f"{build} not sealed complete: {tb}")
    ident = None
    try:
        ident = identity(build, b)
        listed = dict(line.split()[::-1] for line in (b / "wheel.sha256").read_text().splitlines())
        observed = {f"./{w['name']}": w["sha256"] for w in ident["wheels"]}
        if listed != observed or len(observed) != 2:
            failures.append(f"{build} wheel.sha256 {listed} != observed {observed}")
        if ident["source"] != {"ngsolve": owner["source_sha"], "netgen": owner["netgen_sha"]}:
            failures.append(f"{build} receipt source {ident['source']} differs from owner.json")
    except Exception as e:  # noqa: BLE001 - any missing piece is a failed gate
        failures.append(f"{build} identity: {e!r}")
    v = root / validate
    if (v / "job.json").exists():
        jobs.append(dict(json.loads((v / "job.json").read_text()), step=validate))
    else:
        failures.append(f"{validate} has no job.json")
    tv = terminal(v)
    if tv.get("phase") != "complete" or tv.get("exit_code") != "0":
        failures.append(f"{validate} not complete: {tv}")
    vw = json.loads((v / "validated-wheels.json").read_text()) if (v / "validated-wheels.json").exists() else None
    if vw != ident:
        failures.append(f"{validate} validated wheels differ from the sealed build")
    ev = {}
    if (v / "evidence.sha256").exists():
        for line in (v / "evidence.sha256").read_text().splitlines():
            h, f = line.split(maxsplit=1)
            f = f.lstrip("*")
            ok = (v / f).exists() and sha(v / f) == h
            ev[f] = h
            if not ok:
                failures.append(f"{validate}/{f} hash mismatch")
    else:
        failures.append(f"{validate} has no evidence.sha256")
    if any(not j.get("pbs_jobid") for j in jobs):
        failures.append("a run lacks its PBS job ID")
    return {"schema": "ngsolve-fork-flow2-wheels/1", "machine": owner["machine"],
            "source": {"ngsolve": owner["source_sha"], "netgen": owner["netgen_sha"]},
            "jobs": jobs, "build_links": links, "build": ident, "validated": vw,
            "evidence_sha256": ev, "failures": failures, "pass": not failures}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("identity"); i.add_argument("build"); i.add_argument("dir")
    r = sub.add_parser("report"); r.add_argument("root"); r.add_argument("--build", required=True)
    r.add_argument("--validate", required=True); r.add_argument("--report")
    a = ap.parse_args()
    if a.cmd == "identity":
        json.dump(identity(a.build, a.dir), sys.stdout, indent=2)
        return 0
    res = report(a.root, a.build, a.validate)
    out = pathlib.Path(a.report or pathlib.Path(a.root) / "result.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    for f in res["failures"]:
        print("FAIL", f, file=sys.stderr)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
