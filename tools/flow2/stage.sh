#!/bin/bash
# Stage a Flow2 job root on the login node: exact NGSolve and Netgen sources,
# local version tags, prefetched source archives, hash-locked wheelhouses,
# owner.json and inputs.sha256. Compute jobs then run without network access.
# Usage: stage.sh NGSOLVE_SRC NETGEN_SRC NGSOLVE_SHA JOB_ROOT OWNER REQUEST DEADLINE_HOURS MACHINE
#   *_SRC    git URL or bundle containing the commit (Netgen: the NGSolve gitlink)
#   MACHINE  aarch64 (Type II, CUDA + static OpenCASCADE + site NVPL) or x86_64 (Type I, CPU, MKL)
set -Eeuo pipefail
[ $# -eq 8 ] || { sed -n '2,7p' "$0"; exit 2; }
ngs_src=$1 ng_src=$2 sha=$3 root=$4 owner=$5 request=$6 hours=$7 machine=$8
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "NGSOLVE_SHA must be a full 40-hex commit" >&2; exit 2; }
case $machine in
  aarch64) profile=cuda; scope="Flow2 Type II Linux GPU wheels (t2-dev build, t2-share GB200 checks)" ;;
  x86_64) profile=cpu; scope="Flow2 Type I Linux CPU wheels, reference agreement and 1/16/48-thread timing" ;;
  *) echo "MACHINE must be aarch64 or x86_64" >&2; exit 2 ;;
esac
mkdir "$root"   # fails if the root exists: never reuse or overwrite a job root
umask 077
git clone -q --no-checkout "$ngs_src" "$root/source"
git -C "$root/source" checkout -q --detach "$sha"
test "$(git -C "$root/source" rev-parse HEAD)" = "$sha"
ng_sha=$(git -C "$root/source" ls-tree HEAD external_dependencies/netgen | awk '{print $3}')
[[ $ng_sha =~ ^[0-9a-f]{40}$ ]] || { echo "no Netgen gitlink" >&2; exit 1; }
ng=$root/source/external_dependencies/netgen
rmdir "$ng" 2>/dev/null || true
git clone -q --no-checkout "$ng_src" "$ng"
git -C "$ng" checkout -q --detach "$ng_sha"
git -C "$ng" submodule -q update --init --recursive   # pybind11 (network, login node only)
# PEP 440 local version from an exact tag, as in the Windows candidate builds
git -C "$ng" tag -f "v6.2.2608+radia.${ng_sha:0:8}" > /dev/null
git -C "$root/source" tag -f "v6.2.2608+radia.${sha:0:9}.$profile" > /dev/null
for r in "$root/source" "$ng"; do
  [ -z "$(git -C "$r" status --porcelain --ignore-submodules=none)" ] || { echo "$r is not clean" >&2; exit 1; }; done
here=$root/source/tools/flow2
# Netgen's superbuild zlib; *.zip is ignored by Netgen's .gitignore
fetch() { # url md5 dest
  curl -fsSL -o "$3" "$1"; echo "$2  $3" | md5sum -c --quiet -; }
fetch https://github.com/madler/zlib/archive/refs/tags/v1.3.1.zip 127b8a71a3fb8bebe89df1080f15fdf6 "$ng/external_dependencies/v1.3.1.zip"
if [ "$machine" = aarch64 ]; then
  # the OpenCASCADE archive and MD5 of Netgen's own BUILD_OCC superbuild
  mkdir "$root/occ"
  fetch https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V8_0_1.zip 5b0b171d7028cf73bd9369997091347a "$root/occ/V8_0_1.zip"
fi
platform="--only-binary=:all: --implementation cp --python-version 3.12 --abi cp312"
for tag in manylinux_2_34 manylinux_2_28 manylinux_2_27 manylinux_2_24 manylinux_2_17 manylinux2014; do
  platform+=" --platform ${tag}_$machine"; done
for set in build validate; do
  # shellcheck disable=SC2086
  python3.12 -m pip download -q $platform --no-deps --require-hashes \
    -r "$here/requirements-$set-$machine.txt" -d "$root/wheelhouse-$set"
done
printf '%s\n%s\n' "$sha" "$ng_sha" > "$root/source-heads.txt"
python3.12 - "$root" "$owner" "$request" "$sha" "$ng_sha" "$hours" "$machine" "$scope" <<'PY'
import datetime, json, sys
root, owner, request, sha, ng, hours, machine, scope = sys.argv[1:]
now = datetime.datetime.now(datetime.timezone.utc)
json.dump({"owner": owner, "request": request, "source_sha": sha, "netgen_sha": ng,
           "created_utc": now.isoformat(),
           "deadline_utc": (now + datetime.timedelta(hours=float(hours))).isoformat(),
           "machine": machine, "scope": scope},
          open(f"{root}/owner.json", "w"), indent=2)
PY
(cd "$root" && { find wheelhouse-build wheelhouse-validate -type f; ls occ/*.zip 2>/dev/null;
  echo source/external_dependencies/netgen/external_dependencies/v1.3.1.zip; echo source-heads.txt; echo owner.json; } \
  | sort | xargs sha256sum > inputs.sha256)
echo "staged $root at ngsolve $sha netgen $ng_sha ($machine)"
