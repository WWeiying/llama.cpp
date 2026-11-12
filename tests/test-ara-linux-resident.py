#!/usr/bin/env python3
"""Host syscall tests and GNU RV64 compilation; no hardware or full model run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ara", type=Path, default=Path("/home/wangwy/openproject/ara_dsa"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cross-prefix", default="/home/wangwy/llama/platforms/cva6-qemu/tools/bin/riscv64-linux-")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ara = args.ara.resolve()
    out = args.out.resolve()
    if out.parent != Path("/home/wangwy/openproject/ara_dsa_signoff_integration/verification/out") or not out.name.startswith("resident_deployment_ggml_"):
        parser.error("--out must be a new signoff verification/out/resident_deployment_ggml_* directory")
    out.mkdir(exist_ok=False)
    steps = []

    def run(name, cmd, cwd=None, ok=True):
        with (out / (name + ".log")).open("w") as log:
            log.write(shlex.join(map(str, cmd)) + "\n")
            log.flush()
            process = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        steps.append({"name": name, "returncode": process.returncode})
        (out / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
        if ok and process.returncode:
            raise RuntimeError("failed: " + name + "; see " + str(out / (name + ".log")))
        return process.returncode

    def archive(repo, dest, name, paths):
        dest.mkdir(parents=True)
        with (out / (name + ".tar")).open("wb") as data:
            subprocess.run(["git", "-C", str(repo), "archive", "HEAD", *paths], stdout=data, check=True)
        with tarfile.open(out / (name + ".tar")) as data:
            data.extractall(dest)

    llama = out / "inputs/llama"
    sdk = out / "inputs/sdk"
    archive(root, llama, "llama-head", [])
    archive(ara, sdk, "ara-head-akv-qbs", ["software/akv", "software/qbs"])
    # Preserve the existing research fork in the build input, not a development worktree.
    changed = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--modified", "--others", "--exclude-standard", "ggml", "cmake", "tests/test-ara-linux-resident.cpp", "tests/test-ara-linux-resident.py"]).decode().split("\0")
    origins = {}
    for relative in changed:
        source = root / relative
        if relative and source.is_file():
            target = llama / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            origins[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    linux = sdk / "software/llama/platform/linux"
    linux.mkdir(parents=True)
    for name in ("client.c", "client.h", "client_rv64.S", "ara_resident_uapi.h", "job_layout.h"):
        source = ara / "software/llama/platform/linux" / name
        shutil.copy2(source, linux / name)
        origins[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source in (root / "AGENTS.md", ara / "AGENTS.md", Path(__file__).resolve()):
        origins[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "inputs").rglob("*") if p.is_file()}
    (out / "source-pins.json").write_text(json.dumps({**origins, **pins}, sort_keys=True, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "llama_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "ara_head": subprocess.check_output(["git", "-C", str(ara), "rev-parse", "HEAD"], text=True).strip(),
        "sdk": "git archive HEAD software/akv software/qbs; only the new Linux client files are overlaid; no local AKV assembly",
        "scope": "host mocks and true-source compilation only; no board acceptance or cost claim",
    }, indent=2) + "\n")
    includes = ["-I" + str(llama / p) for p in ("ggml/include", "ggml/src", "ggml/src/ggml-cpu")]
    includes += ["-I" + str(linux)]
    scalar = ["-fno-tree-vectorize", "-fno-tree-slp-vectorize", "-ffreestanding", "-fno-builtin", "-fno-fast-math", "-ffp-contract=off"]
    warnings = ["-Wall", "-Wextra", "-Werror", "-Wno-unused-function"]
    for mode in ("host", "sanitized"):
        sanitize = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "sanitized" else []
        cc, cxx = ("clang", "clang++") if mode == "sanitized" else ("cc", "g++")
        mode_scalar = ["-fno-vectorize", "-fno-slp-vectorize", *scalar[2:]] if mode == "sanitized" else scalar
        cxx_scalar = [flag for flag in mode_scalar if flag != "-ffreestanding"]
        client = out / (mode + "-client.o")
        binary = out / (mode + "-test")
        run(mode + "-client", [cc, "-O2", "-std=gnu11", *warnings, *mode_scalar, *sanitize, "-DARA_RESIDENT_CLIENT_TEST=1", "-I" + str(linux), "-c", linux / "client.c", "-o", client])
        run(mode + "-build", [cxx, "-O2", "-std=c++17", *warnings, *cxx_scalar, *sanitize, "-DGGML_RISCV_ACCEL_AUTO=1", *includes, llama / "tests/test-ara-linux-resident.cpp", client, "-o", binary])
        run(mode + "-run", [binary])

    config = ["cmake", "-G", "Ninja", "-S", llama,
              "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=riscv64",
              "-DCMAKE_C_COMPILER=" + args.cross_prefix + "gcc",
              "-DCMAKE_CXX_COMPILER=" + args.cross_prefix + "g++",
              "-DCMAKE_ASM_COMPILER=" + args.cross_prefix + "gcc",
              "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", "-DCMAKE_BUILD_TYPE=Release",
              "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_RVV=ON", "-DGGML_RV_ZFH=ON", "-DGGML_RV_ZVFH=ON",
              "-DGGML_RISCV_AKV=ON", "-DGGML_RISCV_ACCEL_AUTO=ON",
              "-DGGML_RISCV_AKV_RUNTIME_DIR=" + str(sdk / "software/akv"),
              "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
              "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF"]
    audit = []
    for mode, options in (
        ("default-off", []),
        ("native-portal", ["-DGGML_RISCV_AKV_LINUX_PORTAL=ON"]),
        ("resident-portal", ["-DGGML_RISCV_AKV_RESIDENT=ON", "-DGGML_RISCV_AKV_LINUX_PORTAL=ON"]),
        ("quant-portal", ["-DGGML_RISCV_QBS=ON", "-DGGML_RISCV_QBS_RUNTIME_DIR=" + str(sdk / "software/qbs"), "-DGGML_RISCV_AKV_QUANT_KV=ON", "-DGGML_RISCV_AKV_LINUX_PORTAL=ON", "-DGGML_RISCV_AKV_LINUX_PORTAL_DIR=" + str(linux)]),
    ):
        build = out / mode
        run(mode + "-configure", [*config, "-B", build, *options])
        commands = json.loads((build / "compile_commands.json").read_text())
        selected = [c for c in commands if Path(c["file"]).name in ("akv.cpp", "client.c", "client_rv64.S")]
        clients = [c for c in selected if Path(c["file"]).name != "akv.cpp"]
        assert bool(clients) == (mode != "default-off")
        assert len(selected) == (1 if mode == "default-off" else 3)
        for command in selected:
            parts = shlex.split(command["command"])
            if Path(command["file"]).name == "client.c":
                assert [p for p in parts if p.startswith("-march=")][-1] == "-march=rv64gc"
                assert [p for p in parts if p.startswith("-mabi=")][-1] == "-mabi=lp64d"
                assert all(flag in parts for flag in scalar)
            if Path(command["file"]).name == "client_rv64.S":
                assert [p for p in parts if p.startswith("-march=")][-1] == "-march=rv64gcv"
                assert [p for p in parts if p.startswith("-mabi=")][-1] == "-mabi=lp64d"
            object_file = build / (Path(command["file"]).name + ".o")
            parts[parts.index("-o") + 1] = str(object_file)
            run(mode + "-" + Path(command["file"]).name, parts, cwd=build)
            run(mode + "-" + Path(command["file"]).name + "-disassembly", [args.cross_prefix + "objdump", "-dr", object_file])
            audit.append({"mode": mode, "file": command["file"], "command": parts})
    (out / "compile-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # The scalar client must not acquire compiler-generated FP/RVV instructions.
    disassembly = (out / "native-portal-client.c-disassembly.log").read_text()
    assert not re.search(r"\t(?:v[a-z][a-z0-9_.]*|f(?:l[wdh]|s[wdh]|add|sub|mul|div|madd|mv|cvt)[a-z0-9_.]*)\s", disassembly)
    entry = (out / "native-portal-client_rv64.S-disassembly.log").read_text()
    assert entry.index("vstart") < entry.index("ara_resident_linux_execute_scalar")
    assert "vcsr" in entry and "frcsr" in entry and "fscsr" in entry
    # Retain provider gating and try the portal at both public entry definitions.
    akv = (llama / "ggml/src/ggml-cpu/arch/riscv/akv.cpp").read_text()
    entries = akv.split("bool ggml_riscv_akv_flash_attn(const ggml_compute_params * params, ggml_tensor * dst) {")[1:]
    assert len(entries) == 2
    assert all("ggml_akv_linux::flash_attn(params, dst)" in entry.split("#endif", 1)[0] for entry in entries)
    assert entries[0].index("ggml_akv_linux::flash_attn") < entries[0].index("ggml_akv_platform_entry_v1") < entries[0].index("stage5_native")
    portal = (llama / "ggml/src/ggml-cpu/arch/riscv/linux-resident.inc").read_text()
    assert not any(token in portal for token in ("supports_decode(", "device()", "akv_native_info", "akv_resident_platform_"))
    for path, digest in origins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source drift: " + path)
    (out / "result.json").write_text(json.dumps({"status": "PASS", "steps": steps, "board_run": False, "full_model_run": False, "performance_measured": False}, indent=2) + "\n")
    print("PASS: host + ASan/UBSan mocks; GNU true-source default-off/native/resident/quant compilation; scalar/ASM and entry audits")
    print(out)


if __name__ == "__main__":
    main()
