#!/usr/bin/env python3
"""Focused v2 successor: host admission mocks and real GNU objects, no model run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instructions(disassembly):
    decoded = {}
    for line in disassembly.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]+\s+([a-z][a-z0-9_.]*)\s*(.*)", line)
        if match:
            address, mnemonic, operands = match.groups()
            decoded[int(address, 16)] = (mnemonic, operands)
    return decoded


def preguard_audit(disassembly, symbols):
    code = instructions(disassembly)
    guards = [pc for pc, (op, args) in code.items() if op == "csrr" and "vstart" in args]
    assert len(guards) == 1, guards
    guard = guards[0]
    functions = []
    for line in symbols.splitlines():
        match = re.match(r"([0-9a-f]+) ([0-9a-f]+) [tT] (\S+)", line)
        if match:
            start, size, name = match.groups()
            functions.append((int(start, 16), int(start, 16) + int(size, 16), name))
    start, end, name = next(f for f in functions if f[0] <= guard < f[1])
    pcs = sorted(pc for pc in code if start <= pc < end)
    successors = {}
    for index, pc in enumerate(pcs):
        op, operands = code[pc]
        edges = [pcs[index + 1]] if index + 1 < len(pcs) and pc != guard else []
        target = re.search(r"(?:^|,)\s*([0-9a-f]+)\s+<", operands)
        if op == "j":
            assert target, (pc, operands)
            edges = [int(target[1], 16)]
        elif op.startswith("b") and target:
            edges.append(int(target[1], 16))
        elif op in ("ret", "jr", "unimp"):
            edges = []
        if pc == guard:
            edges = []
        successors[pc] = [edge for edge in edges if edge in code and start <= edge < end]

    def reachable(seed, graph):
        visited, todo = set(), [seed]
        while todo:
            pc = todo.pop()
            if pc not in visited:
                visited.add(pc)
                todo.extend(graph.get(pc, []))
        return visited

    reverse = {}
    for pc, edges in successors.items():
        for edge in edges:
            reverse.setdefault(edge, []).append(pc)
    before = reachable(start, successors) & reachable(guard, reverse)
    assert guard in before and len(before) > 50
    bad = [(hex(pc), *code[pc]) for pc in sorted(before) if code[pc][0].startswith("v")]
    assert not bad, (name, bad)
    calls = []
    for line in disassembly.splitlines():
        match = re.match(r"\s*([0-9a-f]+): R_RISCV_CALL_PLT\s+(\S+)", line)
        if match and int(match[1], 16) in before:
            calls.append({"pc": "0x" + match[1], "symbol": match[2]})
    assert any(c["symbol"] == "ara_resident_linux_open" for c in calls)
    helpers = [f for f in functions if "ggml_akv_linux" in f[2] and "tensor_span" in f[2]]
    assert helpers
    for lo, hi, helper in helpers:
        assert not any(op.startswith("v") for pc, (op, _) in code.items() if lo <= pc < hi), helper
    return {"function": name, "entry": hex(start), "guard": hex(guard), "preguard_instructions": len(before),
            "direct_rvv": bad, "calls_before_guard": calls,
            "scope": "CFG paths reaching the guard and tensor_span bodies; external libc implementations are not certified here"}


def scalar_client_audit(disassembly):
    code = instructions(disassembly)
    assert not any(op.startswith("v") or op.startswith("f") or op.startswith("csr") for op, _ in code.values())
    entries = [int(match[1], 16) for match in re.finditer(r"^([0-9a-f]+) <ara_resident_linux_\w+>:", disassembly, re.M)]
    assert len(entries) == 3
    checked = []
    for index, start in enumerate(entries):
        end = entries[index + 1] if index + 1 < len(entries) else max(code) + 4
        pcs = sorted(pc for pc in code if start <= pc < end)
        following = dict(zip(pcs, pcs[1:]))
        incoming, pending = {start: frozenset()}, [start]
        while pending:
            pc = pending.pop()
            zeros = set(incoming[pc])
            op, args = code[pc]
            operands = args.split("#", 1)[0].strip().split(",")
            dest = operands[0].strip() if operands else ""
            # Linux ecall retains these scalar input registers. Ordinary calls do not.
            if op == "jalr":
                zeros.clear()
            elif dest in ("a3", "a4", "a5") and op not in ("sd", "sw", "sb", "sh", "ecall") and not op.startswith("b"):
                zeros.discard(dest)
                if op == "li" and len(operands) == 2 and operands[1].strip() == "0":
                    zeros.add(dest)
                if op == "mv" and len(operands) == 2 and operands[1].strip() == "zero":
                    zeros.add(dest)
            edges = [following[pc]] if pc in following else []
            target = re.search(r"(?:^|,)\s*([0-9a-f]+)\s+<", args)
            if op == "j":
                assert target
                edges = [int(target[1], 16)]
            elif op.startswith("b") and target:
                edges.append(int(target[1], 16))
            elif op in ("ret", "jr", "unimp"):
                edges = []
            for edge in edges:
                if edge not in following and edge not in pcs:
                    continue
                merged = frozenset(zeros) if edge not in incoming else incoming[edge] & zeros
                if edge not in incoming or merged != incoming[edge]:
                    incoming[edge] = merged
                    pending.append(edge)
        for pc in pcs:
            if code[pc][0] == "ecall":
                assert incoming.get(pc) == {"a3", "a4", "a5"}, (hex(pc), incoming.get(pc))
                checked.append(hex(pc))
    assert len(checked) >= 5
    return {"scalar_no_csr": True, "ecalls_with_zero_a3_a5_on_all_cfg_paths": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Immutable sealed05 output")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ara", type=Path, default=Path("/home/wangwy/openproject/ara_dsa"))
    parser.add_argument("--cross-prefix", default="/home/wangwy/llama/platforms/cva6-qemu/tools/bin/riscv64-linux-")
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    if out.parent != Path("/home/wangwy/openproject/ara_dsa_signoff_integration/verification/out") or not out.name.startswith("resident_deployment_ggml_"):
        parser.error("--out must be a new signoff verification/out/resident_deployment_ggml_* directory")
    assert json.loads((base / "result.json").read_text())["status"] == "PASS"
    base_pins = json.loads((base / "source-pins.json").read_text())
    for path, expected in base_pins.items():
        if Path(path).is_relative_to(base / "inputs"):
            assert digest(Path(path)) == expected, "sealed input drift: " + path
    out.mkdir(exist_ok=False)
    shutil.copytree(base / "inputs", out / "inputs")
    llama, sdk = out / "inputs/llama", out / "inputs/sdk"
    origins = {}
    for relative in ("ggml/src/ggml-cpu/CMakeLists.txt", "ggml/src/ggml-cpu/arch/riscv/linux-resident.inc",
                     "tests/test-ara-linux-resident-v2.cpp", "tests/test-ara-linux-resident-v2.py"):
        source = root / relative
        shutil.copy2(source, llama / relative)
        origins[str(source)] = digest(source)
    linux = sdk / "software/llama/platform/linux"
    for name in ("client_v2.c", "client.h", "ara_resident_uapi.h", "job_layout.h", "client_rv64.S"):
        source = args.ara / "software/llama/platform/linux" / name
        shutil.copy2(source, linux / name)
        origins[str(source)] = digest(source)
    assert digest(linux / "client_rv64.S") == digest(base / "inputs/sdk/software/llama/platform/linux/client_rv64.S")
    for source in (root / "AGENTS.md", args.ara / "AGENTS.md"):
        origins[str(source)] = digest(source)
    pins = {str(path): digest(path) for path in (out / "inputs").rglob("*") if path.is_file()}
    (out / "source-pins.json").write_text(json.dumps({**origins, **pins}, sort_keys=True, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({"base": str(base), "base_pins_sha256": digest(base / "source-pins.json"),
        "sdk": "Reuse sealed05 git-archive SDK, overlay only current Linux client inputs; no local AKV assembly",
        "scope": "Focused v2 host mocks and native/resident entry objects only; not a repeat of the four-configuration campaign"}, indent=2) + "\n")
    steps = []

    def run(name, cmd, cwd=None):
        with (out / (name + ".log")).open("w") as log:
            log.write(shlex.join(map(str, cmd)) + "\n")
            log.flush()
            rc = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
        steps.append({"name": name, "returncode": rc})
        (out / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
        if rc:
            raise RuntimeError("failed: " + str(out / (name + ".log")))
        return (out / (name + ".log")).read_text()

    scalar = ["-fno-tree-vectorize", "-fno-tree-slp-vectorize", "-ffreestanding", "-fno-builtin", "-fno-fast-math", "-ffp-contract=off"]
    warnings = ["-Wall", "-Wextra", "-Werror", "-Wno-unused-function"]
    run("host-client-v2", ["cc", "-O2", "-std=gnu11", *warnings, *scalar, "-DARA_RESIDENT_CLIENT_TEST=1", "-I" + str(linux), "-c", linux / "client_v2.c", "-o", out / "host-client-v2.o"])
    includes = ["-I" + str(llama / p) for p in ("ggml/include", "ggml/src", "ggml/src/ggml-cpu")]
    run("host-build", ["g++", "-O2", "-std=c++17", *warnings, *[f for f in scalar if f != "-ffreestanding"], "-DGGML_RISCV_ACCEL_AUTO=1", *includes, "-I" + str(linux), llama / "tests/test-ara-linux-resident-v2.cpp", out / "host-client-v2.o", "-o", out / "host-test-v2"])
    run("host-run", [out / "host-test-v2"])
    config = ["cmake", "-G", "Ninja", "-S", llama, "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=riscv64",
              "-DCMAKE_C_COMPILER=" + args.cross_prefix + "gcc", "-DCMAKE_CXX_COMPILER=" + args.cross_prefix + "g++",
              "-DCMAKE_ASM_COMPILER=" + args.cross_prefix + "gcc", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
              "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_RVV=ON",
              "-DGGML_RV_ZFH=ON", "-DGGML_RV_ZVFH=ON", "-DGGML_RISCV_AKV=ON", "-DGGML_RISCV_ACCEL_AUTO=ON",
              "-DGGML_RISCV_AKV_RUNTIME_DIR=" + str(sdk / "software/akv"), "-DGGML_RISCV_AKV_LINUX_PORTAL=ON",
              "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
              "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF"]
    audit, commands_run = {}, []
    for resident in (False, True):
        mode = "resident" if resident else "native"
        build = out / mode
        run(mode + "-configure", [*config, "-B", build, "-DGGML_RISCV_AKV_RESIDENT=" + ("ON" if resident else "OFF")])
        commands = json.loads((build / "compile_commands.json").read_text())
        assert not any(Path(c["file"]).name == "client.c" for c in commands)
        assert any(Path(c["file"]).name == "client_v2.c" for c in commands)
        for command in commands:
            name = Path(command["file"]).name
            if name not in (("akv.cpp",) if resident else ("akv.cpp", "client_v2.c", "client_rv64.S")):
                continue
            parts = shlex.split(command["command"])
            if name == "client_v2.c":
                assert [p for p in parts if p.startswith("-march=")][-1] == "-march=rv64gc"
                assert [p for p in parts if p.startswith("-mabi=")][-1] == "-mabi=lp64d"
                assert all(flag in parts for flag in scalar)
            if name == "client_rv64.S":
                assert [p for p in parts if p.startswith("-march=")][-1] == "-march=rv64gcv"
                assert [p for p in parts if p.startswith("-mabi=")][-1] == "-mabi=lp64d"
            obj = build / (name + ".o")
            parts[parts.index("-o") + 1] = str(obj)
            run(mode + "-" + name, parts, cwd=build)
            commands_run.append(parts)
            disassembly = run(mode + "-" + name + "-disassembly", [args.cross_prefix + "objdump", "-dr", obj])
            if name == "akv.cpp":
                symbols = run(mode + "-symbols", [args.cross_prefix + "nm", "-S", "--defined-only", obj])
                audit[mode] = preguard_audit(disassembly, symbols)
            elif name == "client_v2.c":
                audit["client_v2"] = scalar_client_audit(disassembly)
            else:
                assert disassembly.index("vstart") < disassembly.index("ara_resident_linux_execute_scalar")
                assert all(token in disassembly for token in ("vcsr", "frcsr", "fscsr"))
    (out / "compile-audit.json").write_text(json.dumps(commands_run, indent=2) + "\n")
    (out / "preguard-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for path, expected in {**origins, **pins}.items():
        assert digest(Path(path)) == expected, "source drift: " + path
    result = {"status": "PASS", "steps": steps, "audit": audit, "board_run": False, "full_model_run": False,
              "caller_entry_vstart_preserved": False,
              "vstart_scope": "Policy/hwprobe before CSR, no direct preguard RVV; Linux syscalls can discard caller vector state"}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: v2 admission host cases, two GNU entry objects, scalar client/ASM and CFG preguard audits")
    print(out)


if __name__ == "__main__":
    main()
