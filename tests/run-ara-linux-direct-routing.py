#!/usr/bin/env python3
"""Build focused Direct-U host routing tests. Default: compile only, no sealing.

--run executes development checks only after the source owners authorize them.
The real routing entry, GGML host libraries, SDK plans, platform admission and
scalar reference are linked. Direct-U lifecycle and native execution are mocks.
No RTL, native arithmetic, kernel driver, hardware or performance is validated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


DEFAULT_SDK = Path("/home/wangwy/openproject/ara_dsa_signoff_integration/verification/out/"
                   "resident_direct_build_compact_20261113_01/inputs/sdk/software/akv")
CASES = {"unsupported": 13, "no-device": 1, "prepare-failure": 1, "begin-failure": 1,
         "matrix": 84, "policy": 16, "reuse": 2, "features": 2, "reference": 2,
         "prefill": 3, "legacy-required": 1, "admission-denied": 1, "state-denied": 2,
         "failed-release": 2, "quant-fallback": 4,
         "info4-absent": 4, "info4-pv-only": 4, "info4-qk-only": 4, "info4-version": 4,
         "info4-lanes": 4, "info4-vlen": 4, "info4-reserved": 4}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK, help="frozen SDK input directory, never live assembly")
    parser.add_argument("--out", type=Path, required=True, help="new agent-owned build output directory")
    parser.add_argument("--ggml-build", type=Path, help="reuse an existing static ordinary-host GGML build")
    parser.add_argument("--run", action="store_true", help="run non-sealing development checks after owner authorization")
    parser.add_argument("--case", choices=CASES, action="append", help="run only selected cases")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    if args.case and not args.run:
        parser.error("--case requires --run")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    sdk, out = args.sdk.resolve(), args.out.resolve()
    if not (sdk / "src/akv_runtime.c").is_file():
        parser.error("--sdk must contain the frozen software/akv directory")
    if sdk == Path("/home/wangwy/openproject/ara_dsa/software/akv").resolve():
        parser.error("use frozen SDK build inputs, not the dirty development SDK")
    out.mkdir(parents=True, exist_ok=False)
    steps = []
    report = dict(status="BUILDING", sealed=False, scope="ordinary-host routing mocks",
                  native_execution=False, rtl_run=False, board_run=False, performance_measured=False,
                  sdk=str(sdk), llama_repository=str(root), cases=[], case_results=[], steps=steps)

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, argv, timeout=300):
        argv = list(map(str, argv))
        with (out / (name + ".log")).open("w") as log:
            log.write(shlex.join(argv) + "\n")
            log.flush()
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                    env={**os.environ, "LC_ALL": "C"})
        steps.append(dict(name=name, returncode=result.returncode))
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {out / (name + '.log')}")

    try:
        # Copies are disposable build inputs, not worktrees or sealed campaign inputs.
        inputs = out / "inputs"
        inputs.mkdir()
        source = inputs / "ggml"
        shutil.copytree(root / "ggml", source)
        frozen_sdk = inputs / "sdk"
        shutil.copytree(sdk, frozen_sdk)
        test = inputs / "test-ara-linux-direct-routing.cpp"
        shutil.copy2(root / "tests" / test.name, test)
        shutil.copy2(Path(__file__), inputs / Path(__file__).name)
        (inputs / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.14)\n"
            "project(direct_routing_host LANGUAGES C CXX ASM)\n"
            "add_subdirectory(ggml)\n")
        hashes = {str(p.relative_to(inputs)): digest(p) for p in inputs.rglob("*") if p.is_file()}
        (out / "development-input-hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        report["input_hashes"] = "development-input-hashes.json"
        report["input_hashes_sha256"] = digest(out / report["input_hashes"])
        report["llama_head"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        build = args.ggml_build.resolve() if args.ggml_build else out / "ggml-host"
        if not args.ggml_build:
            run("ggml-configure", ["cmake", "-S", inputs, "-B", build, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + args.cc,
                "-DCMAKE_CXX_COMPILER=" + args.cxx, "-DBUILD_SHARED_LIBS=OFF", "-DGGML_CCACHE=OFF",
                "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_CPU_REPACK=OFF",
                "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF", "-DGGML_RISCV_AKV=OFF",
                "-DGGML_RISCV_QBS=OFF", "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF",
                "-DGGML_FMA=OFF", "-DGGML_F16C=OFF"])
            run("ggml-build", ["cmake", "--build", build, "--target", "ggml-cpu", "ggml-base", "-j", args.jobs])
        libraries = []
        for name in ("libggml-cpu.a", "libggml-base.a"):
            matches = list(build.rglob(name))
            if len(matches) != 1:
                raise RuntimeError(f"need one host {name} in {build}; found {matches}")
            libraries.append(matches[0])
        report["host_libraries"] = {str(p): digest(p) for p in libraries}
        arch = source / "src/ggml-cpu/arch/riscv"
        includes = ["-I" + str(p) for p in (source / "include", source / "src", source / "src/ggml-cpu",
                                           arch, frozen_sdk / "include", frozen_sdk / "experimental")]
        flags = ["-O2", "-g", "-UNDEBUG", "-fno-fast-math", "-ffp-contract=off", "-ffunction-sections", "-fdata-sections"]
        objects = []
        for path in (arch / "entry.c", arch / "platform.c", frozen_sdk / "src/akv_runtime.c",
                     frozen_sdk / "src/akv_features.c", frozen_sdk / "src/akv_prefill_reference.c"):
            obj = out / (path.stem + ".o")
            run(path.stem, [args.cc, "-std=c11", *flags, *includes, "-c", path, "-o", obj])
            objects.append(obj)
        definitions = ["-DGGML_AKV_RESIDENT=1", "-DGGML_USE_RISCV_AKV=1", "-DGGML_RISCV_ACCEL_AUTO=1",
                       "-DGGML_USE_RISCV_AKV_LINUX_DIRECT=1", "-DGGML_USE_RISCV_QUANT_KV=1",
                       "-DAKV_NATIVE_AUTO_PAIR_POLICY=1"]
        for path in (arch / "akv.cpp", test):
            obj = out / (path.stem + ".o")
            run(path.stem, [args.cxx, "-std=c++17", *flags, *definitions, *includes,
                           "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", "-c", path, "-o", obj])
            objects.append(obj)
        binary = out / "test-ara-linux-direct-routing"
        wraps = ["-Wl,--wrap=" + name for name in ("malloc", "calloc", "realloc", "aligned_alloc",
                                                   "posix_memalign", "free", "fprintf", "ggml_abort")]
        run("link", [args.cxx, *objects, "-Wl,--start-group", *libraries, "-Wl,--end-group",
                     *wraps, "-Wl,--gc-sections", "-pthread", "-ldl", "-lm", "-o", binary])
        report["binary"] = {str(binary): digest(binary)}
        if args.run:
            for name in args.case or CASES:
                run("case-" + name, [binary, name], timeout=60)
                output = (out / ("case-" + name + ".log")).read_text()
                if f"DIRECT_ROUTING_PASS case={name} native_execution=0 board_run=0" not in output:
                    raise RuntimeError("missing pass marker: " + name)
                summaries = [json.loads(line) for line in output.splitlines() if line.startswith('{"case":')]
                if len(summaries) != 1 or summaries[0]["case"] != name or summaries[0]["attention_calls"] != CASES[name]:
                    raise RuntimeError("wrong Attention invocation count: " + name)
                report["case_results"].append(summaries[0])
                report["cases"].append(name)
            report["case_count"] = len(report["cases"])
            report["attention_calls"] = sum(case["attention_calls"] for case in report["case_results"])
            report["status"] = "DEVELOPMENT_PASS"
        else:
            report["status"] = "COMPILED_NOT_RUN"
        for relative, expected in hashes.items():
            if digest(inputs / relative) != expected:
                raise RuntimeError("build-input drift: " + relative)
        if any(digest(Path(path)) != expected for path, expected in report["host_libraries"].items()):
            raise RuntimeError("host library drift")
        save()
        print(report["status"] + ": " + str(out))
        print("Non-sealing host routing evidence only; native math and hardware are not exercised.")
    except BaseException as error:
        report.update(status="FAILED", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
