#!/usr/bin/env python3
import os, re, json, sys, argparse

def looks_failed(text:str)->bool:
    return bool(re.search(r"(Traceback \(most recent call last\)|\bERROR\b|AssertionError|Segmentation fault|^error:)", text, re.I|re.M))

def failure_message(text:str)->str:
    blocks = list(re.finditer(r"Traceback \(most recent call last\):([\s\S]*?)(?:\n\s*\n|\Z)", text, re.S))
    if blocks:
        lines = blocks[-1].group(1).strip().splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        if re.search(r"(error|exception|failed|segmentation fault|assert)", line, re.I):
            return line
    return "failed (see log)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="logs/<timestamp> folder")
    ap.add_argument("--fail-on-failures", action="store_true")
    args = ap.parse_args()

    root = args.root
    precs = ["fp32","fp16"]
    summary = {"totals":{"pass":0,"fail":0},"regressions":{}}

    for prec in precs:
        d = os.path.join(root, prec)
        reg = {"passed":[], "failed":[]}
        if os.path.isdir(d):
            for fn in sorted(f for f in os.listdir(d) if fn.endswith(".log")):
                model = fn[:-4]
                with open(os.path.join(d, fn), "r", errors="replace") as f:
                    txt = f.read()
                if looks_failed(txt):
                    reg["failed"].append({"model": model, "message": failure_message(txt)})
                    summary["totals"]["fail"] += 1
                else:
                    reg["passed"].append(model)
                    summary["totals"]["pass"] += 1
        summary["regressions"][prec] = reg

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root,"summary.json"),"w") as f:
        json.dump(summary,f,indent=2)

    lines=[]
    lines.append("## Totals")
    lines.append(f"- PASS: {summary['totals']['pass']}")
    lines.append(f"- FAIL: {summary['totals']['fail']}\n")
    for prec in precs:
        reg = summary["regressions"][prec]
        lines.append(f"## {prec.upper()}")
        lines.append(f"**Passed ({len(reg['passed'])})**")
        lines += [f"- {m}" for m in reg["passed"]] or ["- none"]
        lines.append("")
        lines.append(f"**Failed ({len(reg['failed'])})**")
        lines += [f"- {it['model']}: `{it['message']}`" for it in reg["failed"]] or ["- none"]
        lines.append("")
    md = "\n".join(lines).strip() + "\n"
    with open(os.path.join(root,"summary.md"),"w") as f:
        f.write(md)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary,"a") as f:
            f.write(md)

    if args.fail_on_failures and summary["totals"]["fail"]>0:
        sys.exit(1)

if __name__ == "__main__":
    main()
