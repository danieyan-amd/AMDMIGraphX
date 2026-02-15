import os, pathlib, datetime, subprocess

print("Hello from", os.uname().nodename)
subprocess.run(["/opt/rocm/bin/rocm-smi"], check=False)

pathlib.Path("results").mkdir(exist_ok=True)
with open("results/hello.txt", "w") as f:
    f.write(f"ran: {datetime.datetime.utcnow().isoformat()}Z\n")

print("Wrote results/hello.txt")
