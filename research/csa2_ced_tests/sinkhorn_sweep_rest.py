"""Rest of the E-42 Sinkhorn rate sweep (arms 0.1 and 0.2), waiting for free GPU.
Polls torch.cuda.mem_get_info before each arm; never starts while <4.0 GiB
free; aborts cleanly (saves partial JSON) if a wait times out.
"""
import json, sys, time
import torch
sys.path.insert(0, r"E:/python_projects/nano_SLMs/research/csa2_ced_tests")
import bench_extra

WAIT_CAP_S = 3600     # give up after 1h total waiting
POLL_S = 30
MIN_FREE_GB = 4.0


def wait_for_gpu(cap_remain=WAIT_CAP_S):
    t0 = time.time()
    while True:
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= MIN_FREE_GB:
            return True
        if time.time() - t0 > cap_remain:
            return False
        print(f"[wait] free={free / 2**30:.2f} GiB < {MIN_FREE_GB}; sleeping {POLL_S}s", flush=True)
        time.sleep(POLL_S)


def main():
    out = []
    wait_used = 0.0
    for name, rate in [("sinkhorn_0.1", 0.1), ("sinkhorn_0.2", 0.2)]:
        if not wait_for_gpu(WAIT_CAP_S - wait_used):
            print("[abort] GPU stayed busy for the whole wait cap", flush=True)
            break
        tw = time.time()
        try:
            r = bench_extra.run(name, rate)
            out.append(r)
        except Exception as exc:
            print(f"[fail] {name}: {exc}", flush=True)
        wait_used += time.time() - tw if False else 0.0
        with open(r"E:/python_projects/nano_SLMs/research/csa2_ced_tests/sinkhorn_sweep.json", "w") as f:
            json.dump(out, f, indent=2)
    print("DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
