"""Fresh-process cold-start probe, independent of scored test observations."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from signal_rescue.channels import ChannelBank
from signal_rescue.hybrid import detect
from signal_rescue.samplers import Config


def main():
    frozen = json.loads(Path(sys.argv[1]).read_text())
    instance = ChannelBank(seed=7).instance("validation", "rayleigh", "square", 12, 999)
    started = time.perf_counter()
    result = detect(instance.H, instance.y, instance.sigma2, config=Config(**frozen["config"]),
                    params=np.asarray(frozen["params"]), seed=7)
    print(json.dumps({"inference_s": time.perf_counter() - started,
                      "shots": result.telemetry["shots"], "circuit_invocations": result.telemetry["circuit_invocations"],
                      "scope": "fresh process, controlled Rayleigh 8x8 validation channel; not a scored test point"}))


if __name__ == "__main__":
    main()
