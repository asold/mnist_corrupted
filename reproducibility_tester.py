import sys
import torch

if __name__ == "__main__":
    exp1 = sys.argv[1]
    exp2 = sys.argv[2]

    print(f"Comparing run {exp1} to {exp2}")

    sd1 = torch.load(f"{exp1}/models/model.pth")
    sd2 = torch.load(f"{exp2}/models/model.pth")

    if sd1.keys() != sd2.keys():
        raise RuntimeError("State dicts have different parameter sets")

    for k in sd1:
        if not torch.allclose(sd1[k], sd2[k]):
            raise RuntimeError(
                f"Parameter '{k}' differs — script is not reproducible"
            )

    print("✅ Models are identical — experiment is reproducible")
