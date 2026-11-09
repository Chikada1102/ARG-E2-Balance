"""Read-only checks of the existing VM-UNet Synapse case lists."""
import hashlib
import re
from pathlib import Path


def synapse_split_manifest(list_dir):
    root = Path(list_dir)
    result = {}
    for split in ("train", "test_vol"):
        path = root / (split + ".txt")
        if not path.is_file():
            raise FileNotFoundError("Missing existing VM-UNet list: " + str(path))
        content = path.read_bytes()
        lines = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
        if not lines or len(lines) != len(set(lines)):
            raise ValueError("Empty or duplicated entries in " + str(path))
        cases = []
        for name in lines:
            match = re.match(r"^(case\d+)(?:_slice\d+)?(?:\.npz|\.npy\.h5|\.h5)?$", name)
            if not match:
                raise ValueError("Unexpected Synapse list entry: " + name)
            cases.append(match.group(1))
        result[split] = dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest(),
                             entries=len(lines), cases=sorted(set(cases)))
    training, testing = set(result["train"]["cases"]), set(result["test_vol"]["cases"])
    if len(training) != 18 or len(testing) != 12:
        raise ValueError("Keep VM-UNet's 18/12 case split; found %d/%d" % (len(training), len(testing)))
    if training & testing:
        raise ValueError("A patient occurs in both train and test: " + repr(sorted(training & testing)))
    if result["test_vol"]["entries"] != 12:
        raise ValueError("test_vol.txt must list the 12 test volumes, not slices")
    # These checks do not establish that names match an external reference.
    # Reuse the supplied official data lists; compare hashes across experiments.
    return result
