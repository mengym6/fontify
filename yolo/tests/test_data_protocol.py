"""Prevent new validation splits from inheriting old Calli training exposure."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("calli_train_protocol", Path(__file__).resolve().parents[1] / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class DatasetProtocolTests(unittest.TestCase):
    def test_read_dataset_preserves_split_and_resolves_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\nnames: [phase]\n"
                "annotation_availability: flags.json\nsplit_id: fixed-split\n"
            )
            result = train.read_dataset(root / "data.yaml")
            self.assertEqual(result["split_id"], "fixed-split")
            self.assertEqual(result["annotation_availability"], str((root / "flags.json").resolve()))

    def test_generic_initialization_allowed_old_calli_rejected_and_same_split_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "run/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"placeholder; no model loading")
            flags = root / "flags.json"
            flags.write_text("{}")
            data = {"names": {0: "phase", 1: "component"}, "split_id": "new-split",
                    "annotation_availability": str(flags)}
            train.validate_initialization_split({0: "person", 1: "car"}, weights, data)
            with self.assertRaisesRegex(ValueError, "COCO"):
                train.validate_initialization_split(data["names"], weights, data)
            sidecar = weights.parent.parent / "partial_supervision.json"
            protocol = {"schema": "calli-partial-supervision-v1", "split_id": "old-split",
                        "availability_sha256": hashlib.sha256(flags.read_bytes()).hexdigest()}
            sidecar.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "protocols differ"):
                train.validate_initialization_split(data["names"], weights, data)
            protocol["split_id"] = "new-split"
            sidecar.write_text(json.dumps(protocol))
            train.validate_initialization_split(data["names"], weights, data)

    def test_manual_stopping_and_every_epoch_sixty_defaults(self):
        args = train.parser().parse_args([])
        self.assertEqual(args.patience, 0)
        self.assertEqual(args.preview_every, 1)
        self.assertEqual(args.preview_count, 60)
        self.assertEqual(args.data.parent.name, "dataset_full")


if __name__ == "__main__":
    unittest.main()
