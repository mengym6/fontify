"""Run the established real-model DDP smoke with pixel-union training masks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import dual_view_protocol, smoke_dual_view


def main():
    original = dual_view_protocol.build_dataset

    def pixel_dataset(args):
        dataset = original(args)
        dataset.pixel_masking = True
        return dataset

    dual_view_protocol.build_dataset = pixel_dataset
    smoke_dual_view.main()


if __name__ == '__main__':
    main()
