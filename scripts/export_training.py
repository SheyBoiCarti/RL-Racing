"""Export, verify or import a complete checkpoint lineage."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing.training.export import (
    export_runs,
    import_export,
    select_resume_checkpoint,
    verify_export,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "import", "verify", "checkpoint"])
    parser.add_argument("source")
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.command == "checkpoint":
        print(select_resume_checkpoint(args.source))
        return
    if args.command == "verify":
        result = verify_export(args.source)
    else:
        if not args.destination:
            parser.error("--destination required")
        result = (export_runs if args.command == "export" else import_export)(args.source, args.destination)
    print(f"Verified {len(result['files'])} files")


if __name__ == "__main__":
    main()
