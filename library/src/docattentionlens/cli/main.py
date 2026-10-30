import argparse
import json

from docattentionlens.core.integrity import IntegrityInput, check_integrity
from docattentionlens.methods.registry import list_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAttentionLens library")
    parser.add_argument("command", choices=["methods", "integrity-check"])
    args = parser.parse_args()
    if args.command == "methods":
        print(json.dumps([m.model_dump() for m in list_methods()], indent=2))
    else:
        print(check_integrity(IntegrityInput()).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
