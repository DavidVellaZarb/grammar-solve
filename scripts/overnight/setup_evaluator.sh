#!/usr/bin/env bash
set -euo pipefail

DEST="third_party/overnight"

if [ -x "$DEST/evaluator/overnight" ]; then
    echo "Overnight evaluator already installed at $DEST/evaluator/overnight"
    exit 0
fi

if ! command -v java &> /dev/null; then
    echo "ERROR: Java is required but not found. Install Java 8+ (e.g. brew install temurin)." >&2
    exit 1
fi

echo "Setting up Overnight SEMPRE evaluator..."
mkdir -p "$DEST"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/berlino/grammar-prompting.git "$TMPDIR/gp"
cd "$TMPDIR/gp"
git sparse-checkout set third_party/overnight/evaluator third_party/overnight/module-classes.txt
cd - > /dev/null

cp -r "$TMPDIR/gp/third_party/overnight/evaluator" "$DEST/"
cp "$TMPDIR/gp/third_party/overnight/module-classes.txt" "$DEST/"

chmod +x "$DEST/evaluator/overnight"

echo "Overnight evaluator installed at $DEST/evaluator/overnight"
java -version 2>&1 | head -1
echo "Done."
