# Vendored from https://github.com/NVlabs/verilog-eval (release/1.0.0)
# Copyright (c) 2023 NVIDIA Research Projects — MIT License (see LICENSE in this directory)
#
# The upstream package cannot be installed via uv/pip due to a malformed
# entry_points in setup.py, so we vendor the three needed modules.
#
# Changes from upstream:
#   execution.py — removed triple-quote delimiters that intentionally
#       disabled the iverilog execution code (upstream safety measure)
#   execution.py — fixed unescaped regex: "repeat\(" -> r"repeat\("
