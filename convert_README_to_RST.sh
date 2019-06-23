#!/usr/bin/env bash
echo "Manually fix indentations in numbered list in rst!!!"
pandoc --from=markdown --to=rst --output=README.rst README.md
