#!/usr/bin/env bash
# echo "*** WARNING ********"
# echo " do not push to pypi until you fix the indents!!!!!!"
# echo "Manually fix indentations in numbered list in rst!!!"
# echo "*** WARNING ********"

pandoc --from=markdown --to=rst --output=README.rst README.md
