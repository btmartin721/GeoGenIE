#!/bin/zsh

pandoc --listings -H docs/listings-setup.tex  docs/user_manual.md --metadata-file=docs/HEADER.yaml --template=docs/template.tex --citeproc -o GeoGenIE_UserManual.pdf --pdf-engine=xelatex
