#!/bin/bash

if [ "$(basename $(pwd))" != 'docsrc' ]  ; then
  echo "This must be executed from with the docsrc directory"
  exit 1
fi

if [ -d ../venv ] ; then
  source ../venv/bin/activate
  # github target is custom target added to handle populating a docs directory
  # intended for the github io docs page.
  make github
fi
