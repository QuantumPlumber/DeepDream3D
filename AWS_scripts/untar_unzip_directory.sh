#!/bin/bash

find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
find . -name '*.zip' -execdir unzip '{}' \;