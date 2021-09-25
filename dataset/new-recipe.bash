#!/bin/bash

TEMPLATE='{\n\t"name": "",\n\t"url": "",\n\n\t"ingredients": [\n\t\t\n\t],\n\n\t"steps": [\n\t\t\n\t]\n}'

FILES=$(ls dataset/raw/*.json)
NUMS=$(echo $FILES | sed -r 's/\S*recipe([0-9]+)\.json/\1/gm')
MAX=$(echo $NUMS | tr " " "\n" | sort -gr | head -n1)
MAX_PLUS_ONE=$(($MAX + 1))

echo -e $TEMPLATE >> "dataset/raw/recipe${MAX_PLUS_ONE}.json"