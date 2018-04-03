#!/bin/sh

cp $1 $1.bak
jq '(.clips[] | .points_3d) |= []' $1.bak > $1
