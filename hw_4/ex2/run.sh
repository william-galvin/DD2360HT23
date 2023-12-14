#!/bin/bash

for i in {8..16}; do
    ./ex2 $((2**24)) $((2**i))
done