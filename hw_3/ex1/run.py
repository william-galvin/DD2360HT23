#!/bin/python3

"""
Reports the average timing over 10 runs (+ 5 warmups) for ex1 histogram program.
Does not check correctness.
Supports timing for kernels together or separately
Expects input from stdin in the form
    Copy host => device: 654
    both kernels: 170
    Copy device => host: 739

Usage:
    $ cd ex1/base
    $ ../run.py
"""

import subprocess

if __name__ == "__main__":
    copy_HTD = []
    kernel1 = []
    kernel2 = []
    copy_DTH = []

    for i in range(5):
        out = (subprocess.check_output(f"./ex1 {512 ** 2}".split(), stderr=subprocess.PIPE)).decode("utf-8").strip().split("\n")
    print("warmups complete")


    for i in range(10):
        out = (subprocess.check_output(f"./ex1 {512 ** 2}".split(), stderr=subprocess.PIPE)).decode("utf-8").strip().split("\n")
        for i, line in enumerate(out):
            time = int(line[line.find(":") + 2:])
            
            if i == 0:
                copy_HTD.append(time)
    
            if len(out) == 4:
                if i == 1:
                    kernel1.append(time)
                if i == 2:
                    kernel2.append(time)
                if i == 3:
                    copy_DTH.append(time)

            else:
                if i == 1:
                    kernel1.append(time)
                if i == 2:
                    copy_DTH.append(time)

    print("copy H => D:", sum(copy_HTD)/len(copy_HTD))
    if len(kernel2) > 0:
        print("kernel1:", sum(kernel1)/len(kernel1))
        print("kernel2:", sum(kernel2)/len(kernel2))
    else:
        print("kernel (combined):", sum(kernel1)/len(kernel1))
    print("copy D => H:", sum(copy_DTH)/len(copy_DTH))