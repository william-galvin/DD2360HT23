#!/usr/bin/python3

headers = 10
EPS = 1e-6

if __name__ == "__main__":
    
    with open("data/rho_net_10.vtk") as f:
        gpu = f.readlines()[headers:]
    with open("data-cpu/rho_net_10.vtk") as f:
        cpu = f.readlines()[headers:]
    
    if len(gpu) != len(cpu):
        raise ValueError("Length of cpu and gpu input files differ")

    incorrect = 0
    for g, c in zip(gpu, cpu):
        if abs(float(g) - float(c)) > EPS:
            incorrect += 1
            print(f"Expected {float(c)} but got {float(g)}")

    print(f"Total correct: {100 - (100 * incorrect / len(cpu))}% of {len(cpu)} items")