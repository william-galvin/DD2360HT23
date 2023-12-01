# Histogram Optimizations

All bench marks used with input length 262144 (512^2) after several "warm up" runs, averaged over 10 runs

To run a benchmark:
1. cd into ex1/{optimization}
2. make
3. run `$ ../run.py` (10 runs with warmup) or `$ ./ex1 <n>` 

Ideas to try:
- memory optimizations (shared, pinned, unified, etc)
- Combine kernels (one kernel call instead of 2)


## Initial (Naive) Implementation {/base}:
Results:
```
copy H => D: 626.5
kernel1: 24.9
kernel2: 7.7
copy D => H: 130.8
```

These are the times to beat for optimizations to be considered useful.

---

## Attempt 1: Packing {/pack}
Idea: copying from host to device is expensive and happens twice. Can we combine those
into one "packed" array?

Implementation:
- Replace {host | device}{input | bins} with {host | device}Pack of size `inputLength + NUM_BINS`
- The first `inputLength` items are the input, the last `NUM_BINS` are bins
- Leave the kernels the same, referencing `input` as `devicePack` and `bins` as `devicePack[inputLength]`


Results:
```
copy H => D: 405.1
kernel1: 31.1
kernel2: 8.7
copy D => H: 308.6
```
These results appear to show a modest speed up in host -> device copy, but a slowdown everywhere else.

---

## Attempt #2 Streams {/stream}
Idea: Overlap data transfer and computation time using streams

Implementation: Since we are using a modern version of cuda, we have access to Hyper-Q and can choose approach 1 or 2 arbitrarily. For ease of timing, we'll choose approach 2, as described in Lecture: Optimizing Host-Device Data Communication III - Code Examples.

Note: Fore ease of timeing, one combined kernel time is reported, even though the kernels are run separately.

Also note: we need to add synchronization between calls to device, for correctness.

Results:
```
warmups complete
copy H => D: 809.0
kernel (combined): 210.5
copy D => H: 957.2
```

These results appear to show that using streams is not helpful in this context.

---

## Attempt #3: Smaller Datatypes {/small}
Idea: Since the maximum value of a bin is 127 and the maximum random int is 4095 we can use 8-bit ints for the bins and 16-but ints for the random numbers. These are smaller than the default `uint`, and should be faster to copy back and forth. Also, the GPU may have hardware optimizations to make the kernels faster.

Implementation: We use `uint16_t` for both bins and input ints, even though bins can technically be `uint8_t`. This makes clamping values easier, although if we combine both kernels, perhaps we can use `uint8_t` for bins more easily. Also note that we had to find an atomicAdd operation not natively supported by cuda.

Results:
```
copy H => D: 478.2
kernel1: 20.6
kernel2: 5.5
copy D => H: 125.9
```

These results show that using smaller datatypes doesn't dramatically improve speed.

---

## Attempt #4: Combine kernels {/combined}
Idea: launching kernels is expensive---can we combine the two kernerls into one?

Results:
```
copy H => D: 693.4
kernel (combined): 26.3
copy D => H: 188.5
```

If there is a difference, our timing code doesn't pick up on it.

---

## Attempt #5: Use shared memory {/shared}
Idea: We can avoid waiting for locks on atomic add if we have a separate shared histogram for each block and combine them at the end.

Results:
```
copy H => D: 653.9
kernel1: 34.0
kernel2: 11.6
copy D => H: 4705.6
```
Why is the copy-back so slow?

---

## Attempt #6: Use pinned memory {/pinned}
Idea: pinned memory should be faster for transfers, since we don't need to make as many copies

```
copy H => D: 1180.9
kernel1: 29.7
kernel2: 9.7
copy D => H: 238.3
```

This makes copies more expensive, for some reason. 

---

## Attempt #6: Combining all the promising optimizations {/optimized}
We'll take ideas from 
- packing
- small datatypes
- combined kernels

Results
```
copy H => D: 497.8
kernel (combined): 43.8
copy D => H: 307.3
```