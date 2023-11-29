All bench marks used with input length 262144 (512^2) after several "warm up" runs

Ideas to try:
- Shared memory (several kinds to try)
- Smaller datatypes (use problem statement to pick smallest possible)
- Combine kernels


## Initial (Naive) Implementation {/base}:
Results:
```
Copy host => device: 581
kernel 1: 18
kernel 2: 11
Copy device => host: 62
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
Copy host => device: 390
kernel 1: 22
kernel 2: 5
Copy device => host: 257
```
These results appear to show a modest speed up in host -> device copy, but a slowdown everywhere else.

---

## Attempt #2 Streams {/stream}
Idea: Overlap data transfer and computation time using streams

Implementation: Since we are using a modern version of cuda, we have access to Hyper-Q and can choose approach 1 or 2 arbitrarily. For ease of timing, we'll choose approach 2, as described in Lecture: Optimizing Host-Device Data Communication III - Code Examples.

Also note: we need to add synchronization between calls to device, for correctness

Results:
```
Copy host => device: 641
both kernels: 174
Copy device => host: 909
```

These results appear to show that using streams is not helpful in this context.

---