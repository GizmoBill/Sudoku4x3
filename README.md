This repository holds the source code and data files for my 2022 verification that there are
exactly 81,171,437,193,104,932,746,936,103,027,318,645,818,654,720,000 legal configurations
of a 12x12 Sudoku grid with 4x3 boxes. It further holds my ongoing research notes and source code on
programming an Nvidia Jetson AGX Xavier GPU to do the count.

The original exact count was done in 2006 by Kjell Fredrik Pettersen of Norway and Bill Silver
(this author) of the USA. Pettersen created the enumeration methods and wrote a complete,
functional implementation. I examined the primary loops and made significant impovements in speed,
primarily reordering the memory access pattern for better data cache performance. Together
we ran the code on up to six computers in Norway and Massachusetts. The result was obtained in
2568 hours of CPU time, and has never been independently verified until now.

The current code is a completely original, independent design and implementation. I did not
and do not have more than a superficial understanding of Pettersen's methods and implementation,
and I did not look at any 2006 source code in 2022. Furthermore, examination of the 2006
intermediate data and count files, and comparison with the 2022 alternatives, shows very
significant differences in the methods. These differences are described in the source code
comments, and the comparisons can be run by the code itself.

The present enumeration methods are described in source code comments (imperfectly no doubt);
also described are C++ template programming techniques designed to improve programming reliability.
The code is not intended to meet professional style and documentation standards.

Pettersen's method is more sophisticated and efficient than mine. I had the enormous advantage
of using 64-bit multicore CPUs, Pettersen had to squeeze into 32-bit single-core
machines. My total running time is about 40% of Pettersen (1035 hours), all improvement due to
using every parallel thread of 2- and 4-core hyperthreaded 64-bit machines, and about 1.4 GB of
lookup tables. This does make the present version simpler, which may add to its reliability.

While my original verification run consumed 1035 CPU-hours, the current version in the
repository needs only 44.7 hours on my Jetson. This enormous speedup is almost entirely due
to a reorganization of the counting loops that dramatically improves data cache hit rates,
not due to the use of the GPU. The computation is memory bound with a memory
access pattern that the Jetson's 8 ARM cores handle better than the 8 GPU streaming multiprocessors.
The research log has details.

The code compiles and runs with MS Visual Studio C++ under Windows (x64 machines), and Eclipse/GCC
under Ubuntu (x64 and ARM machines). Most of the counts were done on Windows machines, a few on
Ubuntu. C++17 is required for filesystem access. It uses a commandline interface.

(Note: I wrote the Bignum package in about 1995 as my first C++ program and learning exercise,
with a few modifications over the years. The style looks pretty awkward to me now, but it works,
has lots of capabilities, and is pretty efficient.)

The following copyright notice is embeded in the source code:

COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
FROM PORTIONS OR THE ENTIRETY OF THE CODE.
