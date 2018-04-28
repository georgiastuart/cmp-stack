Seismic streamer data (pressure component) recorded in one acquisition line.

The first CMP bin (i.e. CMP #1) is defined at 1618.5 m, and the CMP bin size is 12.5 m.

The data are trimmed to make up 1900 CMP gathers, from CMP #125 to #2024, and each gather has 60 traces.



Acquisition maps after trim:

  acqui_map.png:   acquisition map in shot-receiver coordinate.

  acqui_map2.png:  acquisition map in CMP-offset coordinate.

Each red dot represents a trace at that shot&receiver or CMP&offset location.



CMP gathers:

  1900cmp_gather.bin: all 1900 CMP gathers CMP #125 to #2024.

  190cmp_gather.bin:  decimated every 10th CMP gathers.

  19cmp_gather.bin:   decimated every 100th CMP gathers.

  1cmp_gather.bin:    1 CMP gather CMP #1746.

Each CMP gather has 60 traces (i.e. fold = 60) with offset = 262 m to 3212 m.

All traces have 6s time window, 0.004 sampling.

Note that some traces are dead traces (i.e. no signal). They are not depicted on the acquisition maps but are included in the data.



Other information:

  CMPgather_1746.png: screenshot of CMP gather CMP #1746 (i.e. 1cmp_gather.bin).

  nearoffset.png:     common-near offset gather (offset = 262 m), gained by time^1.5 (in seconds).

  shot_26487.png:     one shot gather (shot position = 26487 m).
