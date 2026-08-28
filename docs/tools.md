# External tools

## SPMF (Sequential Pattern Mining Framework)

- **Version:** 2.64 (NOT the current 2.66; see note below).
- **Source:** https://www.philippe-fournier-viger.com/spmf/spmf2.64.jar
- **Local path:** `scripts/spmf.jar`
- **SHA256:** `c67c2e56bdd2072eadc0e4bc96c8221cec38c9109f99c4b5f10be8eba64ec250`
- **Size:** 15.5 MB
- **License:** GPL v3 (per SPMF project page).

### Why 2.64 and not 2.66

SPMF 2.66 (current release as of 2026-06-15) is compiled to class file
version 69, which requires Java 25+. The local JDK is Temurin 21.0.11 (class
file version 65), which loads only up to class file 65. Two options:

1. Pin SPMF at 2.64, which compiles to a class version Java 21 loads
   cleanly.
2. Install Java 25.

Went with (1) because PrefixSpan and the other classic SPM algorithms
(SPADE, GSP, CM-SPAM, VMSP) are stable in the 2.60-line and this study
does not need the newer algorithms added in 2.65/2.66. Revisit if a
specific 2.66-only algorithm is needed.

### Invocation

```
java -jar scripts/spmf.jar run <ALGO> <INPUT> <OUTPUT> <params...>
```

Input format for sequence algorithms: each line is one sequence; items are
separated by ` -1`; sequences end with ` -2`. See the SPMF documentation
for each algorithm's exact parameter contract.

Smoke test (used at setup) produced 17 frequent sequences from 4 toy
sequences at 50% support, matching the expected count.
