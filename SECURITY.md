# Security Policy

## Supported versions

CellQuorum is pre-1.0 and under active development. Only the latest release on the
`main` branch receives fixes.

| Version | Supported |
|---|---|
| Latest release | ✅ |
| Older releases | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/SecondBook5/cellquorum/security/advisories/new)
on this repository. Include the version, what an attacker could achieve, and a
minimal reproduction if you have one.

Expect an acknowledgement within a week. Because this is a research tool maintained
by a small team, please allow reasonable time for a fix before disclosing publicly.

## Threat model, and what counts as a vulnerability

CellQuorum is a **scientific analysis engine** that a user runs on their own machine
or cluster, on their own data, from their own configuration file. It is not a network
service and has no authentication, multi-tenancy, or privilege boundary. That shapes
what is and is not a security issue.

**In scope:**

- Arbitrary code execution triggered by *parsing* an untrusted config, manifest, or
  `.h5ad` file — i.e. before the user has approved anything.
- Path traversal that writes outside the configured run directory.
- Unsafe deserialization (`pickle`, `yaml.load` without a safe loader, `eval`) reached
  from data or config that a user might plausibly receive from someone else.
- Leaking credentials or tokens into run outputs, logs, or provenance files.
- A dependency vulnerability that CellQuorum actually exposes to untrusted input.

**Out of scope:**

- Config-driven execution of R scripts and external tools. Dispatching to R,
  pySCENIC, CellOracle, and similar backends by subprocess is the engine's documented
  purpose. A config *is* a program; treat one from an untrusted source the way you
  would treat a shell script from an untrusted source.
- Resource exhaustion from a large dataset or an expensive method. Single-cell
  analysis is legitimately memory- and compute-hungry.
- Vulnerabilities in an optional heavyweight backend that we only invoke. Report
  those upstream; tell us too if CellQuorum's usage makes them reachable in a way
  the upstream project would not expect.

## Note on reproducibility, not security

Runtime dependencies are declared as lower bounds so CellQuorum can co-exist with
other single-cell packages. That means a fresh install resolves to current versions,
which is good for security patches and bad for byte-identical reproducibility. If you
need to reproduce a specific run, use the pinned conda lock files
(`envs/*.conda-lock.yml`) or the Docker image, and keep the `provenance/` directory —
it records the resolved environment for the run.
