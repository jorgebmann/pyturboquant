# Security Policy

## Supported Versions

pyturboquant is pre–1.0 and evolving quickly. Security fixes are applied to the **latest release in the current minor line** (today **0.1.x**). Older lines are not guaranteed backports unless noted below.

| Version   | Supported          |
| --------- | ------------------ |
| 0.1.x     | :white_check_mark: |
| pre-0.1   | :x:                |

When **0.2.0** ships, this table will be updated (for example, 0.2.x supported, 0.1.x best-effort or unsupported). Check the latest tag on [Releases](https://github.com/jorgbahlmann/pyturboquant/releases) before reporting against an old install.

**Out of scope for this policy**

- Vulnerabilities in **PyTorch**, **Python**, CUDA drivers, or optional dependencies (e.g. LangChain, BEIR, sentence-transformers): please report those to the respective upstream projects.
- Issues that only affect unreleased git snapshots: we still welcome reports, but please mention the commit hash.

## Reporting a Vulnerability

**Please do not open a public GitHub issue** for undisclosed security problems. That can put users at risk before a fix exists.

### Preferred: GitHub private reporting

If [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability) is enabled for this repository, use it:

1. Open the repo on GitHub: [jorgbahlmann/pyturboquant](https://github.com/jorgbahlmann/pyturboquant).
2. Go to **Security** → **Report a vulnerability** (or **Security advisories**), and follow the form.

Include:

- A short description and impact (confidentiality / integrity / availability).
- Steps to reproduce, or a minimal proof of concept, if possible.
- Affected version(s) or commit, and your environment (OS, Python, PyTorch) if relevant.

### If private reporting is unavailable

Email **jorg.bahlmann@gmail.com** with subject line starting with `[SECURITY] pyturboquant`. Encrypt sensitive details with PGP only if you already use the maintainer’s published key; otherwise GitHub’s private advisory flow is preferred.

### What to expect

- **Acknowledgment**: We aim to confirm receipt within **72 hours** (often sooner). If you do not hear back, a polite bump is welcome.
- **Updates**: We will keep you informed of triage progress and whether we accept the report for a fix.
- **Disclosure**: We prefer coordinated disclosure. Please give us a reasonable window to release a fix before public details. We will not take legal action against good-faith security research that follows this policy.
- **Resolution**: If the report is valid, we will work on a patch, prepare a release or advisory as appropriate, and credit you in the advisory or release notes if you wish. If we decline (e.g. not applicable, out of scope), we will explain briefly.

Thank you for helping keep pyturboquant users safe.
