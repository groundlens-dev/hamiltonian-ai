# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest) | Yes |
| < 0.1.0 | No |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **javier@jmarin.info** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential impact assessment

### What to expect

- **Acknowledgment:** within 48 hours.
- **Initial assessment:** within 5 business days.
- **Disclosure:** coordinated; we follow responsible disclosure and credit
  reporters unless they prefer anonymity.

## Scope

In scope: code-execution, deserialization, and path-traversal issues in the
packaged `hamiltonian_ai` library and its direct dependencies. Out of scope:
the research notebooks under `studies/` and `applications/`, which are
illustrative artifacts run by the user against their own data.
