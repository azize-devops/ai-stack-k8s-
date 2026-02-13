# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please send an email to the project maintainers with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Any suggested fixes (if available)

We will acknowledge receipt within 48 hours and aim to provide a fix within 7 days for critical issues.

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Security Practices

This project follows these security practices:

- **Pod Security Standards**: Kubernetes namespace enforces `baseline` (enforce) and `restricted` (audit/warn)
- **SecurityContext**: All containers run as non-root with dropped capabilities and read-only root filesystems
- **Network Policies**: Default-deny with explicit allow rules per service
- **Image Pinning**: All container images use specific version tags (no `:latest`)
- **Dependency Pinning**: Python dependencies pinned to exact versions (`==`)
- **Secret Management**: No hardcoded credentials; secrets via Kubernetes Secrets
- **TLS**: Ingress configured with TLS termination and SSL redirect
- **API Authentication**: RAG Pipeline API protected with API key middleware
- **CORS**: Locked down by default, configurable via environment variable
- **CI/CD Security**: Automated Trivy scans, pip-audit, Gitleaks secret detection, Kubescape K8s scanning
- **Pre-commit Hooks**: Ruff linting, Gitleaks, yamllint, private key detection

## Dependencies

We regularly audit dependencies using:
- `pip-audit` for Python package vulnerabilities
- Trivy for container image scanning
- Kubescape for Kubernetes security posture

## Disclosure Policy

We follow a coordinated disclosure process:
1. Reporter submits vulnerability details privately
2. We confirm the vulnerability and assess severity
3. We develop and test a fix
4. We release the fix and publicly disclose the vulnerability
