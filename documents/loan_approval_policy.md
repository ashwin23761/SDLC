# Retail Loan Approval Policy (Sample, for prototype use only)

## s.1 Eligibility

**1.1 Minimum criteria** — An applicant must be 18 years or older, have a verified income
source, and have no active default flags in the internal credit system.

## s.2 Credit Assessment

**2.1 Credit bureau report** — A credit bureau report must be pulled for every application.
Reports older than 30 days may not be reused; a fresh report must be pulled.

**2.2 Credit score thresholds** —
- Score 750 and above: eligible for automated approval, subject to eligibility and sanctions
  checks passing.
- Score 600–749: routed to a human underwriter for manual assessment.
- Score below 600: automated decline, unless an underwriter approves a documented exception.

## s.3 Decisioning

**3.1 Auto-decision limit** — Loans above INR 50,000 (or local currency equivalent) may never be
fully auto-approved; they must be routed to a human underwriter regardless of credit score.

**3.2 Adverse action notice** — If an application is declined, the applicant must be given the
main reason(s) for the decline within 30 days, in compliance with fair-lending disclosure
requirements.

## s.4 Controls and Audit

**4.1 Logging** — All decisions, manual overrides, and data changes must be logged with user
identity, timestamp, and previous/new values, for audit purposes.

## s.5 Security and Data Protection

**5.1 Data encryption** — Personal and financial data must be encrypted in transit (TLS 1.2+)
and at rest (AES-256 or equivalent).

## s.6 Availability

**6.1 Online application channel availability** — The online loan application channel must
maintain at least 99.9% monthly uptime.
