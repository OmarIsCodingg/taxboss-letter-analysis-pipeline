TaxBoss — Letter Analysis Pipeline

A standalone excerpt from TaxBoss, a SaaS product I'm building for Dutch freelancers (ZZP'ers). This is the three-stage AI pipeline that analyzes Dutch tax authority letters:

OCR (Claude Sonnet) — extracts text from an uploaded letter image
Classification (Claude Haiku) — categorizes the letter into one of ~18 types with structured metadata
Explanation (Claude Sonnet) — generates a plain-language, actionable explanation for the user

This is a code excerpt shared for a job application, not the full application. Authentication, database persistence, subscription logic, and TaxBoss-specific wiring have been intentionally removed.
