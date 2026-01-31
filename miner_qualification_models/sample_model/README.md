# Sample Qualification Model

This is a **SAMPLE** implementation of a lead qualification model for testing purposes.

## ⚠️ Important

This sample generates **FAKE** intent signals and will score poorly in real evaluation!

Real models should:
1. Search the web for actual intent signals
2. Use LinkedIn/job board APIs to find real activity
3. Use LLMs for sophisticated ICP matching
4. Verify all data before returning

## Usage

```python
from qualify import qualify

lead = {
    "email": "john@example.com",
    "full_name": "John Smith",
    "business": "Acme Corp",
    "role": "VP Engineering",
    "industry": "Technology"
}

icp = {
    "industry": "Technology",
    "target_role": "VP Director",
    "product_service": "cloud software"
}

result = qualify(lead, icp)
```

## Required Function

Your model MUST have a `qualify(lead, icp)` function that:
- Accepts lead and ICP dicts
- Returns a qualified lead dict or None
- Includes an `intent_signal` with real evidence
