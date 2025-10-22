# Leadpoet Takedown & Suppression Request Process

**Last Updated:** January 18, 2025  
**Service Level Agreement (SLA):** 45 business days

---

## Overview

Leadpoet respects your privacy and compliance rights. If your contact information appears in our database and you wish to be removed, you may submit a takedown or suppression request.

**Types of Requests:**
1. **Opt-Out:** CAN-SPAM unsubscribe (immediate)
2. **Suppression:** Prevent future use in marketing (45-day SLA)
3. **Erasure:** GDPR "Right to be Forgotten" (45-day SLA)
4. **Complaint:** Report misuse or inaccuracy (45-day SLA)
5. **Do Not Call (DNC):** Phone number suppression (45-day SLA)

---

## Quick Reference

| Request Type | SLA | Identity Verification | Legal Basis |
|---|---|---|---|
| **Opt-Out** | Immediate | Email domain match | CAN-SPAM |
| **Suppression** | 45 days | Email domain match | Various |
| **Erasure** | 45 days | **Identity proof required** | GDPR Art. 17 |
| **Complaint** | 45 days | Email domain match | General |
| **Do Not Call** | 45 days | **Identity proof required** | TCPA |

---

## How to Submit a Request

### Option 1: Web Form (Recommended)

Visit: **https://leadpoet.com/takedown**

Fill out the form with:
- Your email address OR domain to suppress
- Request type (opt-out, erasure, complaint, DNC)
- Reason for request
- Your contact email (for follow-up)
- Optional: Identity proof (for GDPR/TCPA requests)

### Option 2: Email

Send an email to: **hello@leadpoet.com**

Include:
- Subject line: "Takedown Request"
- Your email address or domain to suppress
- Request type (opt-out, erasure, complaint, DNC)
- Reason for request
- Your contact email
- Attach identity proof if required

### Option 3: API (For Developers)

```bash
curl -X POST "https://your-project.supabase.co/functions/v1/log-takedown-request" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@company.com",
    "domain": "company.com",
    "requester_email": "your-email@company.com",
    "reason": "optout",
    "notes": "Optional notes"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Takedown request logged successfully. Will be reviewed within 45 days.",
  "request_id": "uuid-here",
  "status": "pending"
}
```

---

## Identity Verification Requirements

### When Identity Verification is Required

**REQUIRED for:**
- **GDPR Erasure Requests:** To protect against fraudulent deletion
- **Do Not Call Requests:** Federal TCPA requirements
- **Domain-wide suppressions:** To prevent abuse

**NOT REQUIRED for:**
- **Opt-Out Requests:** CAN-SPAM allows self-service opt-out
- **Single email suppressions:** If email domain matches requester domain

### Accepted Forms of Identity Proof

1. **Corporate Email Verification**
   - Request must come from email address at the domain being suppressed
   - Example: john@acme.com can suppress acme.com contacts

2. **Business License or Registration**
   - Copy of business registration document
   - Tax ID or EIN documentation

3. **Domain Ownership Verification**
   - Add TXT record to DNS: `leadpoet-verify=<code>`
   - We'll provide the verification code via email

4. **Government-Issued ID (for GDPR)**
   - Passport, driver's license, or national ID card
   - We'll redact all information except name and issuance authority

### How to Submit Identity Proof

**Option 1:** Upload to secure form at https://leadpoet.com/verify  
**Option 2:** Email encrypted PDF to hello@leadpoet.com (use PGP key available at https://leadpoet.com/pgp)  
**Option 3:** Send via secure file transfer (we'll provide link)

---

## Processing Timeline

### Phase 1: Request Logged (Immediate)

- Your request is logged in our suppression ledger
- You receive confirmation email with request ID
- Status: **Pending**

### Phase 2: Identity Verification (1-5 business days)

- We verify your identity (if required)
- We may contact you for additional information
- Status: **Verified** or **Rejected**

### Phase 3: Review (5-10 business days)

- Admin reviews request for legitimacy
- We check for abuse or fraudulent requests
- Status: **Approved** or **Rejected**

### Phase 4: Processing (1-3 business days)

- We suppress or delete matching records
- Updates propagate to all systems
- Status: **Processed**

### Phase 5: Confirmation (1 business day)

- You receive confirmation email
- Summary of actions taken (number of records affected)
- Status: **Completed**

**Total SLA:** Up to 45 business days (typically much faster)

---

## Suppression vs. Deletion

### Suppression (Soft Delete)

**What happens:**
- Records are flagged as `suppressed = TRUE`
- They will NOT be used in future marketing campaigns
- They remain in audit logs for compliance

**When to use:**
- CAN-SPAM opt-outs
- General suppression requests
- Temporary removals

**Reversible:** Yes (contact us to reverse)

### Deletion (Hard Delete)

**What happens:**
- Records are permanently removed from active databases
- Audit logs are retained (append-only, cannot delete)
- Cannot be reversed

**When to use:**
- GDPR "Right to Erasure" (Art. 17)
- CCPA deletion requests
- Legal orders requiring deletion

**Reversible:** No (permanent)

---

## Scope of Suppression

### Email-Specific Suppression

**Request:** "Remove john@acme.com"  
**Result:** Only this specific email is suppressed  
**Other emails:** Jane@acme.com, bob@acme.com remain active

### Domain-Wide Suppression

**Request:** "Remove all contacts from acme.com"  
**Result:** All emails with @acme.com are suppressed  
**Requires:** Identity verification proving you represent Acme Corp

### Phone Number Suppression

**Request:** "Remove phone number +1-555-0100"  
**Result:** This phone number is added to Do Not Call list  
**Requires:** Identity verification

---

## What Happens After Suppression

### Immediate Effects

1. **Prospect Queue:** Matching records flagged as suppressed
2. **Leads Table:** Matching records flagged as suppressed
3. **Future Submissions:** New submissions with suppressed contacts will be auto-rejected

### Buyer Notifications

**We do NOT automatically notify buyers who previously purchased your data.**

If you want buyers to stop using your information:
- You must contact them directly
- Provide them with your opt-out request
- They are legally required to honor CAN-SPAM opt-outs within 10 days

### Third-Party Distributors

If Leadpoet sold your data to third parties who then resold it:
- We cannot control third-party use
- You must contact those parties directly
- We will provide a list of known buyers upon request (subject to NDA)

---

## Verification of Takedown

### Check Suppression Status

Email hello@leadpoet.com with your request ID to check status.

**Response includes:**
- Current status (pending, verified, processed, completed)
- Number of records affected
- Timestamp of processing

### Request Report

You may request a detailed report showing:
- All records that were suppressed
- Dates those records were originally added
- Source attribution (if available)

**Note:** We redact miner wallet addresses to protect network participants.

---

## Appeals and Disputes

### If Your Request is Rejected

You will receive an email explaining why, which may include:
- Insufficient identity verification
- Request appears fraudulent
- Domain/email does not match our records

**To Appeal:**
1. Reply to rejection email with additional information
2. Provide requested identity proof
3. Clarify your legal basis for the request

### If You Believe We Missed Records

After processing, if you still see your information in our system:
1. Email hello@leadpoet.com with specifics
2. Include your original request ID
3. Provide evidence (screenshot, lead ID, etc.)

We will investigate within 10 business days.

---

## Special Request Types

### GDPR "Right to be Forgotten" (Art. 17)

**Eligibility:**
- You are an EU resident
- Data is no longer necessary for original purpose
- You withdraw consent
- Data was unlawfully processed
- Legal obligation requires erasure

**Process:**
1. Submit request via web form or email
2. Provide identity proof (government ID)
3. Specify legal basis (Art. 17)
4. We will assess eligibility
5. If approved, hard delete within 45 days
6. You receive confirmation and summary

**Exceptions (We may refuse if):**
- Data needed for compliance with legal obligation
- Data needed for establishment, exercise, or defense of legal claims
- Overriding legitimate interests apply

### CCPA Deletion Requests

**Eligibility:**
- You are a California resident
- Data is your personal information

**Process:**
Similar to GDPR, but:
- No requirement to provide government ID (unless we suspect fraud)
- We must confirm request via email
- We have 45 days to respond

### CAN-SPAM Opt-Out

**Eligibility:**
- You received a commercial email
- Email was sent by a buyer using our data

**Process:**
1. Click "Unsubscribe" in the email (fastest)
2. Or submit opt-out via our form
3. Immediate processing (no identity verification needed)
4. Takes effect within 10 business days per CAN-SPAM

**Note:** Opt-out is specific to the sender. If multiple buyers purchased your contact, you must opt-out with each sender individually.

---

## Fraudulent Requests

### We Will Reject Requests That:

- Attempt to suppress competitor's contacts
- Use forged identity documents
- Come from obviously fake email addresses
- Appear to be automated/mass submissions
- Lack required information

### Penalties for Fraud:

- Request rejection
- IP address blocked
- Report to authorities (if egregious)

---

## Contact Information

**Privacy Team Email:** hello@leadpoet.com  
**Data Protection Officer:** hello@leadpoet.com  
**Takedown Portal:** https://leadpoet.com/takedown  
**Support:** hello@leadpoet.com

**Office Address (for certified mail):**  
Leadpoet Compliance Department  
[Your Physical Address Here]  
[City, State, ZIP]

**Response Time:**  
- Acknowledgment: Within 5 business days
- Processing: Within 45 business days
- Urgent requests: Contact us directly

---

## Legal References

This takedown process complies with:
- **CAN-SPAM Act** (15 U.S.C. § 7701 et seq.)
- **GDPR** (EU Regulation 2016/679), specifically Article 17 (Right to Erasure)
- **CCPA/CPRA** (California Civil Code §§ 1798.100 et seq.)
- **TCPA** (47 U.S.C. § 227) for phone suppressions
- **PECR** (UK Privacy and Electronic Communications Regulations 2003)

---

## Frequently Asked Questions

### How long does it take?

**Target:** 10-15 business days  
**Maximum:** 45 business days per GDPR/CCPA

### Do I need to pay?

**No.** Takedown requests are free.

### Will you notify the miner who submitted my data?

**No.** We do not disclose your takedown request to miners, but we may suspend miners who repeatedly submit data that triggers takedowns.

### Can I request data about who has my information?

**Yes.** This is a "Right to Know" request (CCPA) or "Right of Access" (GDPR). Email hello@leadpoet.com.

### What if I change my mind?

**Before processing:** Contact us to cancel your request.  
**After suppression:** We can reverse soft-deletes. Email hello@leadpoet.com.  
**After hard delete:** Cannot be reversed.

### Do you charge buyers to honor suppressions?

**No.** Buyers receive updated suppression lists at no charge. They are responsible for filtering suppressed contacts.

---

**END OF TAKEDOWN PROCESS DOCUMENTATION**

*Questions? Email hello@leadpoet.com*

