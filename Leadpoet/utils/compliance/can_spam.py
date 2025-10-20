"""
CAN-SPAM Act Compliance Validation

This module provides utilities to validate email content and templates
for compliance with the CAN-SPAM Act (15 U.S.C. § 7701 et seq.).

The CAN-SPAM Act requires:
1. Accurate header information (From, To, Reply-To)
2. Non-deceptive subject lines
3. Clear identification as advertisement
4. Valid physical postal address
5. Clear and conspicuous opt-out mechanism
6. Prompt opt-out processing (10 business days)

Reference: https://www.ftc.gov/tips-advice/business-center/guidance/can-spam-act-compliance-guide-business
"""

from typing import Dict, List, Tuple
import re
import bittensor as bt


def validate_can_spam_compliance(email_content: Dict) -> Tuple[bool, List[str]]:
    """
    Validate email content meets CAN-SPAM requirements.
    
    Checks for:
    1. Truthful headers (From, To, Reply-To)
    2. Honest subject line
    3. Physical postal address included
    4. Clear opt-out mechanism (unsubscribe link)
    5. Valid sender identification
    
    Args:
        email_content: Dict with email template fields:
            - subject: Email subject line
            - from_address: Sender email address
            - from_name: Sender display name
            - reply_to: Reply-to email address (optional)
            - body_html: HTML email body
            - body_text: Plain text email body
            - physical_address: Sender's physical address
            - unsubscribe_link: Unsubscribe URL
            
    Returns:
        Tuple[bool, List[str]]: (is_compliant, list_of_errors)
        - is_compliant: True if all checks pass
        - list_of_errors: List of error messages (empty if compliant)
    """
    errors = []
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECK 1: Physical Address Present
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    physical_address = email_content.get("physical_address", "")
    
    if not physical_address:
        errors.append("CAN-SPAM: Physical postal address required in footer")
    else:
        # Verify address looks legitimate (has street + city/state)
        # Basic validation: should have at least a number and state/ZIP
        if not re.search(r'\d', physical_address):
            errors.append("CAN-SPAM: Physical address appears incomplete (no street number)")
        
        # Check for state or ZIP code
        has_location = (
            re.search(r'\b[A-Z]{2}\b', physical_address) or  # State abbreviation
            re.search(r'\b\d{5}\b', physical_address)         # ZIP code
        )
        
        if not has_location:
            errors.append("CAN-SPAM: Physical address should include city/state/ZIP")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECK 2: Unsubscribe Link Present
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    unsubscribe_link = email_content.get("unsubscribe_link", "")
    body_html = email_content.get("body_html", "")
    body_text = email_content.get("body_text", "")
    
    if not unsubscribe_link:
        errors.append("CAN-SPAM: Clear unsubscribe mechanism required")
    else:
        # Verify link is valid URL
        if not unsubscribe_link.startswith(('http://', 'https://', 'mailto:')):
            errors.append("CAN-SPAM: Unsubscribe link must be a valid URL")
        
        # Check that link appears in email body
        link_in_html = unsubscribe_link in body_html if body_html else False
        link_in_text = unsubscribe_link in body_text if body_text else False
        text_in_html = "unsubscribe" in body_html.lower() if body_html else False
        text_in_text = "unsubscribe" in body_text.lower() if body_text else False
        
        if not (link_in_html or link_in_text or text_in_html or text_in_text):
            errors.append("CAN-SPAM: Unsubscribe link must be visible in email body")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECK 3: Header Information (Truthful)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from_address = email_content.get("from_address", "")
    from_name = email_content.get("from_name", "")
    
    if not from_address:
        errors.append("CAN-SPAM: Valid 'From' email address required")
    else:
        # Discourage 'noreply' addresses (best practice, not strict requirement)
        if "noreply" in from_address.lower():
            errors.append("CAN-SPAM: 'noreply' addresses discouraged - provide valid reply address")
    
    if not from_name:
        errors.append("CAN-SPAM: Sender name/company required in 'From' field")
    
    # Check reply_to if provided
    reply_to = email_content.get("reply_to", "")
    if reply_to and "noreply" in reply_to.lower():
        errors.append("CAN-SPAM: Reply-to address should accept replies")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECK 4: Subject Line (Non-Deceptive)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    subject = email_content.get("subject", "")
    
    # Check for common deceptive patterns
    deceptive_patterns = [
        r'\bRE:\b',   # Fake reply
        r'\bFWD:\b',  # Fake forward
        r'\bFW:\b',   # Fake forward
        r'URGENT.*RESPONSE.*REQUIRED',  # Fake urgency
        r'ACCOUNT.*SUSPENDED',  # Fake security alert
    ]
    
    for pattern in deceptive_patterns:
        if re.search(pattern, subject, re.IGNORECASE):
            errors.append(f"CAN-SPAM: Subject line may be deceptive - avoid fake RE/FWD/urgent alerts")
            break
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECK 5: Advertisement Disclaimer (if applicable)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    is_commercial = email_content.get("is_commercial", True)  # Default to commercial
    
    if is_commercial:
        # Not strictly required by CAN-SPAM, but best practice for B2B
        has_ad_disclaimer = (
            ("advertisement" in body_html.lower() if body_html else False) or
            ("advertisement" in body_text.lower() if body_text else False)
        )
        
        # This is a soft warning, not a hard requirement
        if not has_ad_disclaimer:
            bt.logging.debug("Best practice: Consider adding 'This is an advertisement' for transparency")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Determine compliance
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    is_compliant = len(errors) == 0
    
    return is_compliant, errors


def validate_unsubscribe_mechanism(unsubscribe_url: str) -> Tuple[bool, str]:
    """
    Validate that unsubscribe URL meets CAN-SPAM requirements.
    
    Requirements:
    - Must be a valid URL (http/https/mailto)
    - Must not require login
    - Must not require payment
    - Should be one-click (or two-click with confirmation)
    
    Args:
        unsubscribe_url: The unsubscribe URL to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    if not unsubscribe_url:
        return False, "No unsubscribe URL provided"
    
    # Check URL format
    if not unsubscribe_url.startswith(('http://', 'https://', 'mailto:')):
        return False, "Unsubscribe URL must start with http://, https://, or mailto:"
    
    # Check for problematic patterns
    url_lower = unsubscribe_url.lower()
    
    # Should not require login
    if "login" in url_lower or "signin" in url_lower:
        return False, "Unsubscribe should not require login (CAN-SPAM violation)"
    
    # Should not require payment
    if "payment" in url_lower or "pay" in url_lower or "subscribe" in url_lower:
        return False, "Unsubscribe must be free (CAN-SPAM violation)"
    
    # Should contain email or token parameter for identification
    has_identifier = ("email=" in url_lower or "token=" in url_lower or "id=" in url_lower)
    
    if not has_identifier and not unsubscribe_url.startswith('mailto:'):
        bt.logging.warning("Best practice: Unsubscribe URL should include email or token parameter")
    
    return True, "Unsubscribe mechanism valid"


def check_physical_address_validity(address: str) -> Tuple[bool, str]:
    """
    Validate that physical address meets CAN-SPAM requirements.
    
    Requirements:
    - Must be current physical postal address
    - Can be P.O. Box or street address
    - Must include city, state, ZIP
    
    Args:
        address: Physical address string
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    if not address or len(address.strip()) < 10:
        return False, "Physical address too short or missing"
    
    # Check for basic components
    has_number = bool(re.search(r'\d+', address))
    has_state = bool(re.search(r'\b[A-Z]{2}\b', address))
    has_zip = bool(re.search(r'\b\d{5}(-\d{4})?\b', address))
    
    if not has_number:
        return False, "Address missing street number or P.O. Box number"
    
    if not (has_state or has_zip):
        return False, "Address missing state/ZIP code"
    
    # Check for obvious placeholders
    placeholder_patterns = [
        r'\bTODO\b',
        r'\bEXAMPLE\b',
        r'\bYOUR.?ADDRESS\b',
        r'\b123.?Main.?Street\b',
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, address, re.IGNORECASE):
            return False, f"Address appears to be a placeholder - use real address"
    
    return True, "Physical address valid"


def generate_can_spam_footer(
    company_name: str,
    address_line1: str,
    address_line2: str,
    unsubscribe_link: str,
    preferences_link: str = "",
    opt_in_source: str = "our website",
    opt_in_date: str = ""
) -> str:
    """
    Generate a CAN-SPAM compliant email footer.
    
    Args:
        company_name: Your company name
        address_line1: Street address or P.O. Box
        address_line2: City, State, ZIP
        unsubscribe_link: URL for unsubscribe
        preferences_link: URL for email preferences (optional)
        opt_in_source: Where recipient opted in
        opt_in_date: Date of opt-in
        
    Returns:
        str: HTML footer string ready to append to email
    """
    from datetime import datetime
    
    # Default opt-in date if not provided
    if not opt_in_date:
        opt_in_date = datetime.now().strftime("%B %Y")
    
    footer_html = f'''
<div style="font-family: Arial, sans-serif; font-size: 11px; color: #666666; background-color: #f8f8f8; padding: 20px; margin-top: 30px; border-top: 1px solid #dddddd;">
  
  <p style="margin: 0 0 10px 0;">
    <strong style="color: #333333;">{company_name}</strong><br>
    {address_line1}<br>
    {address_line2}
  </p>
  
  <p style="margin: 10px 0; line-height: 1.5;">
    You received this email because you provided your contact information at <strong>{opt_in_source}</strong> on {opt_in_date}.
  </p>
  
  <p style="margin: 10px 0; line-height: 1.5;">
    <a href="{unsubscribe_link}" style="color: #0066cc; text-decoration: underline;">Unsubscribe</a> from future emails'''
    
    if preferences_link:
        footer_html += f''' | <a href="{preferences_link}" style="color: #0066cc; text-decoration: underline;">Update Email Preferences</a>'''
    
    footer_html += '''
  </p>
  
  <p style="margin: 10px 0; font-size: 10px; color: #999999; line-height: 1.4;">
    This email was sent in compliance with the CAN-SPAM Act.
  </p>

</div>
'''
    
    return footer_html.strip()


def check_sender_authentication(email_content: Dict) -> Tuple[bool, List[str]]:
    """
    Check sender authentication requirements (best practices).
    
    While not strictly required by CAN-SPAM, sender authentication
    helps prevent spoofing and improves deliverability.
    
    Checks:
    - From address is valid email format
    - Reply-to address is valid (if present)
    - Domain has SPF/DKIM (informational only)
    
    Args:
        email_content: Email template dict
        
    Returns:
        Tuple[bool, List[str]]: (is_authenticated, list_of_warnings)
    """
    warnings = []
    
    from_address = email_content.get("from_address", "")
    reply_to = email_content.get("reply_to", "")
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, from_address):
        warnings.append("Invalid From address format")
    
    if reply_to and not re.match(email_pattern, reply_to):
        warnings.append("Invalid Reply-To address format")
    
    # Check if from_address and reply_to are different domains
    if from_address and reply_to:
        from_domain = from_address.split('@')[1] if '@' in from_address else ""
        reply_domain = reply_to.split('@')[1] if '@' in reply_to else ""
        
        if from_domain != reply_domain:
            warnings.append(f"From domain ({from_domain}) differs from Reply-To ({reply_domain}) - may appear suspicious")
    
    # Informational: SPF/DKIM checks would happen at send time
    # We can't check those here, but we can remind users
    if email_content.get("check_dns"):
        warnings.append("Reminder: Ensure your domain has SPF and DKIM records configured")
    
    is_authenticated = len(warnings) == 0
    
    return is_authenticated, warnings


def validate_subject_line(subject: str) -> Tuple[bool, List[str]]:
    """
    Validate that subject line is not deceptive.
    
    CAN-SPAM prohibits:
    - Materially false or misleading header info
    - Deceptive subject lines
    
    Args:
        subject: Email subject line
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)
    """
    warnings = []
    
    if not subject:
        warnings.append("Subject line is empty")
        return False, warnings
    
    # Check for deceptive patterns
    deceptive_patterns = {
        r'\bRE:\s': "Fake reply (RE:) may violate CAN-SPAM",
        r'\bFWD?:\s': "Fake forward (FWD:) may violate CAN-SPAM",
        r'URGENT.*RESPONSE.*REQUIRED': "Fake urgency may be considered deceptive",
        r'ACCOUNT.*SUSPENDED': "Fake security alert may be considered deceptive",
        r'VERIFY.*ACCOUNT': "Fake verification request may be considered deceptive",
        r'CLAIM.*PRIZE': "Fake prize notification likely violates CAN-SPAM",
        r'YOU.*WON': "Fake prize notification likely violates CAN-SPAM",
    }
    
    for pattern, warning_msg in deceptive_patterns.items():
        if re.search(pattern, subject, re.IGNORECASE):
            warnings.append(warning_msg)
    
    # Check for all caps (not illegal, but bad practice)
    if subject.isupper() and len(subject) > 10:
        warnings.append("All-caps subject line may trigger spam filters")
    
    # Check for excessive punctuation
    if subject.count('!') > 2 or subject.count('?') > 2:
        warnings.append("Excessive punctuation may trigger spam filters")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings


def validate_opt_out_processing(
    opt_out_request_date: str,
    opt_out_processed_date: str = None
) -> Tuple[bool, str]:
    """
    Validate that opt-out was processed within 10 business days.
    
    CAN-SPAM requires opt-outs to be processed within 10 business days.
    
    Args:
        opt_out_request_date: ISO timestamp of opt-out request
        opt_out_processed_date: ISO timestamp when processed (None if pending)
        
    Returns:
        Tuple[bool, str]: (is_compliant, message)
    """
    from datetime import datetime, timedelta
    
    try:
        request_dt = datetime.fromisoformat(opt_out_request_date.replace('Z', '+00:00'))
        
        if opt_out_processed_date:
            processed_dt = datetime.fromisoformat(opt_out_processed_date.replace('Z', '+00:00'))
            
            # Calculate business days (rough approximation - excludes weekends)
            days_elapsed = (processed_dt - request_dt).days
            business_days = days_elapsed * (5/7)  # Approximate
            
            if business_days > 10:
                return False, f"Opt-out processed late ({business_days:.1f} business days - max is 10)"
            else:
                return True, f"Opt-out processed on time ({business_days:.1f} business days)"
        else:
            # Check if still pending beyond 10 days
            now = datetime.now(request_dt.tzinfo)
            days_pending = (now - request_dt).days
            business_days_pending = days_pending * (5/7)
            
            if business_days_pending > 10:
                return False, f"Opt-out pending for {business_days_pending:.1f} business days (exceeds 10-day limit)"
            else:
                return True, f"Opt-out pending for {business_days_pending:.1f} business days (within limit)"
                
    except Exception as e:
        return False, f"Error validating opt-out timing: {e}"


def get_can_spam_checklist() -> List[str]:
    """
    Get a checklist of CAN-SPAM requirements.
    
    Returns:
        List[str]: List of compliance requirements
    """
    return [
        "✓ Don't use false or misleading header information",
        "✓ Don't use deceptive subject lines",
        "✓ Identify the message as an advertisement (if applicable)",
        "✓ Tell recipients where you're located (physical address)",
        "✓ Tell recipients how to opt out",
        "✓ Honor opt-out requests promptly (within 10 business days)",
        "✓ Monitor what others are doing on your behalf (if using third-party marketers)",
    ]


def log_can_spam_validation(email_id: str, is_compliant: bool, errors: List[str]) -> None:
    """
    Log CAN-SPAM validation result for audit trail.
    
    Args:
        email_id: Unique identifier for email campaign
        is_compliant: Whether email passed validation
        errors: List of compliance errors
    """
    if is_compliant:
        bt.logging.info(f"✅ CAN-SPAM validation passed for email: {email_id}")
    else:
        bt.logging.warning(f"❌ CAN-SPAM validation failed for email: {email_id}")
        for error in errors:
            bt.logging.warning(f"   - {error}")

