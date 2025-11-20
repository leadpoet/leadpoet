# Gateway Verification Guide

**For Miners & Validators**: How to verify the gateway is running canonical code and hasn't been tampered with.

---

## **🎯 What You're Verifying**

The gateway runs inside an **AWS Nitro Enclave (TEE)** that provides cryptographic proof of code integrity. You can verify:

1. ✅ **Code Integrity**: Gateway is running the exact code from GitHub (not modified)
2. ✅ **Event Inclusion**: Your events are logged and included in Arweave checkpoints
3. ✅ **Signature Validity**: Checkpoints are signed by the verified TEE enclave

**Why This Matters**: Even if the subnet owner is malicious, they CANNOT run modified code without detection. The TEE attestation makes this cryptographically provable.

---

## **📋 Prerequisites**

```bash
# Python dependencies
pip install cbor2 cryptography requests

# For code hash verification (optional, advanced)
# Install Docker: https://docs.docker.com/get-docker/
# Install AWS Nitro CLI: https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave-cli-install.html
```

---

## **Step 1: Verify Attestation Document**

**Purpose**: Verify the gateway is running inside a genuine AWS Nitro Enclave and extract the code integrity proof (PCR0).

```bash
python scripts/verify_attestation.py http://54.80.97.12:8000
```

**What it does**:
1. Downloads attestation document from `/attest` endpoint
2. Parses COSE Sign1 structure (AWS Nitro format)
3. Verifies AWS Nitro certificate chain
4. Extracts **PCR0** (enclave image hash - this is the code integrity proof)
5. Extracts PCR1, PCR2 (kernel and ramdisk hashes)
6. Shows enclave public key and timestamp

**Example output**:
```
================================================================================
🔐 AWS NITRO ENCLAVE ATTESTATION VERIFIER
================================================================================
📥 Downloading attestation from https://gateway.leadpoet.ai/attest...
   ✅ Downloaded
🔍 Parsing COSE Sign1 structure...
   Protected headers: 4 bytes
   Unprotected headers: <class 'dict'>
   Payload: 4354 bytes
   Signature: 96 bytes

📋 Attestation Document:
   Module ID: i-098ad34c6c2f1cf66-enc019a70bc677080f6
   Timestamp: 1762827994310 ms
   Digest: SHA384

🔒 PCR Measurements (16 total):
   PCR0 (Enclave Image): d2106245cba92cdba289501ef56a6c0e...
   PCR1 (Kernel):        4b4d5b3661b3efc12920900c80e126e4...
   PCR2 (Ramdisk):       f3acfa6acbd051a4a6e3de674b29966c...

🔐 Verifying certificate chain...
   Leaf certificate:
      Subject: CN=..., O=Amazon, C=US
      Valid from: 2025-11-09 00:00:00
      Valid until: 2026-11-09 00:00:00
   ✅ Certificate valid

================================================================================
📊 VERIFICATION RESULT
================================================================================
✅ ATTESTATION CERTIFICATE VALID

================================================================================
PCR0 (Code Integrity Proof): d2106245cba92cdba289501ef56a6c0e972fa100bd3ddde671570bf732ce16f77bc92f239a30150599f24858fd0bb6ff
================================================================================
```

**Save the PCR0 value** - you'll use it in Step 2.

### **⚠️ Debug Mode Warning**

If you see:
```
⚠️  PCR0 is all zeros (enclave running in DEBUG MODE)
⚠️  Debug mode attestations are NOT SECURE (console access enabled)
```

**This means the gateway is running in debug mode**, which:
- Allows console access to the enclave (breaks memory isolation)
- Zeroes out PCR0-3 (intentional AWS Nitro security feature)
- **Should NEVER be used in production**

Debug mode is fine for development/testing, but **reject any production gateway running in debug mode**.

---

## **Step 2: Verify Code Hash (PCR0)**

**Purpose**: Prove the gateway is running the exact code from a specific GitHub commit.

### **Option A: Compare Against Known Good PCR0** (Recommended)

If a trusted source (subnet documentation, Discord announcement) publishes the expected PCR0 for a specific commit:

```bash
# Get PCR0 from Step 1
GATEWAY_PCR0="d2106245cba92cdba289501ef56a6c0e972fa100bd3ddde671570bf732ce16f7..."

# Get expected PCR0 from trusted source
EXPECTED_PCR0="d2106245cba92cdba289501ef56a6c0e972fa100bd3ddde671570bf732ce16f7..."

# Compare
if [ "$GATEWAY_PCR0" == "$EXPECTED_PCR0" ]; then
  echo "✅ CODE HASH MATCHES"
else
  echo "❌ CODE HASH MISMATCH - DO NOT TRUST THIS GATEWAY"
fi
```

### **Option B: Build Locally and Compute PCR0** (Advanced)

If you want to independently verify (requires Docker + Nitro CLI):

```bash
# Get PCR0 from Step 1
GATEWAY_PCR0="d2106245cba92cdba289501ef56a6c0e..."

# Get GitHub commit hash (from gateway docs or /attest endpoint)
GITHUB_COMMIT="abc123def456"

# Build locally and compare
python scripts/verify_code_hash.py $GATEWAY_PCR0 $GITHUB_COMMIT
```

**What it does**:
1. Clones the GitHub repo at the specified commit
2. Builds the enclave Docker image locally
3. Computes PCR0 using `nitro-cli build-enclave`
4. Compares your local PCR0 to the gateway's PCR0

**If they match**: Gateway is running the exact code from GitHub ✅  
**If they don't match**: Gateway is running modified code - **DO NOT TRUST** ❌

---

## **Step 3: Query Real-Time Transparency Log (Supabase)**

**Purpose**: Query the live transparency log from Supabase to track events in real-time (before they're batched to Arweave).

**🔍 Use Cases**: Debug submissions, track lead journey, monitor epoch events, find consensus results.

### **Query Script**

Edit variables at top of `scripts/query_transparency_log.py`:

```python
EMAIL_HASH = ""           # Track specific lead (highest priority)
EVENT_TYPE = ""           # Filter by event type: SUBMISSION_REQUEST, CONSENSUS_RESULT, etc.
SPECIFIC_DATE = ""        # All from date: "2025-11-20" (medium priority)  
LAST_X_HOURS = 8          # Last X hours (default if above blank)
```

Then run:
```bash
python scripts/query_transparency_log.py
```

**Output**: Real-time events with full payloads, TEE sequences, Arweave TX IDs, and complete audit trail.

**Event Types**: `SUBMISSION_REQUEST`, `STORAGE_PROOF`, `SUBMISSION`, `CONSENSUS_RESULT`, `EPOCH_INITIALIZATION`, `EPOCH_END`, `DEREGISTERED_MINER_REMOVAL`, `RATE_LIMIT`

---

## **Step 4: View Complete Event Logs from Arweave**

**Purpose**: Decompress and view all events stored in Arweave. Events are gzip-compressed (100% lossless) to save 96% on storage costs.

**🔒 Trustless**: Script queries Arweave GraphQL directly using tags - does NOT rely on subnet owners' database!

### **Decompression Script**

Edit variables at top of `scripts/decompress_arweave_checkpoint.py`:

```python
ARWEAVE_TX_ID = ""        # Specific checkpoint (highest priority)
SPECIFIC_DATE = ""         # All from date: "2025-11-14" (medium priority)
LAST_X_HOURS = 4           # Last X hours (default if above blank)
```

Then run:
```bash
python scripts/decompress_arweave_checkpoint.py
```

**Output**: Complete events with lead_id, email_hash, lead_blob_hash, hotkeys, timestamps, validator decisions, consensus results, and TEE signatures.

---

## **Step 5: Verify Event Inclusion in Checkpoint**

**Purpose**: Prove your event (submission, validation, reveal) was logged and is permanently stored on Arweave.

```bash
# Your lead_id from submission response
LEAD_ID="8b6482bf-116e-41db-b8ec-a87ba3c86b8b"

# Arweave checkpoint transaction ID (from hourly checkpoint)
# Find this by checking checkpoints with timestamps covering your event
CHECKPOINT_TX="abc123def456..."

# Enclave public key (from Step 1 output)
ENCLAVE_PUBKEY="a1b2c3d4..."  # Optional but recommended

python scripts/verify_merkle_inclusion.py $LEAD_ID $CHECKPOINT_TX $ENCLAVE_PUBKEY
```

**What it does**:
1. Downloads checkpoint from Arweave
2. Verifies TEE signature on checkpoint header (if public key provided)
3. Searches for your event in the checkpoint batch
4. Computes Merkle root from all events
5. Verifies computed root matches header's merkle_root

**Example output**:
```
================================================================================
🔐 MERKLE INCLUSION VERIFIER
================================================================================
Lead ID:        8b6482bf-116e-41db-b8ec-a87ba3c86b8b
Checkpoint TX:  abc123def456...
Enclave Pubkey: a1b2c3d4...

📥 Downloading checkpoint abc123def456... from Arweave...
   ✅ Downloaded checkpoint (500 KB)

📋 Checkpoint Header:
   Merkle Root: 5e992c955d6325bdacb73960a197e443...
   Event Count: 1200
   Time Range: 2025-11-09T00:00:00Z to 2025-11-09T01:00:00Z
   Code Hash: d2106245cba92cdba289501ef56a6c0e...

🔐 Verifying checkpoint signature...
   ✅ Signature valid - checkpoint signed by TEE enclave

🔍 Searching for event with lead_id=8b6482bf-116e-41db-b8ec-a87ba3c86b8b...
   Total events in checkpoint: 1200
   ✅ Event found at index 542
   Event type: SUBMISSION
   Timestamp: 2025-11-09T00:15:23Z

🌳 Computing Merkle root from events...
   Expected root (from header): 5e992c955d6325bdacb73960a197e443...
   Computed root (from events): 5e992c955d6325bdacb73960a197e443...
   ✅ Merkle root matches!

================================================================================
✅ EVENT INCLUDED IN CHECKPOINT
================================================================================
Your event was successfully logged by the gateway and included
in the Arweave checkpoint with a valid Merkle proof.

This proves:
  ✅ Event was accepted by gateway
  ✅ Event is permanently stored on Arweave
  ✅ Event cannot be retroactively modified or deleted
```

### **Finding the Right Checkpoint**

Checkpoints are created **hourly**. To find your event:

1. Note your event timestamp (from submission response)
2. Calculate the hour: `2025-11-09T00:15:23Z` → checkpoint for hour `00:00-01:00`
3. Query Arweave for checkpoints in that time range
4. Try the checkpoint with `time_range` covering your timestamp

---

## **🔍 What Can Go Wrong?**

### **PCR0 Mismatch** ❌
```
❌ CODE HASH MISMATCH - Gateway is running modified code!
```

**Meaning**: The gateway is NOT running the code from the specified GitHub commit.

**Possible causes**:
1. **Malicious gateway** - running modified code to favor certain miners
2. Wrong commit hash - you verified against the wrong version
3. Build non-determinism - rare, but Docker builds can sometimes vary

**Action**: **DO NOT USE THIS GATEWAY** until you investigate. Notify the community.

---

### **Event Not Found** ❌
```
❌ Event not found in this checkpoint
```

**Meaning**: Your event is not in this specific checkpoint.

**Possible causes**:
1. **Wrong checkpoint** - try checkpoints before/after this one
2. **Not yet checkpointed** - checkpoints are created hourly, wait up to 1 hour
3. **Event was rejected** - check if your submission actually succeeded

**Action**: Wait for next checkpoint, or try adjacent checkpoints.

---

### **Merkle Root Mismatch** ❌
```
❌ Merkle root mismatch! Checkpoint data may be corrupted or tampered with
```

**Meaning**: The checkpoint data doesn't match the signed Merkle root in the header.

**Possible causes**:
1. **Corrupted download** - try downloading again
2. **Tampered checkpoint** - someone modified the Arweave data (very unlikely)
3. **Bug in verification script** - Merkle algorithm mismatch

**Action**: Re-download and verify again. If still failing, report to developers.

---

## **🚨 Red Flags - When to Reject a Gateway**

**Reject the gateway if**:

1. ❌ PCR0 is all zeros (debug mode in production)
2. ❌ PCR0 doesn't match published expected value
3. ❌ Attestation certificate is invalid or expired
4. ❌ Checkpoint signatures fail verification
5. ❌ Events consistently missing from checkpoints

**In these cases, the gateway is either misconfigured or malicious. Do not use it.**

---

## **✅ Best Practices**

### **For Miners**

1. **Verify gateway before first submission**:
   ```bash
   python scripts/verify_attestation.py https://gateway.leadpoet.ai
   # Save PCR0, compare to community-announced value
   ```

2. **Verify event inclusion after submission**:
   - Note your `lead_id` and submission timestamp
   - Wait 1 hour for checkpoint
   - Verify inclusion using `verify_merkle_inclusion.py`

3. **Periodically re-check attestation**:
   - Gateway could be restarted with modified code
   - Check attestation every 24 hours or before important submissions

### **For Validators**

1. **Verify gateway before accepting assignments**:
   - Same as miners: verify attestation + PCR0

2. **Verify epoch events on Arweave**:
   - Download `EPOCH_INITIALIZATION` from Arweave checkpoint
   - Verify all validators got the same assignment
   - Verify queue_merkle_root is correct

3. **Cross-check with other validators**:
   - Compare notes: did everyone get the same leads?
   - Compare attestation PCR0 values
   - If discrepancies found, investigate immediately

---

## **📚 Additional Resources**

- **AWS Nitro Enclaves Documentation**: https://docs.aws.amazon.com/enclaves/latest/user/
- **Arweave Documentation**: https://docs.arweave.org/
- **COSE Sign1 Specification**: https://datatracker.ietf.org/doc/html/rfc8152

---

## **🛠️ Troubleshooting**

### **Script fails with "Missing dependencies"**

```bash
pip install cbor2 cryptography requests
```

### **`nitro-cli` not found** (for code hash verification)

Install AWS Nitro CLI:
- Amazon Linux 2: https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave-cli-install.html
- Note: Nitro CLI only works on EC2 instances with enclave support

### **Cannot connect to gateway /attest endpoint**

- Check gateway URL is correct
- Check gateway is running
- Check firewall/network connectivity

### **Checkpoint download times out**

- Arweave can be slow, increase timeout
- Try different Arweave gateway: `https://arweave.net` or `https://ar-io.net`

---

## **💬 Questions or Issues?**

If you find a verification failure or have questions:

1. **Check community announcements** - expected PCR0 values should be published
2. **Ask in Discord/Telegram** - other miners may have encountered the same issue
3. **Report to developers** - if you suspect malicious activity

**Remember**: The verification scripts are your defense against malicious gateways. Use them regularly! 🛡️

