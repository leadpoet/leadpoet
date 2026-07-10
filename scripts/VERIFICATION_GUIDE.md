# Gateway Verification Guide

**For Miners & Validators**: How to verify gateway attestation, validator TEE PCR0, and checkpoint integrity without trusting the operator.

---

## **🎯 What You're Verifying**

The gateway uses an **AWS Nitro Enclave (TEE)** for attestation and signing, and it independently verifies validator TEEs before accepting signed weights. You can verify:

1. ✅ **TEE Integrity**: Gateway and validator attestations are signed by AWS Nitro hardware
2. ✅ **Event Inclusion**: Your events are logged and included in Arweave checkpoints
3. ✅ **Signature Validity**: Checkpoints are signed by the verified TEE enclave
4. ✅ **Validator Weight Integrity**: Validator PCR0 is independently rebuilt by the gateway from GitHub before weights are accepted

**Why This Matters**: Even if the subnet owner is malicious, they CANNOT silently run modified TEE code without detection. AWS Nitro attestation provides the hardware proof, and the gateway's independent PCR0 builder verifies validator weights against code reproducible from GitHub.

---

## **📋 Prerequisites**

```bash
# Python dependencies
pip install cbor2 cryptography requests

# For validator PCR0 rebuild verification (optional, advanced)
# Install Docker: https://docs.docker.com/get-docker/
# Install AWS Nitro CLI: https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave-cli-install.html
```

---

## **Step 1: Verify Attestation Document**

**Purpose**: Verify the gateway's attestation comes from a genuine AWS Nitro Enclave and extract the code integrity proof (PCR0).

```bash
python scripts/verify_attestation.py http://52.91.135.79:8000
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

**Save the PCR0 and code_hash values** - you'll use them when comparing attestations over time.

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

## **Step 2: Verify Code Hash / PCR0**

**Purpose**: Prove the attested code matches the expected GitHub code path.

### **Option A: Gateway Code Hash Check** (Recommended for miners)

The gateway `/attest` response includes a `code_hash` bound into the Nitro attestation. You can recompute that hash from GitHub and compare:

```bash
python scripts/verify_code_hash.py http://52.91.135.79:8000 --commit <github_commit>
```

**What it does**:
1. Downloads the gateway attestation from `/attest`
2. Extracts the attested `code_hash`
3. Clones GitHub at the specified commit
4. Recomputes the same gateway code hash locally
5. Compares your local hash to the attested hash

**If they match**: Gateway attestation is bound to the expected GitHub code hash ✅
**If they don't match**: Gateway code hash does not match the specified commit - **DO NOT TRUST** ❌

### **Option B: Validator PCR0 Rebuild Check** (Advanced)

For validator weight submissions, the gateway independently rebuilds the validator enclave PCR0 from GitHub and compares it to the validator's AWS Nitro attestation before accepting signed weights.

```bash
# Requires Docker + Nitro CLI on a Nitro-capable machine
git clone https://github.com/leadpoet/leadpoet.git
cd leadpoet
git checkout <validator_commit>

docker rmi -f validator-base:v1 2>/dev/null || true
docker build --no-cache -f validator_tee/Dockerfile.base -t validator-base:v1 .
bash validator_tee/scripts/build_enclave.sh

# PCR0 is printed in validator_tee/enclave_build_output.txt
grep '"PCR0"' validator_tee/enclave_build_output.txt
```

**What it does**:
1. Clones the GitHub repo at the specified commit
2. Builds the same validator base image and enclave image
3. Normalizes timestamps and copied-file permissions for reproducibility
4. Computes PCR0 using `nitro-cli build-enclave`
5. Lets you compare your PCR0 to the validator PCR0 in an AWS Nitro attestation

**If they match**: Validator weights came from an enclave image reproducible from the same GitHub commit ✅
**If they don't match**: Validator enclave code or build inputs differ - **DO NOT TRUST THOSE WEIGHTS** ❌

**Important**: The match is between the validator's attested PCR0 and the gateway's independently rebuilt expected validator PCR0. That proves the validator enclave image matches the PCR0-relevant files from the same GitHub commit. The gateway's own `/attest` PCR0 is a separate gateway-enclave measurement.

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
❌ CODE HASH / PCR0 MISMATCH
```

**Meaning**: The attested code does not match the specified GitHub commit or the validator PCR0 does not match the independently rebuilt validator enclave.

**Possible causes**:
1. **Modified TEE code** - gateway or validator is running code that does not match GitHub
2. Wrong commit hash - you verified against the wrong version
3. Validator and gateway are on different commits
4. Build non-determinism or dirty local build inputs

**Action**: **DO NOT TRUST THE AFFECTED GATEWAY OR WEIGHT SUBMISSION** until you investigate. Notify the community.

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
2. ❌ Gateway code hash does not match the expected GitHub commit
3. ❌ Validator PCR0 does not match an independent rebuild from the same commit
4. ❌ Attestation certificate is invalid or expired
5. ❌ Checkpoint signatures fail verification
6. ❌ Events consistently missing from checkpoints

**In these cases, the gateway is either misconfigured or malicious. Do not use it.**

---

## **✅ Best Practices**

### **For Miners**

1. **Verify gateway before first submission**:
   ```bash
   python scripts/verify_attestation.py https://gateway.leadpoet.ai
   python scripts/verify_code_hash.py https://gateway.leadpoet.ai --commit <github_commit>
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
   - Same as miners: verify attestation + code hash

2. **Verify validator PCR0 for weight submissions**:
   - Validator attestation PCR0 must match a reproducible build from the same GitHub commit
   - The gateway performs this check automatically before accepting signed weights

3. **Verify epoch events on Arweave**:
   - Download `EPOCH_INITIALIZATION` from Arweave checkpoint
   - Verify all validators got the same assignment
   - Verify queue_merkle_root is correct

4. **Cross-check with other validators**:
   - Compare notes: did everyone get the same leads?
   - Compare attestation code hashes and validator PCR0 values
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

1. **Check community announcements** - confirm the active GitHub commit and gateway URL
2. **Ask in Discord/Telegram** - compare results with other miners and validators
3. **Report to developers** - if you suspect malicious activity

**Remember**: The verification scripts are your defense against malicious gateways. Use them regularly! 🛡️
