# How the Gateway Works (And Why It's Trustless)

**For miners & validators:** Why you can trust the gateway even though the subnet owner runs it.

---

## The Problem

Without special precautions, the subnet owner could:
- Run modified code that favors certain miners
- Selectively drop submissions
- Manipulate event logs

---

## The Solution: Hardware-Protected Execution

The gateway runs inside an **AWS Nitro Enclave** - a hardware-protected environment that even the subnet owner can't tamper with. Think of it like a sealed black box where the CPU itself (not software) enforces isolation.

**What happens:**
1. Gateway boots inside the enclave
2. Enclave generates a keypair (private key never leaves)
3. AWS Nitro hardware creates an **attestation document** - a cryptographic proof saying: "I'm running code with hash XYZ"
4. When you submit a lead, it's stored in enclave memory (hardware-isolated)
5. Every hour, enclave signs all events and uploads to Arweave

---

## Why This Is Trustless

**Q: What if the operator runs modified code?**  
**A:** Attestation contains PCR0 (code hash). Modified code = different hash = you detect it immediately.

**Q: What if they fake the attestation?**  
**A:** PCR measurements come directly from `/dev/nsm` hardware. Parent EC2 can't influence it.

**Q: What if they modify events after acceptance?**  
**A:** Events are in enclave memory - hardware-isolated. Even root access can't read/modify it.

**Q: What if they selectively log events?**  
**A:** Enclave signs the Merkle root of ALL events. Missing events = signature mismatch.

**Known limitation:** Operator could feed fake blockchain state to the enclave. Mitigations: miner signatures prevent forgery, blockchain state is publicly verifiable, transparency log shows patterns.

---

## How To Verify

### Setup
```bash
git clone https://github.com/leadpoet/Leadpoet.git
cd Leadpoet
pip install -e .
```

### 1. Verify Code Integrity

**Step A: Get attestation and extract PCR0**
```bash
# Get attestation
curl http://54.80.97.12:8000/attest > attestation.json

# Verify AWS Nitro signature + extract PCR0
python scripts/verify_attestation.py http://54.80.97.12:8000
```

**Step B: Verify PCR0 matches expected code**

*Option 1 (Simple):* Compare to published value
```bash
# Get expected PCR0 from Discord/docs
EXPECTED_PCR0="d2106245cba92cdba289501ef56a6c0e..."
GATEWAY_PCR0=$(cat attestation.json | jq -r '.pcr0')

if [ "$GATEWAY_PCR0" == "$EXPECTED_PCR0" ]; then
  echo "✅ CODE HASH MATCHES"
else
  echo "❌ DO NOT TRUST THIS GATEWAY"
fi
```

*Option 2 (Advanced):* Build locally and compare (requires Docker + Nitro CLI)
```bash
# Get GitHub commit from gateway docs
GITHUB_COMMIT="abc123def456"

# Build locally and verify
python scripts/verify_code_hash.py $GATEWAY_PCR0 $GITHUB_COMMIT
```

If PCR0 matches → ✅ Gateway is running canonical code  
If PCR0 doesn't match → 🚨 **DO NOT USE THIS GATEWAY**

### 2. Verify Your Submission
```bash
# After 1 hour, verify your event is in the Arweave checkpoint
python scripts/verify_merkle_inclusion.py <your_lead_id> <checkpoint_tx_id>
```

### 🚨 Red Flags
**DO NOT USE the gateway if:**
- PCR0 is all zeros (debug mode)
- PCR0 doesn't match published value
- Attestation certificate is invalid
- Your events are consistently missing from checkpoints

---

**Questions?** See `scripts/VERIFICATION_GUIDE.md` or ask in Discord #leadpoet

**TL;DR:** Hardware-enforced isolation guarantees the gateway runs correct code. Even a malicious subnet owner can't cheat without cryptographic detection.


