# Migration Tasks: Firestore to Supabase with JWT-Based Access Control

## Summary of Context & Corrected Workflow

**The Problem**: Validators could register as miners, read the entire database (including recent leads from the last 72 minutes), and game the system by publishing correct weights without doing validation work.

**The Solution**: Migrate from Firestore to Supabase with JWT-based access control:
- **Miners**: JWT with `app_role="miner"` → READ-only access to leads >72 minutes old (enforced by RLS)
- **Validators**: JWT with `app_role="validator"` → WRITE-only access
- **Automated Issuance**: When someone registers on the subnet, they call an API endpoint to automatically receive their JWT token based on their metagraph role

**The Corrected Workflow**:
1. Miner/validator registers on the Bittensor subnet
2. They clone the repo
3. They call your API endpoint: `curl https://your-issuer.com/issue-jwt --data '{"hotkey":"5ABC..."}'`
4. Server checks metagraph, determines if they're a miner or validator, issues JWT automatically
5. They receive JWT token in response
6. They add `SUPABASE_JWT=<token>` to `.env`
7. They run their node - done!

---

## Phase 1: Supabase Setup & Schema Migration

### Task 1.1: Use Your Existing Supabase Project
**File**: N/A (already set up)

**Your existing project details**:
- Project ID: `qplwoislplkcegvdmbim`
- Project URL: `https://qplwoislplkcegvdmbim.supabase.co`
- Anon public key: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8`
- Service role key (SECRET - never share): `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDg0NzAwNSwiZXhwIjoyMDYwNDIzMDA1fQ.L3QKWAcZUC-D3-fN4ogueIAmQhuJEjZ2UpYfSlJPp_I`

**Action items**:
1. ✅ Project already exists - no new project creation needed
2. Go to https://supabase.com/dashboard/project/qplwoislplkcegvdmbim
3. Access your project's SQL Editor and Database settings
4. Get your JWT Secret from Settings > API > JWT Settings (you'll need this for Task 2.3)

**Important**: All subsequent tasks will use this existing project. We'll be adding new tables, Edge Functions, and RLS policies to this project.

**Note on JWT Secret**: 
To get your JWT Secret (needed for Task 2.3 Edge Function):
1. Go to https://supabase.com/dashboard/project/qplwoislplkcegvdmbim/settings/api
2. Scroll to "JWT Settings"
3. Copy the "JWT Secret" value
4. You'll use this when deploying the Edge Function in Task 2.3

### Task 1.2: Create Database Schema
**File**: Supabase Dashboard / SQL Editor

Execute the following SQL to create the schema:

```sql
-- Leads table (equivalent to Firestore collection)
CREATE TABLE leads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL,
  company TEXT,
  validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  validator_hotkey TEXT NOT NULL,
  miner_hotkey TEXT,
  score FLOAT,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_leads_validated_at ON leads(validated_at);
CREATE INDEX idx_leads_validator_hotkey ON leads(validator_hotkey);
CREATE INDEX idx_leads_miner_hotkey ON leads(miner_hotkey);

-- Add other tables as needed...
```

### Task 1.3: Enable Row Level Security (RLS)
**File**: Supabase Dashboard / SQL Editor

Enable RLS on all tables:

```sql
ALTER TABLE leads ENABLE ROW LEVEL SECURITY;
-- Repeat for all tables
```

### Task 1.4: Create RLS Policies
**File**: Supabase Dashboard / SQL Editor

Create policies for role-based access:

```sql
-- Policy 1: Miners can READ leads older than 72 minutes
CREATE POLICY "miner_read_old_leads" ON leads
FOR SELECT
TO authenticated
USING (
  (auth.jwt() ->> 'app_role') = 'miner'
  AND validated_at < NOW() - INTERVAL '72 minutes'
);

-- Policy 2: Validators can WRITE (INSERT only)
CREATE POLICY "validator_write_access_leads" ON leads
FOR INSERT
TO authenticated
WITH CHECK (
  (auth.jwt() ->> 'app_role') = 'validator'
);

-- Repeat similar policies for other tables...
```

---

## Phase 2: Automated JWT Token Issuance System

### Task 2.1: Create Supabase Tables for Token Management
**File**: Supabase Dashboard / SQL Editor

Add these tables for token tracking, revocation, and rate limiting:

```sql
-- Track all registered members and their roles
CREATE TABLE members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  hotkey TEXT UNIQUE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('miner', 'validator')),
  registered_at TIMESTAMPTZ DEFAULT NOW(),
  approved BOOLEAN DEFAULT TRUE,
  last_verified TIMESTAMPTZ
);

CREATE INDEX idx_members_hotkey ON members(hotkey);

-- Track all issued JWT tokens for audit and revocation
CREATE TABLE token_issuances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  token_id TEXT UNIQUE NOT NULL,
  hotkey TEXT NOT NULL,
  role TEXT NOT NULL,
  issued_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  revoked BOOLEAN DEFAULT FALSE,
  revoked_at TIMESTAMPTZ,
  revoked_by TEXT,
  revoked_reason TEXT
);

CREATE INDEX idx_token_issuances_token_id ON token_issuances(token_id);
CREATE INDEX idx_token_issuances_hotkey ON token_issuances(hotkey);
CREATE INDEX idx_token_issuances_revoked ON token_issuances(revoked);

-- Track token issuance attempts for rate limiting
CREATE TABLE token_issuance_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  hotkey TEXT NOT NULL,
  attempted_at TIMESTAMPTZ DEFAULT NOW(),
  success BOOLEAN DEFAULT FALSE,
  error_message TEXT
);

CREATE INDEX idx_token_issuance_attempts_hotkey ON token_issuance_attempts(hotkey);
CREATE INDEX idx_token_issuance_attempts_attempted_at ON token_issuance_attempts(attempted_at);
```

### Task 2.2: Update RLS Policies to Check Token Revocation
**File**: Supabase Dashboard / SQL Editor

Add revocation checks to RLS policies:

```sql
-- Function to check if token is valid (not revoked)
CREATE OR REPLACE FUNCTION is_token_valid()
RETURNS BOOLEAN AS $$
BEGIN
  RETURN NOT EXISTS (
    SELECT 1 FROM token_issuances
    WHERE token_id = (auth.jwt() ->> 'token_id')
    AND revoked = TRUE
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Update miner read policy to check revocation
DROP POLICY IF EXISTS "miner_read_old_leads" ON leads;
CREATE POLICY "miner_read_old_leads" ON leads
FOR SELECT
TO authenticated
USING (
  (auth.jwt() ->> 'app_role') = 'miner'
  AND validated_at < NOW() - INTERVAL '72 minutes'
  AND is_token_valid()
);

-- Update validator policy to check revocation
DROP POLICY IF EXISTS "validator_write_access_leads" ON leads;
CREATE POLICY "validator_write_access_leads" ON leads
FOR INSERT
TO authenticated
WITH CHECK (
  (auth.jwt() ->> 'app_role') = 'validator'
  AND is_token_valid()
);

-- Repeat for all other tables...
```

### Task 2.3: Create Automated JWT Issuer Edge Function
**File**: Supabase Dashboard > Edge Functions > New Function: `issue-jwt`

This Edge Function automatically issues JWT tokens when miners/validators request them, with rate limiting and optional approval flow.

**Function code** (TypeScript):
```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { create, verify } from 'https://deno.land/x/djwt@v2.8/mod.ts'

const SUBNET_ID = 401
const NETWORK = "test"
const JWT_TTL_HOURS = 24 // Token expires in 24 hours
const REQUIRE_APPROVAL = Deno.env.get('REQUIRE_APPROVAL') === 'true'
const RATE_LIMIT_MAX = 10 // Max 10 requests per hour
const RATE_LIMIT_WINDOW = 3600 // 1 hour in seconds

// Helper: Verify hotkey is registered on Bittensor subnet and get role
async function verifyHotkeyAndGetRole(hotkey: string): Promise<{valid: boolean, role?: string}> {
  try {
    // Query Bittensor subtensor directly
    const response = await fetch(`https://test.finney.opentensor.ai/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'subnetwork_info',
        params: [SUBNET_ID],
        id: 1
      })
    })
    
    const data = await response.json()
    const metagraph = data.result
    
    // Find the UID for this hotkey
    const uid = metagraph.hotkeys.indexOf(hotkey)
    if (uid === -1) {
      return { valid: false }
    }
    
    // Check if they have validator permit
    const isValidator = metagraph.validator_permit[uid] === true
    
    return {
      valid: true,
      role: isValidator ? 'validator' : 'miner'
    }
    
  } catch (error) {
    console.error('Error verifying hotkey:', error)
    return { valid: false }
  }
}

serve(async (req) => {
  try {
    // Only allow POST requests
    if (req.method !== 'POST') {
      return new Response(
        JSON.stringify({ error: 'Method not allowed' }),
        { status: 405, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    const { hotkey } = await req.json()
    
    if (!hotkey || typeof hotkey !== 'string') {
      return new Response(
        JSON.stringify({ error: 'Invalid hotkey parameter' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Initialize Supabase client with service role
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, serviceRoleKey)
    
    // Check rate limit
    const { data: recentAttempts } = await supabase
      .from('token_issuance_attempts')
      .select('*')
      .eq('hotkey', hotkey)
      .gte('attempted_at', new Date(Date.now() - RATE_LIMIT_WINDOW * 1000).toISOString())

    if (recentAttempts && recentAttempts.length >= RATE_LIMIT_MAX) {
      return new Response(
        JSON.stringify({ 
          error: 'Rate limit exceeded',
          retry_after: RATE_LIMIT_WINDOW
        }),
        { status: 429, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Log attempt
    await supabase.from('token_issuance_attempts').insert({
      hotkey: hotkey,
      attempted_at: new Date().toISOString(),
      success: false
    })
    
    // Verify hotkey is registered on subnet and determine role
    const verification = await verifyHotkeyAndGetRole(hotkey)
    
    if (!verification.valid) {
      await supabase
        .from('token_issuance_attempts')
        .update({ error_message: 'Hotkey not registered on subnet' })
        .eq('hotkey', hotkey)
        .order('attempted_at', { ascending: false })
        .limit(1)
      
      return new Response(
        JSON.stringify({ 
          error: 'Hotkey not registered on subnet 401',
          subnet_id: SUBNET_ID,
          network: NETWORK
        }),
        { status: 403, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    const role = verification.role!
    
    // Check if approval is required
    if (REQUIRE_APPROVAL) {
      const { data: member } = await supabase
        .from('members')
        .select('approved')
        .eq('hotkey', hotkey)
        .single()
      
      if (!member || !member.approved) {
        await supabase
          .from('token_issuance_attempts')
          .update({ error_message: 'Hotkey not approved' })
          .eq('hotkey', hotkey)
          .order('attempted_at', { ascending: false })
          .limit(1)
        
        return new Response(
          JSON.stringify({ 
            error: 'Hotkey not approved. Please contact subnet operator for approval.',
            hotkey: hotkey
          }),
          { status: 403, headers: { 'Content-Type': 'application/json' } }
        )
      }
    }
    
    // Check if member already exists, update last_verified
    const { data: existingMember } = await supabase
      .from('members')
      .select('*')
      .eq('hotkey', hotkey)
      .single()
    
    if (existingMember) {
      // Update last verified timestamp
      await supabase
        .from('members')
        .update({ last_verified: new Date().toISOString(), role: role })
        .eq('hotkey', hotkey)
    } else {
      // Insert new member
      await supabase
        .from('members')
        .insert({
          hotkey: hotkey,
          role: role,
          approved: true,
          last_verified: new Date().toISOString()
        })
    }
    
    // Generate unique token ID for revocation tracking
    const tokenId = crypto.randomUUID()
    
    // Create JWT with custom claims
    const now = Math.floor(Date.now() / 1000)
    const exp = now + (JWT_TTL_HOURS * 3600)
    
    const payload = {
      iss: 'leadpoet-subnet',
      sub: hotkey,
      aud: 'authenticated',
      exp: exp,
      iat: now,
      role: 'authenticated', // Supabase role
      app_role: role, // Custom claim for RLS
      token_id: tokenId, // For revocation
      hotkey: hotkey,
    }
    
    // Sign JWT with Supabase JWT secret
    const jwtSecret = Deno.env.get('SUPABASE_JWT_SECRET')!
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(jwtSecret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    )
    
    const jwt = await create({ alg: 'HS256', typ: 'JWT' }, payload, key)
    
    // Record token issuance for audit and revocation
    await supabase
      .from('token_issuances')
      .insert({
        token_id: tokenId,
        hotkey: hotkey,
        role: role,
        expires_at: new Date(exp * 1000).toISOString()
      })
    
    // Update attempt record as successful
    await supabase
      .from('token_issuance_attempts')
      .update({ success: true })
      .eq('hotkey', hotkey)
      .order('attempted_at', { ascending: false })
      .limit(1)
    
    // Return JWT to client
    return new Response(
      JSON.stringify({ 
        success: true,
        jwt: jwt,
        role: role,
        expires_in: JWT_TTL_HOURS * 3600,
        expires_at: new Date(exp * 1000).toISOString(),
        instructions: 'Add this token to your .env file as SUPABASE_JWT=<token>'
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )
    
  } catch (error) {
    console.error('Error issuing JWT:', error)
    return new Response(
      JSON.stringify({ error: 'Internal server error', details: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
})
```

**Deploy the Edge Function**:
```bash
# Install Supabase CLI
npm install -g supabase

# Login to Supabase
supabase login

# Deploy the function
supabase functions deploy issue-jwt --project-ref qplwoislplkcegvdmbim

# Set environment variables
supabase secrets set SUPABASE_JWT_SECRET=<your_jwt_secret> --project-ref qplwoislplkcegvdmbim
supabase secrets set REQUIRE_APPROVAL=false --project-ref qplwoislplkcegvdmbim
```

**Note**: Set `REQUIRE_APPROVAL=true` if you want to manually approve members before they can get tokens.

### Task 2.4: Create Token Revocation Edge Function (Optional but Recommended)
**File**: Supabase Dashboard > Edge Functions > New Function: `revoke-jwt`

Admin-only function to revoke tokens:

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  try {
    // Require admin authentication (implement your admin auth here)
    const adminKey = req.headers.get('X-Admin-Key')
    if (adminKey !== Deno.env.get('ADMIN_KEY')) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    const { token_id, reason } = await req.json()
    
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, serviceRoleKey)
    
    // Revoke the token
    const { error } = await supabase
      .from('token_issuances')
      .update({
        revoked: true,
        revoked_at: new Date().toISOString(),
        revoked_reason: reason
      })
      .eq('token_id', token_id)
    
    if (error) throw error
    
    return new Response(
      JSON.stringify({ success: true, message: 'Token revoked' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
})
```

**Deploy the revocation function**:
```bash
supabase functions deploy revoke-jwt --project-ref qplwoislplkcegvdmbim
supabase secrets set ADMIN_KEY=<your_secure_admin_key> --project-ref qplwoislplkcegvdmbim
```

### Task 2.5: Update Setup Instructions for Miners/Validators
**File**: `README.md`

Add clear instructions:

```markdown
## Setup for Miners/Validators

### Prerequisites
1. Register on Bittensor subnet 401
2. Ensure your hotkey is registered and serving

### Getting Your JWT Token

Request your JWT token by calling the issuance endpoint:

```bash
curl -X POST https://qplwoislplkcegvdmbim.supabase.co/functions/v1/issue-jwt \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "YOUR_HOTKEY_HERE"}'
```

**Response**:
```json
{
  "success": true,
  "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "role": "miner",
  "expires_in": 86400,
  "expires_at": "2025-10-10T20:00:00.000Z",
  "instructions": "Add this token to your .env file as SUPABASE_JWT=<token>"
}
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Bittensor-subnet.git
   cd Bittensor-subnet
   ```

2. **Create `.env` file** with your JWT token:
   ```bash
   echo "SUPABASE_JWT=<your_jwt_token_from_above>" > .env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run your node**:
   ```bash
   # For miners:
   python neurons/miner.py --wallet_name miner --wallet_hotkey default --netuid 401 --subtensor_network test
   
   # For validators:
   python neurons/validator.py --wallet_name validator --wallet_hotkey default --netuid 401 --subtensor_network test
   ```

That's it! Your node will automatically have the correct database permissions based on your role.

### Token Expiration & Renewal

Tokens expire after 24 hours. To renew:
```bash
curl -X POST https://qplwoislplkcegvdmbim.supabase.co/functions/v1/issue-jwt \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "YOUR_HOTKEY_HERE"}'
```

Update your `.env` file with the new token and restart your node.

### Troubleshooting

**"Hotkey not registered" error**:
- Verify your hotkey is registered on subnet 401: `btcli s metagraph --netuid 401 --subtensor.network test`
- Ensure you're using the correct network (test vs mainnet)

**"403 Forbidden" when reading/writing**:
- Verify token is not expired: tokens last 24 hours
- Check your role matches the operation (miners=read, validators=write)
- Ensure token hasn't been revoked

**"Rate limit exceeded" error**:
- You can only request a token 10 times per hour
- Wait for the cooldown period and try again

**Token renewal failing**:
- Check network connectivity to Supabase
- Verify hotkey is still registered on subnet
- Contact subnet operator if persistent issues
```

### Task 2.6: Set Up Automatic Token Renewal (Optional)
**File**: Client code (`neurons/miner.py` and `neurons/validator.py`)

Add automatic token refresh logic to prevent service interruption:

```python
import os
import time
import requests
import jwt as pyjwt
from datetime import datetime, timedelta

class TokenManager:
    def __init__(self, hotkey, token_endpoint):
        self.hotkey = hotkey
        self.token_endpoint = token_endpoint
        self.jwt = os.getenv("SUPABASE_JWT")
        self.token_expires = None
        self._parse_expiry()
    
    def _parse_expiry(self):
        """Decode JWT to get expiry timestamp"""
        try:
            decoded = pyjwt.decode(self.jwt, options={"verify_signature": False})
            self.token_expires = datetime.fromtimestamp(decoded['exp'])
        except Exception as e:
            print(f"Error parsing token expiry: {e}")
            self.token_expires = datetime.now()
    
    def refresh_if_needed(self):
        """Refresh token 1 hour before expiry"""
        if datetime.now() >= self.token_expires - timedelta(hours=1):
            print(f"Token expiring soon, refreshing...")
            try:
                response = requests.post(
                    self.token_endpoint,
                    json={'hotkey': self.hotkey},
                    timeout=10
                )
                
                if response.status_code == 200:
                    new_token = response.json()['jwt']
                    
                    # Update .env file
                    self._update_env_file(new_token)
                    
                    # Update in-memory token
                    self.jwt = new_token
                    self._parse_expiry()
                    
                    print(f"Token refreshed successfully. New expiry: {self.token_expires}")
                    return True
                else:
                    print(f"Token refresh failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"Error refreshing token: {e}")
                return False
        return False
    
    def _update_env_file(self, new_token):
        """Update .env file with new token"""
        env_path = '.env'
        
        # Read existing .env
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Update or add SUPABASE_JWT
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('SUPABASE_JWT='):
                lines[i] = f'SUPABASE_JWT={new_token}\n'
                updated = True
                break
        
        if not updated:
            lines.append(f'SUPABASE_JWT={new_token}\n')
        
        # Write back to .env
        with open(env_path, 'w') as f:
            f.writelines(lines)

# Usage in miner.py or validator.py
token_manager = TokenManager(
    hotkey="YOUR_HOTKEY",
    token_endpoint="https://qplwoislplkcegvdmbim.supabase.co/functions/v1/issue-jwt"
)

# In your main loop
while True:
    # Check and refresh token if needed
    token_manager.refresh_if_needed()
    
    # Your normal mining/validation logic
    # ...
    
    time.sleep(60)  # Check every minute
```

---

## Phase 3: Code Migration

### Task 3.1: Update Miner Code to Use Supabase
**File**: `neurons/miner.py`

Replace Firestore client with Supabase:

```python
from supabase import create_client, Client
import os

# Initialize Supabase client
SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")  # From .env file
supabase: Client = create_client(SUPABASE_URL, SUPABASE_JWT)

# Read leads (only >72 minutes old due to RLS)
def get_old_leads():
    response = supabase.table("leads").select("*").execute()
    return response.data
```

### Task 3.2: Update Validator Code to Use Supabase
**File**: `neurons/validator.py`

Replace Firestore client with Supabase:

```python
from supabase import create_client, Client
import os

# Initialize Supabase client
SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")  # From .env file
supabase: Client = create_client(SUPABASE_URL, SUPABASE_JWT)

# Write validated leads
def save_validated_lead(lead_data):
    response = supabase.table("leads").insert(lead_data).execute()
    return response.data
```

---

## Phase 4: Testing & Deployment

### Task 4.1: Test Miner Access
Verify miners can only read old leads:
```python
# Should work: Reading leads >72 minutes old
old_leads = supabase.table("leads").select("*").execute()

# Should fail: Writing leads
supabase.table("leads").insert({"email": "test@test.com"}).execute()  # 403 Forbidden
```

### Task 4.2: Test Validator Access
Verify validators can write:
```python
# Should work: Writing validated leads
supabase.table("leads").insert({
    "email": "test@test.com",
    "validator_hotkey": "5ABC..."
}).execute()
```

### Task 4.3: Test Token Revocation
Revoke a token and verify access is denied:
```bash
curl -X POST https://qplwoislplkcegvdmbim.supabase.co/functions/v1/revoke-jwt \
  -H "X-Admin-Key: YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"token_id": "TOKEN_ID", "reason": "Testing revocation"}'
```

### Task 4.4: Deploy to Production
1. Update all environment variables in production
2. Deploy updated code to miners and validators
3. Monitor logs for any access issues
4. Set up token renewal reminders (24-hour expiry)
---

## Notes
- Validators have WRITE-only access to prevent gaming the system
- Miners have READ-only access to leads older than 72 minutes
- Tokens expire after 24 hours. Use the TokenManager class (Task 2.6) in your node code for automatic renewal, or manually request a new token before expiry
- All access is enforced at the database level via RLS policies
- Token revocation is instant and enforced by RLS policies