import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { create } from 'https://deno.land/x/djwt@v2.8/mod.ts'
import { decodeAddress, signatureVerify } from 'https://esm.sh/@polkadot/util-crypto@12.6.2'
import { u8aToHex } from 'https://esm.sh/@polkadot/util@12.6.2'

// Configuration constants
const SUBNET_ID = 401
const NETWORK = "test"
const JWT_TTL_HOURS = 24
const REQUIRE_APPROVAL = Deno.env.get('REQUIRE_APPROVAL') === 'true'
const RATE_LIMIT_MAX = 100  // Increased for initial setup
const RATE_LIMIT_WINDOW = 3600
const SIGNATURE_VALIDITY_WINDOW = 300 // 5 minutes

/**
 * Verify SR25519 signature from Bittensor wallet
 * @param hotkey - SS58 address (public key)
 * @param message - Original message that was signed
 * @param signature - Hex-encoded signature
 * @returns boolean indicating if signature is valid
 */
async function verifySignature(hotkey: string, message: string, signature: string): Promise<boolean> {
  try {
    console.log(`[verifySignature] Verifying signature for hotkey: ${hotkey}`)
    
    // Decode SS58 address to public key bytes
    const publicKey = decodeAddress(hotkey)
    
    // Convert message to bytes
    const messageBytes = new TextEncoder().encode(message)
    
    // Convert hex signature to bytes
    const signatureBytes = new Uint8Array(
      signature.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16))
    )
    
    // Verify signature
    const result = signatureVerify(messageBytes, signatureBytes, publicKey)
    
    console.log(`[verifySignature] Signature valid: ${result.isValid}`)
    return result.isValid
  } catch (error) {
    console.error(`[verifySignature] Error:`, error)
    return false
  }
}

/**
 * Verify hotkey is registered on Bittensor subnet and determine role
 * Uses TaoStats API to query the metagraph
 */
async function verifyHotkeyAndGetRole(hotkey: string, supabase: any): Promise<{valid: boolean, role?: string, uid?: number}> {
  try {
    console.log(`[verifyHotkey] Checking hotkey: ${hotkey} on subnet ${SUBNET_ID}`)
    
    // Try TaoStats API first (mainnet)
    try {
      const taostatsResponse = await fetch(
        `https://api.taostats.io/api/v1/subnet/metagraph/${SUBNET_ID}`,
        { 
          headers: { 'Accept': 'application/json' },
          signal: AbortSignal.timeout(5000) // 5 second timeout
        }
      )
      
      if (taostatsResponse.ok) {
        const data = await taostatsResponse.json()
        const neurons = data.neurons || []
        const neuron = neurons.find((n: any) => n.hotkey === hotkey)
        
        if (neuron) {
          const uid = neuron.uid
          const hasValidatorPermit = neuron.validator_permit === true
          const role = hasValidatorPermit ? 'validator' : 'miner'
          
          console.log(`[verifyHotkey] ✅ Found in TaoStats - UID ${uid}, role: ${role}`)
          return { valid: true, role, uid }
        }
      }
    } catch (taostatsError) {
      console.log(`[verifyHotkey] TaoStats failed (likely testnet):`, taostatsError)
    }
    
     // Fallback: Query testnet metagraph from Supabase cache
    // The metagraph is periodically synced by the Python validators
    console.log(`[verifyHotkey] Querying Supabase metagraph cache for testnet...`)
    console.log(`[verifyHotkey] Looking for hotkey: ${hotkey}`)
    console.log(`[verifyHotkey] Looking for netuid: ${SUBNET_ID}`)
    
    try {
      // First, check if there are ANY records in the cache
      const { data: allRecords, error: countError } = await supabase
        .from('metagraph_cache')
        .select('hotkey, netuid, uid, validator_permit, active')
        .limit(10)
      
      console.log(`[verifyHotkey] Sample cache records (first 10):`, JSON.stringify(allRecords, null, 2))
      if (countError) {
        console.error(`[verifyHotkey] Error reading cache:`, countError)
      }
      
      // Now query for the specific hotkey
      const { data: cachedNeuron, error } = await supabase
        .from('metagraph_cache')
        .select('uid, hotkey, validator_permit, active')
        .eq('hotkey', hotkey)
        .eq('netuid', SUBNET_ID)
        .single()
      
      console.log(`[verifyHotkey] Query result - data:`, cachedNeuron, `error:`, error)
      
      if (cachedNeuron && !error) {
        // CRITICAL: Validator must have BOTH validator_permit AND be actively validating
        // Many neurons have validator_permit=true but are not actively validating
        // Only active validators should get 'validator' role for write access to main DB
        const isValidator = cachedNeuron.validator_permit === true && cachedNeuron.active === true
        const role = isValidator ? 'validator' : 'miner'
        
        console.log(`[verifyHotkey] ✅ Found in metagraph cache - UID ${cachedNeuron.uid}`)
        console.log(`[verifyHotkey]    validator_permit: ${cachedNeuron.validator_permit}, active: ${cachedNeuron.active}`)
        console.log(`[verifyHotkey]    Determined role: ${role}`)
        
        return { valid: true, role, uid: cachedNeuron.uid }
      }
      
      console.log(`[verifyHotkey] Metagraph cache miss or error:`, error)
    } catch (cacheError) {
      console.error(`[verifyHotkey] Cache query failed:`, cacheError)
    }
    
    console.log(`[verifyHotkey] Hotkey ${hotkey.slice(0, 10)}... not found in any source`)
    return { valid: false }
    
  } catch (error) {
    console.error(`[verifyHotkey] Error:`, error)
    return { valid: false }
  }
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  const requestId = crypto.randomUUID().slice(0, 8)
  console.log(`[${requestId}] Incoming request: ${req.method} ${req.url}`)
  
  try {
    if (req.method !== 'POST') {
      return new Response(
        JSON.stringify({ error: 'Method not allowed' }),
        { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    // Parse request body
    const body = await req.json()
    const { hotkey, message, signature, timestamp } = body
    
    // Validate required fields
    if (!hotkey || !message || !signature || !timestamp) {
      console.log(`[${requestId}] Missing required fields`)
      return new Response(
        JSON.stringify({ error: 'Missing required fields: hotkey, message, signature, timestamp' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    // Verify timestamp freshness (prevent replay attacks)
    const now = Math.floor(Date.now() / 1000)
    if (Math.abs(now - timestamp) > SIGNATURE_VALIDITY_WINDOW) {
      console.log(`[${requestId}] Request expired: timestamp ${timestamp}, now ${now}`)
      return new Response(
        JSON.stringify({ error: 'Request expired. Please generate a new signature.' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    // Verify message format
    const expectedMessage = `leadpoet-jwt-request:${timestamp}`
    if (message !== expectedMessage) {
      console.log(`[${requestId}] Invalid message format`)
      return new Response(
        JSON.stringify({ error: 'Invalid message format' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    // Verify signature proves ownership of private key
    console.log(`[${requestId}] About to verify signature for hotkey: ${hotkey}`)
    const signatureValid = await verifySignature(hotkey, message, signature)
    if (!signatureValid) {
      console.log(`[${requestId}] ❌ Invalid signature`)
      return new Response(
        JSON.stringify({ error: 'Invalid signature. You must sign with the private key corresponding to the hotkey.' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    console.log(`[${requestId}] ✅ Signature verified - requester controls hotkey: ${hotkey}`)
    
    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, serviceRoleKey)
    
    // Check rate limit
    const rateLimitStart = new Date(Date.now() - RATE_LIMIT_WINDOW * 1000).toISOString()
    const { data: recentAttempts, error: rateLimitError } = await supabase
      .from('token_issuance_attempts')
      .select('*')
      .eq('hotkey', hotkey)
      .gte('attempted_at', rateLimitStart)

    if (rateLimitError) {
      console.error(`[${requestId}] Rate limit check error:`, rateLimitError)
    }

    if (recentAttempts && recentAttempts.length >= RATE_LIMIT_MAX) {
      console.log(`[${requestId}] Rate limit exceeded: ${recentAttempts.length} attempts`)
      return new Response(
        JSON.stringify({ 
          error: 'Rate limit exceeded',
          message: `Maximum ${RATE_LIMIT_MAX} requests per hour. Try again later.`
        }),
        { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Log attempt
    await supabase.from('token_issuance_attempts').insert({
      hotkey,
      success: false,
      error_message: 'Pending verification'
    })

    // Verify hotkey is registered on subnet using TaoStats
    console.log(`[${requestId}] ✅ Signature verified, checking metagraph...`)
    
    const verification = await verifyHotkeyAndGetRole(hotkey, supabase)
    
    if (!verification.valid) {
      console.log(`[${requestId}] Hotkey not registered on subnet`)
      await supabase.from('token_issuance_attempts').insert({
        hotkey,
        success: false,
        error_message: 'Hotkey not found in subnet metagraph'
      })
      return new Response(
        JSON.stringify({ error: 'Hotkey not registered on subnet 401' }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const { role, uid } = verification

    // ALLOW BOTH MINERS AND VALIDATORS
    console.log(`[${requestId}] ✅ Issuing token for ${role} (UID: ${uid})`)

    // Check approval requirement
    if (REQUIRE_APPROVAL) {
      const { data: member } = await supabase
        .from('members')
        .select('approved')
        .eq('hotkey', hotkey)
        .single()

      if (!member || !member.approved) {
        console.log(`[${requestId}] Hotkey not approved`)
        return new Response(
          JSON.stringify({ 
            error: 'Approval required',
            message: 'Your registration is pending approval. Contact the subnet operator.'
          }),
          { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }
    }

    // Upsert member record
    await supabase.from('members').upsert({
      hotkey,
      role,
      uid,
      approved: !REQUIRE_APPROVAL,
      last_seen: new Date().toISOString()
    }, { onConflict: 'hotkey' })

    // Generate JWT
    const tokenId = crypto.randomUUID()
    const issuedAt = Math.floor(Date.now() / 1000)
    const exp = issuedAt + (JWT_TTL_HOURS * 3600)

    const payload = {
      iss: 'supabase',  // CRITICAL: Must be 'supabase' (not full URL!)
      ref: 'qplwoislplkcegvdmbim',  // CRITICAL: Project reference for Postgrest
      sub: hotkey,
      aud: 'authenticated',  // CRITICAL: Must be 'authenticated' for RLS
      exp,
      iat: issuedAt,
      role: 'authenticated',  // CRITICAL: Must match RLS policies
      app_role: role,
      token_id: tokenId,
      hotkey,
      uid
    }

    // Get JWT secret and TRIM any whitespace (important!)
    const jwtSecret = Deno.env.get('JWT_SECRET')!.trim()
    
    console.log(`[${requestId}] JWT_SECRET length: ${jwtSecret.length} (after trim)`)
    console.log(`[${requestId}] JWT_SECRET first 20 chars: '${jwtSecret.substring(0, 20)}'`)
    console.log(`[${requestId}] JWT_SECRET last 20 chars: '${jwtSecret.substring(jwtSecret.length - 20)}'`)
    
    // Import key directly from the TRIMMED base64 string
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(jwtSecret),  // Use trimmed base64 string
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    )

    const jwt = await create(
      { alg: 'HS256', typ: 'JWT' },
      payload,
      key
    )

    // Record successful issuance
    await supabase.from('token_issuances').insert({
      token_id: tokenId,
      hotkey,
      role,
      uid,
      expires_at: new Date(exp * 1000).toISOString(),
      issued_by: 'edge-function'
    })

    // Update attempt as successful
    await supabase.from('token_issuance_attempts').insert({
      hotkey,
      success: true,
      error_message: null
    })

    console.log(`[${requestId}] ✅ JWT issued successfully for ${role} ${hotkey}`)

    return new Response(
      JSON.stringify({
        success: true,
        token: jwt,
        role,
        uid,
        expires_in_hours: JWT_TTL_HOURS,
        message: `Token issued for ${role} on subnet ${SUBNET_ID}`
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error(`[${requestId}] Error:`, error)
    return new Response(
      JSON.stringify({ error: 'Internal server error', details: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})


