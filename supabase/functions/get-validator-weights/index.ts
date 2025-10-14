import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get JWT from Authorization header
    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: 'No authorization header provided' }), 
        { 
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    const token = authHeader.replace('Bearer ', '')
    
    // Decode JWT to get validator hotkey (parse the base64 payload)
    let decoded: any
    try {
      // JWT format is header.payload.signature
      const parts = token.split('.')
      if (parts.length !== 3) {
        throw new Error('Invalid JWT format')
      }
      
      // Decode the payload (second part)
      const payload = parts[1]
      // Add padding if necessary for base64 decoding
      const paddedPayload = payload + '=='.slice((2 - payload.length * 3) & 3)
      const decodedPayload = atob(paddedPayload.replace(/-/g, '+').replace(/_/g, '/'))
      decoded = JSON.parse(decodedPayload)
      
      console.log('Decoded JWT claims:', JSON.stringify(decoded))
    } catch (error) {
      console.error('Error decoding JWT:', error)
      return new Response(
        JSON.stringify({ error: 'Invalid token format', details: error.message }), 
        { 
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Extract validator hotkey and role from JWT claims
    const validatorHotkey = decoded.hotkey
    const appRole = decoded.app_role
    
    if (!validatorHotkey) {
      return new Response(
        JSON.stringify({ error: 'No hotkey in token claims' }), 
        { 
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Verify this is a validator token
    if (appRole !== 'validator') {
      return new Response(
        JSON.stringify({ error: 'This endpoint is only accessible to validators' }), 
        { 
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Create Supabase client with service role for full access
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, serviceRoleKey)

    // Verify validator is registered in members table
    const { data: memberCheck, error: memberError } = await supabase
      .from('members')
      .select('hotkey')
      .eq('hotkey', validatorHotkey)
      .eq('role', 'validator')
      .single()

    if (memberError || !memberCheck) {
      return new Response(
        JSON.stringify({ error: 'Validator not registered in members table' }), 
        { 
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Calculate time 72 minutes ago (1 epoch)
    const epochStartTime = new Date(Date.now() - 72 * 60 * 1000).toISOString()
    
    console.log(`Checking eligibility for validator: ${validatorHotkey.substring(0, 10)}...`)
    console.log(`Epoch start time (72 min ago): ${epochStartTime}`)

    // ===== STEP 1: CHECK VALIDATOR ELIGIBILITY (10% threshold) =====
    
    // Get all validations by this validator in the last 72 minutes
    const { data: validations, error: valError } = await supabase
      .from('validation_tracking')
      .select('prospect_id, is_valid')
      .eq('validator_hotkey', validatorHotkey)
      .gte('created_at', epochStartTime)

    if (valError) {
      console.error('Error fetching validations:', valError)
      return new Response(
        JSON.stringify({ error: 'Failed to fetch validation data' }), 
        { 
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    if (!validations || validations.length === 0) {
      return new Response(
        JSON.stringify({ 
          error: 'No validations found in current epoch',
          eligible: false,
          validated_count: 0,
          total_count: 0,
          percentage: 0,
          weights: {}
        }), 
        { 
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Get unique prospect IDs this validator validated
    const prospectIds = [...new Set(validations.map(v => v.prospect_id))]
    
    console.log(`Validator submitted ${validations.length} validations for ${prospectIds.length} unique prospects`)

    // Get ALL consensus results for the epoch (not just ones this validator participated in)
    // We need to count all prospects that reached consensus, even if this validator didn't participate
    const { data: consensusResults, error: consError } = await supabase
      .from('consensus_results')
      .select('prospect_id, total_validations, valid_count, invalid_count')

    if (consError) {
      console.error('Error fetching consensus results:', consError)
      return new Response(
        JSON.stringify({ error: 'Failed to fetch consensus results' }), 
        { 
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Calculate consensus participation
    let consensusCount = 0
    let totalWithConsensus = 0
    
    // Create a map of prospect_id to validator's vote
    const validatorVotes: { [key: string]: boolean } = {}
    for (const val of validations) {
      validatorVotes[val.prospect_id] = val.is_valid
    }

    // Check each prospect that reached consensus
    for (const result of consensusResults || []) {
      // Count prospects where consensus was reached (2+ or 3+ validators)
      // With our new logic:
      // - 3+ validators: Always process consensus if 2+ agree
      // - 2 validators: Process consensus if both agree (2/2)
      // - 1 validator: No consensus possible
      
      if (result.total_validations >= 3) {
        // Standard case: 3+ validators participated
        totalWithConsensus++
        
        // Consensus is reached if 2+ validators agree (either valid or invalid)
        const consensusAccepted = result.valid_count >= 2
        const validatorVote = validatorVotes[result.prospect_id]
        
        // Check if validator's vote aligned with consensus
        if (validatorVote !== undefined && validatorVote === consensusAccepted) {
          consensusCount++
        }
      } else if (result.total_validations === 2) {
        // 2-validator case: consensus only if both agree
        // Check if there was unanimous agreement (2/2 valid or 0/2 valid)
        const unanimous = result.valid_count === 2 || result.valid_count === 0
        
        if (unanimous) {
          totalWithConsensus++
          
          // Consensus decision is based on whether both voted valid or invalid
          const consensusAccepted = result.valid_count === 2
          const validatorVote = validatorVotes[result.prospect_id]
          
          // Check if this validator was one of the two and agreed
          if (validatorVote !== undefined && validatorVote === consensusAccepted) {
            consensusCount++
          }
        }
        // If 2 validators disagreed (1 valid, 1 invalid), no consensus - don't count
      }
      // If only 1 validator, no consensus possible - don't count
    }

    // Calculate percentage participation in consensus
    const percentage = totalWithConsensus > 0 
      ? (consensusCount / totalWithConsensus) * 100 
      : 0

    console.log(`Consensus participation: ${consensusCount}/${totalWithConsensus} (${percentage.toFixed(1)}%)`)

    // ===== ENFORCE 10% THRESHOLD SERVER-SIDE =====
    const eligible = percentage >= 10.0

    if (!eligible) {
      return new Response(
        JSON.stringify({ 
          error: `Not eligible - only ${percentage.toFixed(1)}% consensus participation (need ≥10%)`,
          eligible: false,
          validated_count: consensusCount,
          total_count: totalWithConsensus,
          percentage: percentage,
          weights: {}
        }), 
        { 
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    console.log(`✅ Validator eligible with ${percentage.toFixed(1)}% consensus participation`)

    // ===== STEP 2: CALCULATE MINER WEIGHTS (100% sourcing-based) =====
    
    // Get all leads added to the leads table in the last 72 minutes
    const { data: acceptedLeads, error: leadsError } = await supabase
      .from('leads')
      .select('miner_hotkey')
      .gte('validated_at', epochStartTime)

    if (leadsError) {
      console.error('Error fetching accepted leads:', leadsError)
      return new Response(
        JSON.stringify({ error: 'Failed to fetch accepted leads' }), 
        { 
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    if (!acceptedLeads || acceptedLeads.length === 0) {
      // No leads in epoch, but validator is still eligible
      return new Response(
        JSON.stringify({
          eligible: true,
          validated_count: consensusCount,
          total_count: totalWithConsensus,
          percentage: percentage,
          weights: {},
          total_leads: 0,
          message: 'No leads found in current epoch'
        }),
        { 
          status: 200,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Count leads per miner
    const minerCounts: { [key: string]: number } = {}
    for (const lead of acceptedLeads) {
      if (lead.miner_hotkey) {
        minerCounts[lead.miner_hotkey] = (minerCounts[lead.miner_hotkey] || 0) + 1
      }
    }

    const totalLeads = acceptedLeads.length
    const uniqueMiners = Object.keys(minerCounts).length

    console.log(`Found ${totalLeads} consensus-accepted leads from ${uniqueMiners} miners`)

    // Calculate proportional weights (100% sourcing-based)
    const weights: { [key: string]: number } = {}
    
    for (const [miner, count] of Object.entries(minerCounts)) {
      // Weight = (miner's leads / total leads)
      // Normalized to sum to 1.0 for Bittensor
      weights[miner] = count / totalLeads
      console.log(`Miner ${miner.substring(0, 10)}...: ${count} leads (${(weights[miner] * 100).toFixed(1)}%)`)
    }

    // Sort miners by weight for display
    const sortedMiners = Object.entries(weights)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5) // Top 5 for logging

    // Create response
    const response = {
      eligible: true,
      validated_count: consensusCount,
      total_count: totalWithConsensus,
      percentage: percentage,
      weights: weights,
      total_leads: totalLeads,
      unique_miners: uniqueMiners,
      top_miners: sortedMiners.map(([hotkey, weight]) => ({
        hotkey: hotkey.substring(0, 10) + '...',
        weight: weight,
        percentage: (weight * 100).toFixed(1)
      })),
      message: `Successfully calculated weights for ${uniqueMiners} miners based on ${totalLeads} leads`
    }

    console.log('✅ Successfully calculated miner weights')

    return new Response(
      JSON.stringify(response),
      { 
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    )

  } catch (error) {
    console.error('Unexpected error:', error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        details: error.message 
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    )
  }
})
