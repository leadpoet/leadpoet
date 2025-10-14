import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

/**
 * Admin-only Edge Function to revoke JWT tokens
 * Requires X-Admin-Key header for authentication
 */
serve(async (req) => {
  const requestId = crypto.randomUUID().slice(0, 8)
  console.log(`[${requestId}] Revoke request: ${req.method} ${req.url}`)
  
  try {
    // Only allow POST requests
    if (req.method !== 'POST') {
      console.log(`[${requestId}] Method not allowed: ${req.method}`)
      return new Response(
        JSON.stringify({ error: 'Method not allowed' }),
        { status: 405, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Require admin authentication
    const adminKey = req.headers.get('X-Admin-Key')
    const expectedAdminKey = Deno.env.get('ADMIN_KEY')
    
    if (!adminKey || !expectedAdminKey) {
      console.log(`[${requestId}] Missing admin key`)
      return new Response(
        JSON.stringify({ 
          error: 'Unauthorized',
          message: 'Admin authentication required'
        }),
        { status: 401, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    if (adminKey !== expectedAdminKey) {
      console.log(`[${requestId}] Invalid admin key`)
      return new Response(
        JSON.stringify({ 
          error: 'Unauthorized',
          message: 'Invalid admin credentials'
        }),
        { status: 401, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    console.log(`[${requestId}] Admin authenticated successfully`)
    
    // Parse request body
    const body = await req.json()
    const { token_id, reason, revoked_by } = body
    
    if (!token_id) {
      console.log(`[${requestId}] Missing token_id parameter`)
      return new Response(
        JSON.stringify({ 
          error: 'Bad request',
          message: 'token_id is required'
        }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    console.log(`[${requestId}] Revoking token: ${token_id}`)
    
    // Initialize Supabase client with service role
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, serviceRoleKey)
    
    // Check if token exists
    const { data: existingToken, error: fetchError } = await supabase
      .from('token_issuances')
      .select('*')
      .eq('token_id', token_id)
      .single()
    
    if (fetchError || !existingToken) {
      console.log(`[${requestId}] Token not found: ${token_id}`)
      return new Response(
        JSON.stringify({ 
          error: 'Not found',
          message: 'Token ID not found in database',
          token_id: token_id
        }),
        { status: 404, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Check if already revoked
    if (existingToken.revoked) {
      console.log(`[${requestId}] Token already revoked`)
      return new Response(
        JSON.stringify({ 
          success: false,
          message: 'Token already revoked',
          token_id: token_id,
          revoked_at: existingToken.revoked_at,
          revoked_reason: existingToken.revoked_reason
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Revoke the token
    const { data, error } = await supabase
      .from('token_issuances')
      .update({
        revoked: true,
        revoked_at: new Date().toISOString(),
        revoked_reason: reason || 'No reason provided',
        revoked_by: revoked_by || 'admin'
      })
      .eq('token_id', token_id)
      .select()
    
    if (error) {
      console.error(`[${requestId}] Database error:`, error)
      throw error
    }
    
    console.log(`[${requestId}] Token revoked successfully`)
    
    return new Response(
      JSON.stringify({ 
        success: true,
        message: 'Token revoked successfully',
        token_id: token_id,
        hotkey: existingToken.hotkey,
        role: existingToken.role,
        revoked_at: data[0].revoked_at,
        revoked_reason: data[0].revoked_reason
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )
    
  } catch (error) {
    console.error(`[${requestId}] Error:`, error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        details: error.message 
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
})