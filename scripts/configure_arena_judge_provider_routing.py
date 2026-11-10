#!/usr/bin/env python3
"""Check or apply one owner-audited Arena judge transport correction.

The request JSON contains round_id, expected_configuration, routing, request_id
and actor_ref. Without --connection-file the command only prepares a bounded
summary. A private connection JSON uses ordinary psycopg2 connection arguments;
the operator supplies an existing verified connection, never an AWS login.
Deploy and verify transport support before activation. This command changes no
runtime, scoring rule, model, lease, checkpoint, cost or result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True

RPC = 'public.lab_arena_set_judge_provider_routing_v1'
APPLY_ENV = 'LEADPOET_ARENA_JUDGE_ROUTING_APPLY'
REQUEST_FIELDS = {'round_id', 'expected_configuration', 'routing', 'request_id', 'actor_ref'}
CONNECTION_FIELDS = {'host', 'hostaddr', 'port', 'dbname', 'user', 'password',
                     'sslmode', 'sslrootcert', 'sslcert', 'sslkey'}


def read_json(path: Path, *, private: bool = False) -> dict:
    if private and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('connection file must be private')
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('JSON file exceeds the input bound')
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def request_arguments(request: dict) -> dict:
    if set(request) != REQUEST_FIELDS:
        raise ValueError('request fields differ')
    config = request['expected_configuration']
    routing = request['routing']
    if (not isinstance(config, dict) or config.get('round_id') != request['round_id']
            or not isinstance(routing, dict) or not routing):
        raise ValueError('round or routing differs')
    models = (config.get('scorer_policy') or {}).get('judge_models') or {}
    for model, plan in routing.items():
        if (model not in models.values() or not isinstance(plan, list)
                or not 1 <= len(plan) <= 2
                or any(not isinstance(route, str) or not route for route in plan)
                or len(set(plan)) != len(plan)):
            raise ValueError('routing plan invalid')
    if (not isinstance(request['request_id'], str) or not 16 <= len(request['request_id']) <= 128
            or not isinstance(request['actor_ref'], str) or not 1 <= len(request['actor_ref']) <= 200):
        raise ValueError('request identity invalid')
    return {'p_' + key: value for key, value in request.items()
            if key not in {'routing'}} | {'p_routing': routing}


def execute(connection, arguments: dict, *, apply: bool) -> dict:
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout='30s';SET LOCAL lock_timeout='5s'")
        cursor.execute('SELECT ' + RPC + '(%s,%s::jsonb,%s::jsonb,%s,%s,%s)', (
            arguments['p_round_id'], json.dumps(arguments['p_expected_configuration']),
            json.dumps(arguments['p_routing']), arguments['p_request_id'],
            arguments['p_actor_ref'], not apply))
        result = cursor.fetchone()[0]
    if (not isinstance(result, dict)
            or result.get('status') not in ({'applied', 'existing'} if apply else {'dry_run', 'existing'})
            or result.get('request_id') != arguments['p_request_id']
            or result.get('judge_provider_routing') != arguments['p_routing']):
        raise ValueError('routing correction readback differs')
    if apply:
        connection.commit()
    else:
        connection.rollback()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-file', type=Path, required=True)
    parser.add_argument('--connection-file', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    arguments = request_arguments(read_json(args.request_file))
    if args.apply and (os.environ.get(APPLY_ENV) != '1' or args.connection_file is None):
        raise ValueError('apply requires the operator environment and private connection file')
    if args.connection_file is None:
        print(json.dumps({'action': 'set_judge_provider_routing', 'writes': False,
            'round_id': arguments['p_round_id'], 'request_id': arguments['p_request_id'],
            'judge_provider_routing': arguments['p_routing']}, sort_keys=True))
        return 0
    connection_args = read_json(args.connection_file, private=True)
    if (not set(connection_args) <= CONNECTION_FIELDS
            or connection_args.get('sslmode') != 'verify-full'
            or not connection_args.get('sslrootcert')
            or not connection_args.get('host') or not connection_args.get('dbname')):
        raise ValueError('verified TLS connection required')
    import psycopg2
    connection = None
    try:
        connection = psycopg2.connect(**connection_args, connect_timeout=10,
                                     application_name='arena_judge_provider_routing_operator')
        result = execute(connection, arguments, apply=args.apply)
    finally:
        connection_args.clear()
        if connection is not None:
            connection.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Database exceptions can carry DSNs and private frozen configuration.
        raise SystemExit('judge provider routing failed: ' + type(error).__name__) from None
