-- Resume only the fifteen Sep21 Stage 1 judgments that had no accepted result
-- when the old scorer returned out-of-scope Intent Details coverage indexes.
-- Keep all accepted judgments and their original scorer-image identities.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $recover_sep21_unresolved_scoring$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_targets CONSTANT JSONB := $targets$[{"icp_position":8,"judgment_cache_key":"sha256:d02df836162f44b123863175538b0ae236964c07987f017eff261288a1974832","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"judgment_input_hash":"sha256:04680a219eda0591657685588f52f733caf5f7792c8808f1992a6972cf8cd50e","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:04680a219eda0591657685588f52f733caf5f7792c8808f1992a6972cf8cd50e","cache_key":"sha256:d02df836162f44b123863175538b0ae236964c07987f017eff261288a1974832"},"latest_attempt":2,"latest_run_id":"arena-2026-09-21:baseline-2026-09-21:1:8:score:2","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","old_judgment_cache_key":"sha256:a3f38b77334dd8234f2d8d346adf18dfcf0ae478a3a8d9c3ac86a4b147bc3424","scored_run_id":"arena-2026-09-21:baseline-2026-09-21:1:8:2","submission_id":"baseline-2026-09-21"},{"icp_position":9,"judgment_cache_key":"sha256:9c2bbad36f862c2ab04d159d89f04c1c253fe422b0efc6e9992216c357b3e0b9","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"judgment_input_hash":"sha256:05d5ee9855236729f72994c0b7644429f68d35e3ed11f16826a3a5ff3426fd3d","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:05d5ee9855236729f72994c0b7644429f68d35e3ed11f16826a3a5ff3426fd3d","cache_key":"sha256:9c2bbad36f862c2ab04d159d89f04c1c253fe422b0efc6e9992216c357b3e0b9"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:baseline-2026-09-21:1:9:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","old_judgment_cache_key":"sha256:9f4ca3e5cebf5402458cfe6c50c72113330a68199748402415b02fa54a838c08","scored_run_id":"arena-2026-09-21:baseline-2026-09-21:1:9:2","submission_id":"baseline-2026-09-21"},{"icp_position":6,"judgment_cache_key":"sha256:5b07c31c157c16d174f3fd3ec16419e317aa5aeddb725e9e4cf857b592ebc7fb","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6"],"judgment_input_hash":"sha256:5a60e5eeaf8a6b936eedac67a746c7d9d5caff433706431d7a16b7a7e9047f49","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:5a60e5eeaf8a6b936eedac67a746c7d9d5caff433706431d7a16b7a7e9047f49","cache_key":"sha256:5b07c31c157c16d174f3fd3ec16419e317aa5aeddb725e9e4cf857b592ebc7fb"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:6:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","old_judgment_cache_key":"sha256:eb1d7c601d36de21761da6c584116e3838b34232a72a0b5a2d44e3f28238c940","scored_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:6:1","submission_id":"sub-335b9e187d44c6b5905538aaef21f4ef"},{"icp_position":7,"judgment_cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u"],"judgment_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:7:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","old_judgment_cache_key":"sha256:d83a518052fbe0c77df17d2f623eeee9b10d2d0d76db5eafd507ac8dea68cfaf","scored_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:7:1","submission_id":"sub-335b9e187d44c6b5905538aaef21f4ef"},{"icp_position":8,"judgment_cache_key":"sha256:b0eb3e40330b95322e378114163e0a16a170f734c67c66c98dcdd16ad8fed962","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6"],"judgment_input_hash":"sha256:0400927e4822286c2bff62c860a9505cf5bcecca4858e27522bc76c85227bccd","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:0400927e4822286c2bff62c860a9505cf5bcecca4858e27522bc76c85227bccd","cache_key":"sha256:b0eb3e40330b95322e378114163e0a16a170f734c67c66c98dcdd16ad8fed962"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:8:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","old_judgment_cache_key":"sha256:f95d23e87a823de563b07cd90fadcf9c8eb6d6ad1a5f97f58945b27dd997218d","scored_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:8:1","submission_id":"sub-335b9e187d44c6b5905538aaef21f4ef"},{"icp_position":9,"judgment_cache_key":"sha256:0cb7652678a4328f7f2d90eef39a18620f5f8e50e6390b96ada071e9c3e01bd2","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6"],"judgment_input_hash":"sha256:0bcece049b52030d3cd8f282011f46d2209fcb792880ccc4068fdbbd50fd36ca","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:0bcece049b52030d3cd8f282011f46d2209fcb792880ccc4068fdbbd50fd36ca","cache_key":"sha256:0cb7652678a4328f7f2d90eef39a18620f5f8e50e6390b96ada071e9c3e01bd2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:9:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","old_judgment_cache_key":"sha256:03a03efe7a58a8cc62c685f6dd5cefb03c77c67e5ed8d1771e25a97f5fe569b8","scored_run_id":"arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:9:1","submission_id":"sub-335b9e187d44c6b5905538aaef21f4ef"},{"icp_position":7,"judgment_cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2","judgment_group_leader":false,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u"],"judgment_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:7:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","old_judgment_cache_key":"sha256:d83a518052fbe0c77df17d2f623eeee9b10d2d0d76db5eafd507ac8dea68cfaf","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:7:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799"},{"icp_position":6,"judgment_cache_key":"sha256:1ccc1d9190bc5489810768505cc10e9eed9cc2dd888d556a24eecd7f0a335c3e","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX"],"judgment_input_hash":"sha256:8d865b3c2dc962e65144a3513cf8c2e2168e10e8cf9e2a29ddc7dd169cc03947","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8d865b3c2dc962e65144a3513cf8c2e2168e10e8cf9e2a29ddc7dd169cc03947","cache_key":"sha256:1ccc1d9190bc5489810768505cc10e9eed9cc2dd888d556a24eecd7f0a335c3e"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:6:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","old_judgment_cache_key":"sha256:f376cb230ac1a39f2f44a65c10c812b6f02a011cd24ff5687c7c77336ca652fa","scored_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:6:1","submission_id":"sub-a6c72a590f2014449fb6ca7f66e420f8"},{"icp_position":7,"judgment_cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2","judgment_group_leader":false,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u"],"judgment_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:7:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","old_judgment_cache_key":"sha256:d83a518052fbe0c77df17d2f623eeee9b10d2d0d76db5eafd507ac8dea68cfaf","scored_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:7:1","submission_id":"sub-a6c72a590f2014449fb6ca7f66e420f8"},{"icp_position":8,"judgment_cache_key":"sha256:85acba0e35edc6b86881f6668158816d7966055f2cfd9640b64b9507338cc495","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX"],"judgment_input_hash":"sha256:73025d91b7924965a2fe07212af54cdcf43e0578d88fe2ea11e4f7d977c3e920","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:73025d91b7924965a2fe07212af54cdcf43e0578d88fe2ea11e4f7d977c3e920","cache_key":"sha256:85acba0e35edc6b86881f6668158816d7966055f2cfd9640b64b9507338cc495"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:8:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","old_judgment_cache_key":"sha256:24baf8072d5f2a3649c52d1fd2c9a722a1da2903dd583bd0d46ed6c10e7550fc","scored_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:8:1","submission_id":"sub-a6c72a590f2014449fb6ca7f66e420f8"},{"icp_position":9,"judgment_cache_key":"sha256:7561de9a98f8bf343036675badcd0b675ebb0f348f09f8bfc2b00d3f79170393","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX"],"judgment_input_hash":"sha256:37c1c7410d57b354ab4afa4bec38d3df495fe738ef44c9ebd8a116b66392fce7","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:37c1c7410d57b354ab4afa4bec38d3df495fe738ef44c9ebd8a116b66392fce7","cache_key":"sha256:7561de9a98f8bf343036675badcd0b675ebb0f348f09f8bfc2b00d3f79170393"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:9:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","old_judgment_cache_key":"sha256:a0741b485e7a671ee90cab1d2324d4942210325f18c90bb4fa437119135a1e46","scored_run_id":"arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:9:1","submission_id":"sub-a6c72a590f2014449fb6ca7f66e420f8"},{"icp_position":7,"judgment_cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2","judgment_group_leader":false,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u"],"judgment_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:7:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","old_judgment_cache_key":"sha256:d83a518052fbe0c77df17d2f623eeee9b10d2d0d76db5eafd507ac8dea68cfaf","scored_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:7:1","submission_id":"sub-c99d1357151933192f533043283d8e27"},{"icp_position":8,"judgment_cache_key":"sha256:450dfa54fd1b702e780e1544c4e64b35e33fae57777dab8a6d57d702d227702b","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe"],"judgment_input_hash":"sha256:a7a4466fbd7124038eb3f26b9232a7649402ac0b437b55e5c1a5b84b760f2ce6","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:a7a4466fbd7124038eb3f26b9232a7649402ac0b437b55e5c1a5b84b760f2ce6","cache_key":"sha256:450dfa54fd1b702e780e1544c4e64b35e33fae57777dab8a6d57d702d227702b"},"latest_attempt":2,"latest_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:8:score:2","latest_status":"failed","latest_terminal_cause":"judge_error","miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","old_judgment_cache_key":"sha256:5bfdba14f096853cc10ad86781a2f246a115cec497dec960f73ba9af47292b3d","scored_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:8:1","submission_id":"sub-c99d1357151933192f533043283d8e27"},{"icp_position":9,"judgment_cache_key":"sha256:14c885f51e6d4fc235526ed238e090c64382047ad1f16c2c5f05ecdd352d4724","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe"],"judgment_input_hash":"sha256:b7ab5141d80138d6430bd1441d05e4bc39cc63c5ff036250be4b0df61268f539","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:b7ab5141d80138d6430bd1441d05e4bc39cc63c5ff036250be4b0df61268f539","cache_key":"sha256:14c885f51e6d4fc235526ed238e090c64382047ad1f16c2c5f05ecdd352d4724"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:9:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","old_judgment_cache_key":"sha256:f17b168d437aa38382466f48f97335bb79e68e14117b62fcef3340f78f6ca3ce","scored_run_id":"arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:9:1","submission_id":"sub-c99d1357151933192f533043283d8e27"},{"icp_position":7,"judgment_cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2","judgment_group_leader":false,"judgment_group_miner_hotkeys":["5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u"],"judgment_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","judgment_scope_doc":{"evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-09-21","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scoring_input_hash":"sha256:8dda8a3bc60a31ee9d6af801f55eb46b0124703addfdde259f9666850377c892","cache_key":"sha256:3dc92be0b3c4b91bf7a1985bc485a758b8a87a772f8e009531c1c9586b2429b2"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-f58841a95c0ed4ffd3ecf4da0bc4063b:1:7:score:1","latest_status":"failed","latest_terminal_cause":"stage_closed","miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","old_judgment_cache_key":"sha256:d83a518052fbe0c77df17d2f623eeee9b10d2d0d76db5eafd507ac8dea68cfaf","scored_run_id":"arena-2026-09-21:sub-f58841a95c0ed4ffd3ecf4da0bc4063b:1:7:1","submission_id":"sub-f58841a95c0ed4ffd3ecf4da0bc4063b"},{"icp_position":7,"judgment_cache_key":"sha256:b5aacb82d9d79ce280772faa8fb5ec92a09c87fc456576a67fbb9c9b1adc6d6c","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"judgment_input_hash":"sha256:9214230fd5ab18571a6c5ff456c38dc794229aa03938dc9df52b6785a6bd608b","judgment_scope_doc":{"netuid":71,"round_id":"arena-2026-09-21","network_name":"finney","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","scoring_input_hash":"sha256:9214230fd5ab18571a6c5ff456c38dc794229aa03938dc9df52b6785a6bd608b","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","cache_key":"sha256:b5aacb82d9d79ce280772faa8fb5ec92a09c87fc456576a67fbb9c9b1adc6d6c"},"latest_attempt":2,"latest_run_id":"arena-2026-09-21:baseline-2026-09-21:1:7:score:2","latest_status":"failed","latest_terminal_cause":"judge_error","miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","old_judgment_cache_key":"sha256:2f7a9b38618dd13f2c82e0fa8b69159180b421ef2d24711697138697706a0d57","scored_run_id":"arena-2026-09-21:baseline-2026-09-21:1:7:2","submission_id":"baseline-2026-09-21"},{"icp_position":2,"judgment_cache_key":"sha256:c7feb7a666becd62911ef5ee496adfeb09e87119b5ca7296922e2fcd361f64f5","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K"],"judgment_input_hash":"sha256:fe8497a82dc2bb19b7d5444e71b4c7bd2ee6c349dda7d05cbca0f4e6b97f472e","judgment_scope_doc":{"netuid":71,"round_id":"arena-2026-09-21","network_name":"finney","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","scoring_input_hash":"sha256:fe8497a82dc2bb19b7d5444e71b4c7bd2ee6c349dda7d05cbca0f4e6b97f472e","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","cache_key":"sha256:c7feb7a666becd62911ef5ee496adfeb09e87119b5ca7296922e2fcd361f64f5"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:score:1","latest_status":"failed","latest_terminal_cause":"judge_error","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","old_judgment_cache_key":"sha256:4421b3989d993275a11ed2aa6ebd5c1a87b4b9214ec2e1d78d1432a1c5dfb853","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799"},{"icp_position":5,"judgment_cache_key":"sha256:0b48758d44ed8c668248bad246d3a28cd7f8a5def7648009589b335cff6a3c71","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K"],"judgment_input_hash":"sha256:c27cad0c326264b42f93b6d6ef5f427917b0c317146b931fcd7e6eb885dfc7b1","judgment_scope_doc":{"netuid":71,"round_id":"arena-2026-09-21","network_name":"finney","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","scoring_input_hash":"sha256:c27cad0c326264b42f93b6d6ef5f427917b0c317146b931fcd7e6eb885dfc7b1","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","cache_key":"sha256:0b48758d44ed8c668248bad246d3a28cd7f8a5def7648009589b335cff6a3c71"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:score:1","latest_status":"failed","latest_terminal_cause":"judge_error","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","old_judgment_cache_key":"sha256:b6c4f86ec2abd270dfa2663c6452363616b187cbf02731ce16e57b6a84cd4eb6","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799"},{"icp_position":8,"judgment_cache_key":"sha256:bac40ab5881ab679ebb031ccf76745194bfd54976c5cb60abc44d2569ebcaa52","judgment_group_leader":true,"judgment_group_miner_hotkeys":["5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K"],"judgment_input_hash":"sha256:5b16540ac5ab87db4a0d010ef379db46bc67a3cba21e0f2126853b8bb0b4d468","judgment_scope_doc":{"netuid":71,"round_id":"arena-2026-09-21","network_name":"finney","schema_version":"leadpoet.lab_arena.judgment_cache_scope.v1","evaluation_date":"2026-09-21","integrity_policy":"arena_integrity_v1","scoring_input_hash":"sha256:5b16540ac5ab87db4a0d010ef379db46bc67a3cba21e0f2126853b8bb0b4d468","scorer_image_digest":"sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6","cache_key":"sha256:bac40ab5881ab679ebb031ccf76745194bfd54976c5cb60abc44d2569ebcaa52"},"latest_attempt":1,"latest_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:score:1","latest_status":"failed","latest_terminal_cause":"judge_error","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","old_judgment_cache_key":"sha256:56c5f6b8afcb4cc8323ac41590a2bce6131e4d87fcaefd779fb7bfccdbec1e3e","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799"}]$targets$::JSONB;
  v_invalidations CONSTANT JSONB := $invalidations$[{"run_id":"arena-2026-09-21:baseline-2026-09-21:1:7:score:2","scored_run_id":"arena-2026-09-21:baseline-2026-09-21:1:7:2","submission_id":"baseline-2026-09-21","miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","icp_position":7,"attempt":2,"old_judgment_cache_key":"sha256:2f7a9b38618dd13f2c82e0fa8b69159180b421ef2d24711697138697706a0d57","judgment_input_hash":"sha256:9214230fd5ab18571a6c5ff456c38dc794229aa03938dc9df52b6785a6bd608b","old_output_ref":"arena/arena-2026-09-21/scores/items/arena-2026-09-21:baseline-2026-09-21:1:7:score:2.json","old_output_hash":"sha256:9ed7b1bd916e712535aecc3c2b2e83c77a5ae7b83204bce1d8b000246b08d098","reason":"baseline ICP 7 OncoC4 false coverage","bug":"oncoC4_false_coverage"},{"run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:score:1","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","icp_position":2,"attempt":1,"old_judgment_cache_key":"sha256:4421b3989d993275a11ed2aa6ebd5c1a87b4b9214ec2e1d78d1432a1c5dfb853","judgment_input_hash":"sha256:fe8497a82dc2bb19b7d5444e71b4c7bd2ee6c349dda7d05cbca0f4e6b97f472e","old_output_ref":"arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:score:1.json","old_output_hash":"sha256:eec6cc9f586bde491ce9ce5f29202941560d4c5636504c0e328ac520b5390a5c","reason":"miner ICP 2 HiddenLayer wrong region","bug":"hiddenlayer_wrong_region"},{"run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:score:1","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","icp_position":5,"attempt":1,"old_judgment_cache_key":"sha256:b6c4f86ec2abd270dfa2663c6452363616b187cbf02731ce16e57b6a84cd4eb6","judgment_input_hash":"sha256:c27cad0c326264b42f93b6d6ef5f427917b0c317146b931fcd7e6eb885dfc7b1","old_output_ref":"arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:score:1.json","old_output_hash":"sha256:35f846aab31d734b2043ec549ffba6af4f0eb2886a34b90a57c6df2eb07f2b74","reason":"miner ICP 5 Cyberhaven wrong region","bug":"cyberhaven_wrong_region"},{"run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:score:1","scored_run_id":"arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:1","submission_id":"sub-3c31076fbb4d3e914495fdd5dc0f4799","miner_hotkey":"5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K","icp_position":8,"attempt":1,"old_judgment_cache_key":"sha256:56c5f6b8afcb4cc8323ac41590a2bce6131e4d87fcaefd779fb7bfccdbec1e3e","judgment_input_hash":"sha256:5b16540ac5ab87db4a0d010ef379db46bc67a3cba21e0f2126853b8bb0b4d468","old_output_ref":"arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:score:1.json","old_output_hash":"sha256:2c5b8305b38a256653b6e239b4b33406cce0e900ba3143c9f9827dae62fac84a","reason":"miner ICP 8 9fin first claim rejection","bug":"9fin_first_claim_rejection"}]$invalidations$::JSONB;
  v_old_digest CONSTANT TEXT :=
    'sha256:088d77919300b6cb210003862ebd5b25608369e8a22478e16bae8e69f6adc1af';
  v_old_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:088d77919300b6cb210003862ebd5b25608369e8a22478e16bae8e69f6adc1af';
  v_new_digest CONSTANT TEXT :=
    'sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6';
  v_new_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6';
  v_namespace CONSTANT TEXT := ':score:recovery350';
  v_existing BIGINT;
  v_round_before JSONB;
  v_submissions_before JSONB;
  v_execute_before JSONB;
  v_old_scores_before JSONB;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_ledger_sum NUMERIC;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep21 score recovery round is missing'
      USING ERRCODE = 'P0002';
  END IF;

  IF pg_catalog.jsonb_typeof(v_targets) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_targets) <> 19
     OR (SELECT pg_catalog.count(DISTINCT item ->> 'scored_run_id')
         FROM pg_catalog.jsonb_array_elements(v_targets) AS item) <> 19
     OR (SELECT pg_catalog.count(DISTINCT
           (item ->> 'submission_id') || ':' || (item ->> 'icp_position'))
         FROM pg_catalog.jsonb_array_elements(v_targets) AS item) <> 19
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(v_targets) AS item
       WHERE pg_catalog.jsonb_typeof(item) IS DISTINCT FROM 'object'
          OR item ->> 'scored_run_id' IS NULL
          OR item ->> 'submission_id' IS NULL
          OR item ->> 'miner_hotkey' IS NULL
          OR (item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 9
          OR item ->> 'latest_status' IS DISTINCT FROM 'failed'
          OR item ->> 'latest_terminal_cause'
             NOT IN ('judge_error','stage_closed')
          OR (item ->> 'latest_attempt')::INTEGER NOT BETWEEN 1 AND 2
          OR item ->> 'old_judgment_cache_key' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'judgment_cache_key' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'judgment_input_hash' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'judgment_cache_key' = item ->> 'old_judgment_cache_key'
          OR item #>> '{judgment_scope_doc,cache_key}'
             IS DISTINCT FROM item ->> 'judgment_cache_key'
          OR item #>> '{judgment_scope_doc,scoring_input_hash}'
             IS DISTINCT FROM item ->> 'judgment_input_hash'
          OR item #>> '{judgment_scope_doc,scorer_image_digest}'
             IS DISTINCT FROM v_new_digest
          OR item #>> '{judgment_scope_doc,scorer_image_reference}'
             IS DISTINCT FROM v_new_reference
          OR item #>> '{judgment_scope_doc,round_id}'
             IS DISTINCT FROM 'arena-2026-09-21'
          OR item #>> '{judgment_scope_doc,network_name}'
             IS DISTINCT FROM 'finney'
          OR (item #>> '{judgment_scope_doc,netuid}')::INTEGER IS DISTINCT FROM 71
          OR item #>> '{judgment_scope_doc,integrity_policy}'
             IS DISTINCT FROM 'arena_integrity_v1'
          OR item #>> '{judgment_scope_doc,evaluation_date}'
             IS DISTINCT FROM '2026-09-21'
          OR item #>> '{judgment_scope_doc,schema_version}'
             IS DISTINCT FROM 'leadpoet.lab_arena.judgment_cache_scope.v1'
          OR pg_catalog.jsonb_typeof(item -> 'judgment_group_leader')
             IS DISTINCT FROM 'boolean'
          OR pg_catalog.jsonb_typeof(item -> 'judgment_group_miner_hotkeys')
             IS DISTINCT FROM 'array'
     ) THEN
    RAISE EXCEPTION 'Sep21 score recovery target bindings are invalid'
      USING ERRCODE = '22023';
  END IF;

  IF pg_catalog.jsonb_typeof(v_invalidations) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_invalidations) <> 4
     OR (SELECT pg_catalog.count(DISTINCT item ->> 'run_id')
         FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item) <> 4
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item
       WHERE pg_catalog.jsonb_typeof(item) IS DISTINCT FROM 'object'
          OR item ->> 'run_id' IS NULL
          OR item ->> 'scored_run_id' IS NULL
          OR item ->> 'submission_id' IS NULL
          OR item ->> 'miner_hotkey' IS NULL
          OR (item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 9
          OR (item ->> 'attempt')::INTEGER NOT BETWEEN 1 AND 2
          OR item ->> 'old_judgment_cache_key' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'judgment_input_hash' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'old_output_ref' IS NULL
          OR item ->> 'old_output_hash' !~ '^sha256:[0-9a-f]{64}$'
          OR item ->> 'reason' IS NULL
          OR item ->> 'bug' IS NULL
     ) THEN
    RAISE EXCEPTION 'Sep21 score recovery invalidations are invalid'
      USING ERRCODE = '22023';
  END IF;

  -- This independently verifies the group fields which the claim RPC uses to
  -- allocate one cache authority for equal scoring inputs.
  IF EXISTS (
    WITH payload AS (
      SELECT item
      FROM pg_catalog.jsonb_array_elements(v_targets) AS item
    ), expected AS (
      SELECT item ->> 'judgment_cache_key' AS cache_key,
        pg_catalog.jsonb_agg(DISTINCT item ->> 'miner_hotkey'
          ORDER BY item ->> 'miner_hotkey') AS miners,
        min(item ->> 'scored_run_id') AS leader
      FROM payload
      GROUP BY item ->> 'judgment_cache_key'
    )
    SELECT 1 FROM payload JOIN expected
      ON expected.cache_key = payload.item ->> 'judgment_cache_key'
    WHERE payload.item -> 'judgment_group_miner_hotkeys'
            IS DISTINCT FROM expected.miners
       OR (payload.item ->> 'judgment_group_leader')::BOOLEAN
            IS DISTINCT FROM
            (payload.item ->> 'scored_run_id' = expected.leader)
  ) THEN
    RAISE EXCEPTION 'Sep21 score recovery judgment groups are invalid'
      USING ERRCODE = '22023';
  END IF;

  SELECT pg_catalog.count(DISTINCT assignment_id) INTO v_existing
  FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id AND kind = 'score'
    AND assignment_id LIKE '%' || v_namespace;

  IF v_existing = 19 THEN
    -- A migration replay must preserve leased or completed recovery work.
    IF v_round.configuration_doc ->> 'scorer_image_digest'
         IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc ->> 'scorer_image_reference'
         IS DISTINCT FROM v_new_reference
       OR EXISTS (
         SELECT 1 FROM pg_catalog.jsonb_array_elements(v_targets) AS item
         LEFT JOIN public.lab_arena_runs AS retry
           ON retry.run_id = 'arena-2026-09-21:' ||
             (item ->> 'submission_id') || ':1:' ||
             (item ->> 'icp_position') || v_namespace || ':1'
         WHERE retry.run_id IS NULL
            OR retry.assignment_id IS DISTINCT FROM
              'arena-2026-09-21:' || (item ->> 'submission_id') || ':1:' ||
              (item ->> 'icp_position') || v_namespace
            OR retry.round_id IS DISTINCT FROM 'arena-2026-09-21'
            OR retry.submission_id IS DISTINCT FROM item ->> 'submission_id'
            OR retry.miner_hotkey IS DISTINCT FROM item ->> 'miner_hotkey'
            OR retry.stage IS DISTINCT FROM 1
            OR retry.icp_position IS DISTINCT FROM
              (item ->> 'icp_position')::SMALLINT
            OR retry.attempt IS DISTINCT FROM 1
            OR retry.kind IS DISTINCT FROM 'score'
            OR retry.scored_run_id IS DISTINCT FROM item ->> 'scored_run_id'
            OR retry.stage_generation IS DISTINCT FROM 5
            OR retry.judgment_cache_key IS DISTINCT FROM
              item ->> 'judgment_cache_key'
            OR retry.judgment_input_hash IS DISTINCT FROM
              item ->> 'judgment_input_hash'
            OR retry.judgment_scope_doc IS DISTINCT FROM
              item -> 'judgment_scope_doc'
            OR retry.judgment_group_leader IS DISTINCT FROM
              (item ->> 'judgment_group_leader')::BOOLEAN
            OR retry.judgment_group_miner_hotkeys IS DISTINCT FROM ARRAY(
              SELECT pg_catalog.jsonb_array_elements_text(
                item -> 'judgment_group_miner_hotkeys'))
       )
       OR EXISTS (
         SELECT 1 FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item
         LEFT JOIN public.lab_arena_runs AS old_run
           ON old_run.run_id = item ->> 'run_id'
         WHERE old_run.run_id IS NULL
            OR old_run.status IS DISTINCT FROM 'failed'
            OR old_run.terminal_cause IS DISTINCT FROM 'judge_error'
            OR old_run.output_ref IS DISTINCT FROM item ->> 'old_output_ref'
            OR old_run.terminal_doc ->> 'migration' IS DISTINCT FROM '350'
            OR old_run.terminal_doc ->> 'previous_output_hash'
                 IS DISTINCT FROM item ->> 'old_output_hash'
            OR old_run.terminal_doc ->> 'previous_output_ref'
                 IS DISTINCT FROM item ->> 'old_output_ref'
            OR old_run.result_doc ->> 'terminal_status' IS DISTINCT FROM 'accepted'
       ) THEN
      RAISE EXCEPTION 'Sep21 score recovery replay differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN;
  ELSIF v_existing <> 0 THEN
    RAISE EXCEPTION 'Sep21 score recovery namespace is partial'
      USING ERRCODE = '55000';
  END IF;

  -- Recovery uses a new assignment namespace for the same planned execution.
  -- Refuse to reopen the round until close-scoring selects the effective result
  -- by scored_run_id and gives an accepted retry precedence over old failures.
  IF pg_catalog.strpos(
       pg_catalog.pg_get_functiondef(
         'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure
       ),
       'COALESCE(runs.scored_run_id, runs.assignment_id)'
     ) = 0
     OR pg_catalog.strpos(
       pg_catalog.pg_get_functiondef(
         'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure
       ),
       '(runs.status = ''accepted'') DESC'
     ) = 0 THEN
    RAISE EXCEPTION 'apply 349-lab-arena-scoring-effective-outcome.sql first'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.status_generation IS DISTINCT FROM 5
     OR v_round.stage_generation IS DISTINCT FROM 4
     OR v_round.cancel_reason IS DISTINCT FROM 'scoring_incomplete'
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.configuration_doc ->> 'schema_version'
        IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'round_id'
        IS DISTINCT FROM 'arena-2026-09-21'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_old_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_old_reference
     OR v_round.configuration_doc #>> '{schedule,stage_1_scoring_close}'
        IS DISTINCT FROM '2026-09-21T11:00:01Z'
     OR v_round.configuration_doc -> 'scoring_call_quotas'
        IS DISTINCT FROM
        '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
     OR (v_round.configuration_doc ->> 'scoring_cap_microusd')::BIGINT
        IS DISTINCT FROM 50000000
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
        IS DISTINCT FROM 900
     OR v_round.benchmark_ref IS DISTINCT FROM
        'arena/arena-2026-09-21/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-09-21'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-20'
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 6
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items') IS DISTINCT FROM 59
     OR v_round.stage1_scoring_plan_doc -> 'zero_rows' IS DISTINCT FROM
        '[{"cause":"provider_error","icp_position":2,"submission_id":"baseline-2026-09-21"}]'::JSONB
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'execute') <> 130
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'execute'
           AND status = 'accepted') <> 119
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'execute'
           AND status = 'failed') <> 11
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score') <> 62
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND status = 'accepted') <> 44
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND status = 'failed') <> 18
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND terminal_cause = 'judge_error') <> 4
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND terminal_cause = 'stage_closed') <> 14 THEN
    RAISE EXCEPTION 'Sep21 score recovery terminal round differs'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item
    LEFT JOIN public.lab_arena_runs AS old_run
      ON old_run.run_id = item ->> 'run_id'
    WHERE old_run.run_id IS NULL
       OR old_run.round_id IS DISTINCT FROM v_round.round_id
       OR old_run.kind IS DISTINCT FROM 'score'
       OR old_run.status IS DISTINCT FROM 'accepted'
       OR old_run.terminal_cause IS DISTINCT FROM 'accepted'
       OR old_run.scored_run_id IS DISTINCT FROM item ->> 'scored_run_id'
       OR old_run.submission_id IS DISTINCT FROM item ->> 'submission_id'
       OR old_run.miner_hotkey IS DISTINCT FROM item ->> 'miner_hotkey'
       OR old_run.icp_position IS DISTINCT FROM (item ->> 'icp_position')::SMALLINT
       OR old_run.attempt IS DISTINCT FROM (item ->> 'attempt')::SMALLINT
       OR old_run.judgment_cache_key IS DISTINCT FROM item ->> 'old_judgment_cache_key'
       OR old_run.judgment_input_hash IS DISTINCT FROM item ->> 'judgment_input_hash'
       OR old_run.output_ref IS DISTINCT FROM item ->> 'old_output_ref'
       OR old_run.terminal_doc IS NOT NULL
  ) THEN
    RAISE EXCEPTION 'Sep21 score recovery invalidation preimage differs'
      USING ERRCODE = '55000';
  END IF;

  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal;
  UPDATE public.lab_arena_runs AS old_run
  SET status = 'failed', terminal_cause = 'judge_error',
      terminal_doc = pg_catalog.jsonb_build_object(
        'schema_version','leadpoet.lab_arena.score_recovery_invalidation.v1',
        'migration','350', 'reason',item ->> 'reason', 'bug',item ->> 'bug',
        'previous_status','accepted', 'previous_terminal_cause','accepted',
        'previous_output_ref',item ->> 'old_output_ref',
        'previous_output_hash',item ->> 'old_output_hash',
        'previous_judgment_cache_key',item ->> 'old_judgment_cache_key',
        'invalidated_at',pg_catalog.clock_timestamp())
  FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item
  WHERE old_run.run_id = item ->> 'run_id';
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id AND kind = 'score'
        AND status = 'failed' AND terminal_cause = 'judge_error'
        AND terminal_doc ->> 'migration' = '350') <> 4
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id AND kind = 'score'
        AND status = 'accepted' AND terminal_cause = 'accepted') <> 40
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_invalidations) AS item
       JOIN public.lab_arena_runs AS old_run
         ON old_run.run_id = item ->> 'run_id'
       WHERE old_run.terminal_doc ->> 'previous_output_hash'
               IS DISTINCT FROM item ->> 'old_output_hash'
          OR old_run.output_ref IS DISTINCT FROM item ->> 'old_output_ref'
          OR old_run.result_doc ->> 'terminal_status' IS DISTINCT FROM 'accepted'
     ) THEN
    RAISE EXCEPTION 'Sep21 score recovery invalidation result differs'
      USING ERRCODE = '55000';
  END IF;

  -- The unresolved plan membership must equal the exact rendered target list.
  IF EXISTS (
    WITH work AS (
      SELECT item
      FROM pg_catalog.jsonb_array_elements(
        v_round.stage1_scoring_plan_doc -> 'work_items') AS item
    ), unresolved AS (
      SELECT item ->> 'scored_run_id' AS scored_run_id,
        item ->> 'submission_id' AS submission_id,
        (item ->> 'icp_position')::INTEGER AS icp_position
      FROM work
      WHERE NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS score
        WHERE score.round_id = v_round.round_id AND score.kind = 'score'
          AND score.scored_run_id = item ->> 'scored_run_id'
          AND score.status = 'accepted')
    ), targets AS (
      SELECT item ->> 'scored_run_id' AS scored_run_id,
        item ->> 'submission_id' AS submission_id,
        (item ->> 'icp_position')::INTEGER AS icp_position
      FROM pg_catalog.jsonb_array_elements(v_targets) AS item
    )
    (SELECT * FROM unresolved EXCEPT SELECT * FROM targets)
    UNION ALL
    (SELECT * FROM targets EXCEPT SELECT * FROM unresolved)
  ) OR EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(v_targets) AS item
    LEFT JOIN public.lab_arena_runs AS latest
      ON latest.run_id = item ->> 'latest_run_id'
    WHERE latest.run_id IS NULL
       OR latest.round_id IS DISTINCT FROM v_round.round_id
       OR latest.kind IS DISTINCT FROM 'score'
       OR latest.scored_run_id IS DISTINCT FROM item ->> 'scored_run_id'
       OR latest.submission_id IS DISTINCT FROM item ->> 'submission_id'
       OR latest.miner_hotkey IS DISTINCT FROM item ->> 'miner_hotkey'
       OR latest.icp_position IS DISTINCT FROM
          (item ->> 'icp_position')::SMALLINT
       OR latest.attempt IS DISTINCT FROM
          (item ->> 'latest_attempt')::SMALLINT
       OR latest.status IS DISTINCT FROM item ->> 'latest_status'
       OR latest.terminal_cause IS DISTINCT FROM
          item ->> 'latest_terminal_cause'
       OR latest.judgment_cache_key IS DISTINCT FROM
          item ->> 'old_judgment_cache_key'
       OR latest.judgment_input_hash IS DISTINCT FROM
          item ->> 'judgment_input_hash'
       OR latest.judgment_scope_doc #>> '{scorer_image_digest}'
          IS DISTINCT FROM v_old_digest
       OR latest.judgment_scope_doc #>> '{scorer_image_reference}'
          IS DISTINCT FROM v_old_reference
  ) OR EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(
      v_round.stage1_scoring_plan_doc -> 'work_items') AS item
    LEFT JOIN public.lab_arena_runs AS executed
      ON executed.run_id = item ->> 'scored_run_id'
    WHERE executed.run_id IS NULL OR executed.kind IS DISTINCT FROM 'execute'
       OR executed.status IS DISTINCT FROM 'accepted'
       OR executed.output_ref IS DISTINCT FROM item ->> 'output_ref'
       OR executed.submission_id IS DISTINCT FROM item ->> 'submission_id'
       OR executed.icp_position IS DISTINCT FROM
          (item ->> 'icp_position')::SMALLINT
  ) THEN
    RAISE EXCEPTION 'Sep21 score recovery planned evidence differs'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM public.lab_arena_judgment_cache AS cached
    JOIN pg_catalog.jsonb_array_elements(v_targets) AS item
      ON cached.cache_key = item ->> 'judgment_cache_key'
  ) OR EXISTS (
    WITH heads AS (
      SELECT DISTINCT ON (ledger.run_id,ledger.call_identity)
        ledger.entry_kind
      FROM public.lab_arena_ledger AS ledger
      JOIN public.lab_arena_runs AS score ON score.run_id = ledger.run_id
      WHERE score.round_id = v_round.round_id AND score.kind = 'score'
        AND ledger.call_identity IS NOT NULL
      ORDER BY ledger.run_id,ledger.call_identity,ledger.entry_id DESC
    )
    SELECT 1 FROM heads WHERE entry_kind IN ('reservation','dispatch')
  ) THEN
    RAISE EXCEPTION 'Sep21 score recovery has cached new-image evidence or inflight calls'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.to_jsonb(v_round) INTO v_round_before;
  SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY submission_id)
    INTO v_submissions_before
  FROM public.lab_arena_submissions AS row_data
  WHERE round_id = v_round.round_id;
  SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY run_id)
    INTO v_execute_before
  FROM public.lab_arena_runs AS row_data
  WHERE round_id = v_round.round_id AND kind = 'execute';
  SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY run_id)
    INTO v_old_scores_before
  FROM public.lab_arena_runs AS row_data
  WHERE round_id = v_round.round_id AND kind = 'score';
  SELECT pg_catalog.count(*),pg_catalog.max(entry_id),
         COALESCE(pg_catalog.sum(amount_microusd),0)
    INTO v_ledger_count,v_ledger_max,v_ledger_sum
  FROM public.lab_arena_ledger WHERE round_id = v_round.round_id;

  INSERT INTO public.lab_arena_runs (
    run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
    icp_position,attempt,status,lease_generation,stage_generation,kind,
    scored_run_id,judgment_cache_key,judgment_input_hash,
    judgment_scope_doc,judgment_group_leader,
    judgment_group_miner_hotkeys
  )
  SELECT
    'arena-2026-09-21:' || (item ->> 'submission_id') || ':1:' ||
      (item ->> 'icp_position') || v_namespace || ':1',
    'arena-2026-09-21:' || (item ->> 'submission_id') || ':1:' ||
      (item ->> 'icp_position') || v_namespace,
    'arena-2026-09-21',item ->> 'submission_id',
    item ->> 'miner_hotkey',1,(item ->> 'icp_position')::SMALLINT,
    1,'pending',0,5,'score',item ->> 'scored_run_id',
    item ->> 'judgment_cache_key',item ->> 'judgment_input_hash',
    item -> 'judgment_scope_doc',
    (item ->> 'judgment_group_leader')::BOOLEAN,
    ARRAY(SELECT pg_catalog.jsonb_array_elements_text(
      item -> 'judgment_group_miner_hotkeys'))
  FROM pg_catalog.jsonb_array_elements(v_targets) AS item;

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1_scoring',status_generation = 6,stage_generation = 5,
      cancel_reason = NULL,
      configuration_doc = pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(configuration_doc,'{scorer_image_digest}',
          pg_catalog.to_jsonb(v_new_digest),FALSE),
        '{scorer_image_reference}',pg_catalog.to_jsonb(v_new_reference),FALSE),
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round.round_id;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;

  SELECT * INTO v_after FROM public.lab_arena_rounds
  WHERE round_id = v_round.round_id;
  IF v_after.status IS DISTINCT FROM 'stage1_scoring'
     OR v_after.status_generation IS DISTINCT FROM 6
     OR v_after.stage_generation IS DISTINCT FROM 5
     OR v_after.cancel_reason IS NOT NULL
     OR v_after.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_new_digest
     OR v_after.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_new_reference
     OR (v_after.configuration_doc - 'scorer_image_digest' -
           'scorer_image_reference') IS DISTINCT FROM
        (v_round.configuration_doc - 'scorer_image_digest' -
           'scorer_image_reference')
     OR (pg_catalog.to_jsonb(v_after) - 'status' - 'status_generation' -
           'stage_generation' - 'cancel_reason' - 'configuration_doc' -
           'updated_at') IS DISTINCT FROM
        (v_round_before - 'status' - 'status_generation' -
           'stage_generation' - 'cancel_reason' - 'configuration_doc' -
           'updated_at')
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY submission_id) FROM public.lab_arena_submissions AS row_data
         WHERE round_id = v_round.round_id) IS DISTINCT FROM
        v_submissions_before
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY run_id) FROM public.lab_arena_runs AS row_data
         WHERE round_id = v_round.round_id AND kind = 'execute')
        IS DISTINCT FROM v_execute_before
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY run_id) FROM public.lab_arena_runs AS row_data
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND assignment_id NOT LIKE '%' || v_namespace)
        IS DISTINCT FROM v_old_scores_before
     OR (SELECT ROW(pg_catalog.count(*),pg_catalog.max(entry_id),
                    COALESCE(pg_catalog.sum(amount_microusd),0))
         FROM public.lab_arena_ledger WHERE round_id = v_round.round_id)
        IS DISTINCT FROM ROW(v_ledger_count,v_ledger_max,v_ledger_sum)
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND kind = 'score'
           AND assignment_id LIKE '%' || v_namespace
           AND attempt = 1 AND status = 'pending'
           AND stage_generation = 5) <> 19 THEN
    RAISE EXCEPTION 'Sep21 score recovery postcondition differs'
      USING ERRCODE = '55000';
  END IF;
END;
$recover_sep21_unresolved_scoring$;

COMMIT;
