"""Offline behavior and failure tests; never use real credentials or X."""
import dataclasses
import io
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import bot

HERE = Path(__file__).resolve().parent


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = {**os.environ, 'MONITORED_POST_URLS': 'https://x.com/leadpoet/status/100,https://twitter.com/leadpoet/status/200', 'DRY_RUN': 'true',
                    'STATE_DB': str(Path(self.temp.name) / 'state.sqlite3')}
        for key in ('MONITORED_POST_IDS', 'X_API_KEY', 'X_API_SECRET', 'X_ACCESS_TOKEN', 'X_ACCESS_TOKEN_SECRET'):
            self.env.pop(key, None)

    def run_fixture(self, fixture=None):
        return subprocess.run([sys.executable, str(HERE / 'bot.py'), '--fixture',
                               str(fixture or HERE / 'fixture.json')], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def test_fixture_and_restart_deduplication(self):
        first = self.run_fixture()
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_fixture()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stderr.count('Dry run would perform'), 12)
        self.assertNotIn('Dry run would perform', second.stderr)
        # Compare durable state, not just log wording.
        with sqlite3.connect(self.env['STATE_DB']) as db:
            claims = db.execute("SELECT reply_id FROM comments").fetchall()
        self.assertCountEqual(claims, [('101',), ('102',), ('103',), ('104',), ('105',), ('201',)])
        with sqlite3.connect(self.env['STATE_DB']) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM actions').fetchone()[0], 12)

    def test_only_configured_parent(self):
        self.env['MONITORED_POST_URLS'] = 'https://x.com/leadpoet/status/999'
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        with sqlite3.connect(self.env['STATE_DB']) as db:
            claims = db.execute("SELECT reply_id FROM comments").fetchall()
        self.assertEqual(claims, [('109',)])

    def test_fixture_refuses_live_mode(self):
        self.env['DRY_RUN'] = 'false'
        self.assertNotEqual(self.run_fixture().returncode, 0)

    def test_bad_configuration_does_not_start(self):
        for field, value in [('DRY_RUN', 'flase'), ('MONITORED_POST_URLS', 'https://evil.example/status/100'),
                             ('POLL_INTERVAL_SECONDS', '0'), ('POLL_INTERVAL_SECONDS', 'nan'),
                             ('POLL_INTERVAL_SECONDS', 'inf'), ('REPLY_TEXT', '')]:
            with self.subTest(field=field), patch.dict(self.env, {field: value}):
                self.assertNotEqual(self.run_fixture().returncode, 0)


def reply(reply_id='101', text='Poet', parent='100', author='7'):
    return {'id': reply_id, 'text': text, 'author_id': author, 'conversation_id': parent,
            'referenced_tweets': [{'type': 'replied_to', 'id': parent}]}


class MatchingTests(unittest.TestCase):
    def test_cases_punctuation_typos_and_unrelated_words(self):
        config = bot.Config.from_env({'MONITORED_POST_URLS': 'https://x.com/leadpoet/status/100'})
        for text in ['Poet', 'poet', 'POET', 'Poet!', 'peot', 'poett', 'poeet', 'poat']:
            with self.subTest(text=text):
                self.assertTrue(bot.contains_trigger(text, config.trigger_word, config.trigger_aliases))
        for text in ['Thanks!', 'post', 'pet', 'port', 'poetry', 'poetic', 'poets']:
            with self.subTest(text=text):
                self.assertFalse(bot.contains_trigger(text, config.trigger_word, config.trigger_aliases))

    def test_url_validation_and_deduplication(self):
        config = bot.Config.from_env({'MONITORED_POST_URLS':
            'https://x.com/leadpoet/status/100?s=20,https://twitter.com/leadpoet/status/200,https://x.com/leadpoet/status/100'})
        self.assertEqual(config.monitored_post_ids, ('100', '200'))
        for url in ['https://evil.example/a/status/100', 'https://x.com@evil.example/a/status/100',
                    'https://x.com/a/status/not-an-id', 'https://x.com/a/status/100/more']:
            with self.subTest(url=url), self.assertRaises(bot.ConfigurationError):
                bot.Config.from_env({'MONITORED_POST_URLS': url})


class BotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'state.sqlite3'
        self.now = 1800000000.0
        self.config = bot.Config(('100',), state_db=self.path, dry_run=False)
        self.client = Mock()
        self.client.get_me.return_value = {'id': '42'}
        self.client.get_tweet.side_effect = lambda i: {'id': i, 'author_id': '42', 'conversation_id': i}
        self.client.search_replies.side_effect = lambda *args: iter([reply()])
        self.client.send_public_reply.return_value = ('sent', '500')
        self.client.send_dm.return_value = ('sent', '600')
        self.client.dm_unavailable_user_ids = set()

    def new_bot(self, config=None):
        config = config or self.config
        state = bot.StateStore(self.path, config.namespace)
        self.addCleanup(state.close)
        return bot.Bot(config, self.client, state, now=lambda: self.now)

    def test_live_dm_then_one_reply_after_restart(self):
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once_with('101', 'Sending!')
        self.client.send_dm.assert_called_once_with('7',
            'Book a quick demo, we’ll define your ICP, and get you 100 free lead credits:\n\n'
            'https://cal.com/team/leadpoet/chat')
        writes = [c for c in self.client.mock_calls if c[0] in ('send_public_reply', 'send_dm')]
        self.assertEqual([c[0] for c in writes], ['send_dm', 'send_public_reply'])

    def test_dm_unavailable_fallback_is_direct_and_once(self):
        self.client.send_dm.return_value = ('unavailable', 'recipient unavailable')
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once_with('101', 'sending, please open your Dms!')
        writes = [c[0] for c in self.client.mock_calls if c[0] in ('send_public_reply', 'send_dm')]
        self.assertEqual(writes, ['send_dm', 'send_public_reply'])
        self.client.send_dm.assert_called_once()

    def test_uncertain_fallback_is_not_repeated_after_restart(self):
        self.client.send_dm.return_value = ('unavailable', 'recipient unavailable')
        self.client.send_public_reply.return_value = ('unknown', 'timeout')
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once()
        self.client.send_dm.assert_called_once()

    def test_removed_post_does_not_resume_pending_dm(self):
        self.client.send_dm.return_value = ('rate_limited', self.now + 600)
        with self.assertRaises(bot.RateLimitedCycle):
            self.new_bot().run_once()
        self.now += 601
        self.client.search_replies.side_effect = lambda *args: iter([])
        self.new_bot(dataclasses.replace(self.config, monitored_post_ids=('200',))).run_once()
        self.client.send_dm.assert_called_once()

    def test_unknown_dm_does_not_retry_or_send_fallback(self):
        self.client.send_dm.return_value = ('unknown', 'timeout')
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_dm.assert_called_once()
        self.client.send_public_reply.assert_not_called()

    def test_dm_permission_rejection_does_not_blame_recipient(self):
        self.client.send_dm.return_value = ('rejected', 'HTTP 403')
        self.new_bot().run_once()
        self.client.send_public_reply.assert_not_called()

    def test_dm_crash_never_sends_public_reply_or_duplicate_dm(self):
        self.client.send_dm.side_effect = RuntimeError('simulated crash during DM')
        with self.assertRaises(RuntimeError):
            self.new_bot().run_once()
        self.client.send_dm.side_effect = None
        self.new_bot().run_once()
        self.client.send_public_reply.assert_not_called()
        self.client.send_dm.assert_called_once()

    def test_dm_rate_limit_resumes_without_search_or_ack_repeat(self):
        self.client.send_dm.side_effect = [('rate_limited', self.now + 600), ('sent', '600')]
        with self.assertRaises(bot.RateLimitedCycle):
            self.new_bot().run_once()
        self.client.search_replies.side_effect = lambda *args: iter([])
        self.now += 601
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once()
        self.assertEqual(self.client.send_dm.call_count, 2)

    def test_unknown_send_is_not_retried_after_restart(self):
        self.client.send_public_reply.return_value = ('unknown', 'timeout')
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once()

    def test_crash_after_claim_does_not_resend(self):
        self.client.send_public_reply.side_effect = RuntimeError('simulated process crash')
        with self.assertRaises(RuntimeError):
            self.new_bot().run_once()
        self.client.send_public_reply.side_effect = None
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once()

    def test_rate_limit_durable_and_cursor_does_not_skip_reply(self):
        self.client.send_public_reply.side_effect = [('rate_limited', self.now + 600), ('sent', '500')]
        first = self.new_bot()
        with self.assertRaises(bot.RateLimitedCycle):
            first.run_once()
        self.assertIsNone(first.store.cursor('100'))
        with self.assertRaises(bot.RateLimitedCycle):
            self.new_bot().run_once()
        self.assertEqual(self.client.send_public_reply.call_count, 1)
        self.now += 601
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.assertEqual(self.client.send_public_reply.call_count, 2)

    def test_existing_public_reply_blocks_a_second_reply_after_update(self):
        worker = self.new_bot()
        worker.store.add_comment('101', '100', '7', self.now)
        worker.store.claim_action('101', 'ack', self.now)
        worker.store.set_action('101', 'ack', 'sent', self.now, '500')
        self.client.send_dm.return_value = ('unavailable', 'recipient unavailable')
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.client.send_dm.assert_called_once()
        self.client.send_public_reply.assert_not_called()

    def test_public_reply_retry_does_not_repeat_delivered_dm(self):
        self.client.send_public_reply.side_effect = [('rate_limited', self.now + 600), ('sent', '500')]
        with self.assertRaises(bot.RateLimitedCycle):
            self.new_bot().run_once()
        self.now += 601
        self.new_bot().run_once()
        self.client.send_dm.assert_called_once()
        self.assertEqual(self.client.send_public_reply.call_count, 2)

    def test_dry_run_does_not_consume_live_claim(self):
        self.new_bot(dataclasses.replace(self.config, dry_run=True)).run_once()
        self.client.send_public_reply.assert_not_called()
        self.client.send_dm.assert_not_called()
        self.new_bot().run_once()
        self.client.send_public_reply.assert_called_once()

    def test_unrelated_and_own_replies_ignored(self):
        self.client.search_replies.side_effect = lambda *args: iter([
            reply(text='Thanks'), reply(text='poetry'), reply(parent='999'), reply(author='42'),
            {**reply(), 'referenced_tweets': [{'type': 'replied_to', 'id': '88'}]}])
        self.new_bot().run_once()
        self.client.send_public_reply.assert_not_called()

    def test_opt_out_ignored(self):
        self.new_bot(dataclasses.replace(self.config, denied_user_ids=frozenset({'7'}))).run_once()
        self.client.send_public_reply.assert_not_called()

    def test_posts_not_owned_by_us_rejected(self):
        self.client.get_tweet.side_effect = None
        self.client.get_tweet.return_value = {'id': '100', 'author_id': '9', 'conversation_id': '100'}
        with self.assertRaises(bot.ConfigurationError):
            self.new_bot().run_once()
        self.client.search_replies.assert_not_called()

    def test_forbidden_reply_does_not_stop_next_recipient(self):
        self.client.search_replies.side_effect = lambda *args: iter([reply(), reply('102')])
        self.client.send_public_reply.side_effect = [('rejected', 'HTTP 403'), ('sent', '500')]
        self.new_bot().run_once()
        self.new_bot().run_once()
        self.assertEqual(self.client.send_public_reply.call_count, 2)

    def test_recent_root_waits_for_search_index_delay(self):
        self.client.get_tweet.side_effect = None
        self.client.get_tweet.return_value = {'id': '100', 'author_id': '42', 'conversation_id': '100',
                                             'created_at': bot._rfc3339(self.now - 5)}
        self.new_bot().run_once()
        self.client.search_replies.assert_not_called()

    def test_checkpoint_held_when_later_page_fails(self):
        def pages(*args):
            yield reply()
            raise bot.FatalApiError('page two failed')
        self.client.search_replies.side_effect = pages
        first = self.new_bot()
        with self.assertRaises(bot.FatalApiError):
            first.run_once()
        self.assertIsNone(first.store.cursor('100'))
        self.client.search_replies.side_effect = lambda *args: iter([reply(), reply('102')])
        self.new_bot().run_once()
        self.assertEqual(self.client.send_public_reply.call_count, 2)

    def test_old_post_search_stays_within_seven_days(self):
        self.new_bot().run_once()
        _, start, end = self.client.search_replies.call_args.args
        self.assertGreater(bot._to_timestamp(start), self.now - 7 * 86400)
        self.assertLess(bot._to_timestamp(end), self.now)

    def test_partial_startup_is_revalidated(self):
        config = dataclasses.replace(self.config, monitored_post_ids=('100', '200'))
        self.client.get_tweet.side_effect = [
            {'id': '100', 'author_id': '42', 'conversation_id': '100'},
            bot.TransientApiError('offline')]
        worker = self.new_bot(config)
        with self.assertRaises(bot.TransientApiError):
            worker.run_once()
        self.assertEqual(worker.roots, {})
        self.client.get_tweet.side_effect = lambda i: {'id': i, 'author_id': '42', 'conversation_id': i}
        worker.run_once()
        self.assertEqual(set(worker.roots), {'100', '200'})

    def test_startup_network_failure_retries_without_process_restart(self):
        worker = self.new_bot()
        self.client.get_me.side_effect = bot.TransientApiError('offline')
        worker.sleep = lambda delay: worker.stop()
        worker.run_forever()
        self.client.send_public_reply.assert_not_called()

    def test_second_process_cannot_acquire_same_lock(self):
        with bot.ProcessLock(self.path):
            with self.assertRaises(bot.BotError):
                with bot.ProcessLock(self.path):
                    self.fail('second process acquired lock')


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = bot.XApiClient(bot.Credentials('fake-key', 'fake-secret', 'fake-token',
                                                    'fake-token-secret'), now=lambda: 1000,
                                    opener=Mock(side_effect=AssertionError('Network is forbidden in tests')),
                                    sleep=Mock())

    def test_pagination_keeps_exact_query_and_window(self):
        self.client.get_json = Mock(side_effect=[
            {'data': [reply()], 'meta': {'result_count': 1, 'next_token': 'page-two'}},
            {'data': [reply('102')], 'meta': {'result_count': 1}}])
        rows = list(self.client.search_replies('100', 'start', 'end'))
        self.assertEqual([row['id'] for row in rows], ['101', '102'])
        calls = self.client.get_json.call_args_list
        self.assertEqual(calls[0].args[1]['query'], 'conversation_id:100 is:reply -is:retweet')
        self.assertNotIn('next_token', calls[0].args[1])
        self.assertEqual(calls[1].args[1]['next_token'], 'page-two')
        for call in calls:
            self.assertEqual(call.args[1]['start_time'], 'start')
            self.assertEqual(call.args[1]['end_time'], 'end')

    def test_partial_search_error_does_not_look_complete(self):
        self.client.get_json = Mock(return_value={'errors': [{'title': 'Partial failure'}],
                                                  'meta': {'result_count': 0}})
        with self.assertRaises(bot.BotError):
            list(self.client.search_replies('100', 'start', 'end'))

    def test_get_retries_rate_limit_then_server_failure(self):
        self.client._request_once = Mock(side_effect=[
            (429, {'x-rate-limit-reset': '1030'}, None), (503, {}, None),
            (200, {}, {'data': {'id': '42'}})])
        self.assertEqual(self.client.get_me()['id'], '42')
        self.assertEqual(self.client._request_once.call_count, 3)
        self.assertGreaterEqual(self.client.sleep.call_args_list[0].args[0], 30)

    def test_last_rate_limit_carries_delay_to_next_cycle(self):
        self.client.get_max_attempts = 1
        self.client._request_once = Mock(return_value=(429, {'x-rate-limit-reset': '1500'}, None))
        with self.assertRaises(bot.RateLimitedCycle) as caught:
            self.client.get_me()
        self.assertGreaterEqual(caught.exception.retry_at, 1500)

    def test_get_auth_failure_does_not_retry(self):
        self.client._request_once = Mock(return_value=(401, {}, None))
        with self.assertRaises(bot.FatalApiError):
            self.client.get_me()
        self.client._request_once.assert_called_once()

    def test_uncertain_post_never_retries(self):
        for result in [(503, {}, None), (201, {}, {'data': {}}), URLError('disconnected')]:
            with self.subTest(result=result):
                self.client._request_once = Mock(side_effect=result if isinstance(result, Exception) else None,
                                                 return_value=result)
                self.assertEqual(self.client.send_public_reply('101', 'CTA')[0], 'unknown')
                self.client._request_once.assert_called_once()

    def test_post_uses_correct_reply_payload(self):
        self.client._request_once = Mock(return_value=(201, {}, {'data': {'id': '500'}}))
        self.assertEqual(self.client.send_public_reply('101', 'CTA'), ('sent', '500'))
        self.client._request_once.assert_called_once_with('POST', '/2/tweets',
            payload={'text': 'CTA', 'reply': {'in_reply_to_tweet_id': '101'}})

    def test_dm_payload_and_event_id(self):
        self.client._request_once = Mock(return_value=(201, {}, {'data': {'dm_event_id': '600'}}))
        self.assertEqual(self.client.send_dm('7', 'hello'), ('sent', '600'))
        self.client._request_once.assert_called_once_with('POST', '/2/dm_conversations/with/7/messages',
                                                         payload={'text': 'hello'})

    def test_only_recipient_dm_errors_enable_fallback(self):
        for code, expected in [(150, 'unavailable'), (349, 'unavailable'), (93, 'rejected'),
                               (220, 'rejected'), (261, 'rejected')]:
            with self.subTest(code=code):
                self.client._request_once = Mock(return_value=(403, {}, {'errors': [{'code': code}]}))
                self.assertEqual(self.client.send_dm('7', 'hello')[0], expected)
        self.client._request_once = Mock(return_value=(403, {}, {'title': 'Forbidden'}))
        self.assertEqual(self.client.send_dm('7', 'hello')[0], 'rejected')

    def test_real_http_error_body_classifies_recipient_without_logging_body(self):
        self.client.opener = Mock(side_effect=HTTPError(
            'https://api.x.com/2/dm_conversations/with/7/messages', 403, 'Forbidden', {},
            io.BytesIO(b'{"errors":[{"code":349,"message":"You cannot send messages to this user."}]}')))
        status, detail = self.client.send_dm('7', 'hello')
        self.assertEqual(status, 'unavailable')
        self.assertNotIn('You cannot send', str(detail))

    def test_dm_uncertainty_never_retries_in_transport(self):
        for result in [(500, {}, None), (500, {}, {'errors': [{'code': 349}]}),
                       (201, {}, {}), URLError('disconnected')]:
            with self.subTest(result=result):
                self.client._request_once = Mock(side_effect=result if isinstance(result, Exception) else None,
                                                 return_value=result)
                self.assertEqual(self.client.send_dm('7', 'hello')[0], 'unknown')
                self.client._request_once.assert_called_once()

    def test_retry_after_parsing_and_both_limits(self):
        self.assertGreaterEqual(bot._retry_delay({'Retry-After': '20', 'x-rate-limit-reset': '1100'}, 1000, 5), 100)
        self.assertGreaterEqual(bot._retry_delay({'Retry-After': 'bad'}, 1000, 5), 5)

    def test_oauth_known_public_test_vector(self):
        # Public OAuth 1.0 example values, not operational credentials.
        credentials = bot.Credentials('dpf43f3p2l4k3l03', 'kd94hf93k423kf44',
                                      'nnch734d00sl2jdk', 'pfkkdhi9sl3r4s00')
        header = bot.oauth1_authorization('GET', 'http://photos.example.net/photos', credentials,
            query={'file': 'vacation.jpg', 'size': 'original'}, nonce='kllo9940pd9333jh', timestamp=1191242096)
        self.assertIn('tR3%2BTy81lMeYAr%2FFid0kMTYa%2FWM%3D', header)


if __name__ == '__main__':
    unittest.main()
