"""Readable explanations must not acquire unsupported basketball meaning."""
import copy
import unittest

from backend.app.game_summary import _validate
from backend.app.trusted_evidence import build_trusted_evidence_v2 as build
from backend.app.trusted_evidence import validate_trusted_evidence_v2 as validate


def play():
    return {
        'source': {'durationSeconds': 5, 'fps': 10, 'frameCount': 50},
        'frames': [{'frameIndex': i, 'players': [{'id': 1, 'teamId': 1}, {'id': 2, 'teamId': 1}]} for i in range(50)],
        'diagnostics': {'possessionTimeline': {'semantic': {'frames': [
            {'frame_index': i, 'holder_id': 1 if i < 10 else 2 if i <= 22 else None,
             'state': 'controlled' if i <= 22 else 'unknown'} for i in range(50)]}}},
        'events': [
            {'type': 'pass', 'frameIndex': 10, 'timeSeconds': 1.0, 'fromTeamId': 1, 'toTeamId': 1,
             'evidence': {'from_player_id': 1, 'to_player_id': 2, 'release_frame': 7, 'catch_frame': 10,
                          'possession_evidence': {'source_support_frames': 3, 'receiver_support_frames': 3}}},
            {'type': 'shot_attempt', 'frameIndex': 25, 'timeSeconds': 2.5, 'fromTeamId': 1, 'toTeamId': 1,
             'evidence': {'from_player_id': 2, 'release_frame': 22}},
        ],
    }


class ReplayReviewTests(unittest.TestCase):
    def test_linked_play_mentions_timing_actor_and_replay_not_cause(self):
        p = build(play())
        self.assertEqual(len(p['supportedClaims']), 1)
        claim = p['supportedClaims'][0]
        self.assertIn('followed by a shot-attempt candidate near 2.5s', claim['claim'])
        self.assertIn('Team 1 track T2', claim['claim'])
        self.assertIn('Replay from 0.0s', claim['claim'])
        for word in ('completed', 'leading to', 'caused', 'frames', 'meters', 'nearby'):
            self.assertNotIn(word, claim['claim'])
        seq = next(s for s in p['sequences'] if s['id'] in claim['evidenceIds'])
        self.assertEqual(len(seq['evidenceIds']), 3)
        validate(p)

    def test_gaps_cuts_different_actor_and_short_support_prevent_links(self):
        for mode in ('unknown', 'cut', 'actor', 'support', 'unknown_team', 'missing_release'):
            a = play()
            if mode == 'unknown':
                a['diagnostics']['possessionTimeline']['semantic']['frames'][16]['state'] = 'unknown'
            elif mode == 'cut':
                a['measurements'] = {'discontinuityFrames': [16]}
            elif mode == 'actor':
                a['events'][1]['evidence']['from_player_id'] = 1
            elif mode == 'support':
                a['events'][0]['evidence']['possession_evidence']['receiver_support_frames'] = 1
            elif mode == 'unknown_team':
                a['events'][0]['toTeamId'] = None
            else:
                a['events'][1]['evidence'].pop('release_frame')
            with self.subTest(mode=mode):
                self.assertNotIn('followed by', str(build(a)['supportedClaims']))

    def test_intervening_event_prevents_combining_plays(self):
        a = play()
        e = copy.deepcopy(a['events'][1])
        e.update(frameIndex=18, timeSeconds=1.8)
        e['evidence']['release_frame'] = 17
        a['events'].append(e)
        p = build(a)
        # A potential shot at 1.8s may link, but never skip it to link the 2.5s shot.
        self.assertNotIn('followed by a shot-attempt candidate near 2.5s', str(p['supportedClaims']))

    def test_invalid_sequence_cannot_be_injected(self):
        p = build(play())
        sequence = next(s for s in p['sequences'] if s['id'] == p['supportedClaims'][0]['evidenceIds'][0])
        sequence['evidenceIds'].reverse()
        with self.assertRaises(ValueError):
            validate(p)

    def test_single_shot_is_replay_review_without_invented_pass(self):
        a = play()
        a['events'] = a['events'][1:]
        p = build(a)
        prose = p['summaryOptions'][0] + p['supportedClaims'][0]['claim']
        self.assertIn('Team 1 track T2', prose)
        self.assertIn('Start the replay at 0.5s', prose)
        self.assertNotIn('pass', prose)
        self.assertNotIn('frames', prose)

    def test_no_events_has_honest_non_statistical_fallback(self):
        a = play()
        a['events'] = []
        p = build(a)
        self.assertIn('does not support a clear play-by-play', p['summaryOptions'][0])
        self.assertNotIn('frames', str(p['supportedClaims']))
        report = {'summary': p['summaryOptions'][0], 'tacticalInsights': [], 'limitations': p['limitations']}
        _validate(report, p)

    def test_invented_completed_pass_and_nearby_count_rejected(self):
        p = build(play())
        report = {'summary': p['summaryOptions'][0], 'limitations': [],
                  'tacticalInsights': [{**p['supportedClaims'][0], 'caveat': None}]}
        _validate(report, p)
        for claim in ('A pass was completed at 1.0s.', 'Three defenders surround T2 at the shot.'):
            report['tacticalInsights'][0]['claim'] = claim
            with self.assertRaises(ValueError):
                _validate(report, p)

    def test_deterministic_review(self):
        self.assertEqual(build(play()), build(play()))


if __name__ == '__main__':
    unittest.main()
