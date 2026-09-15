import json

import pytest

from curve_fx_harness_client.models import OpenSessionFrame
from fxopt.config import ConfigError, RunConfig
from fxopt.run import open_session_request


def test_session_admission_matches_retained_harness_modes(tmp_path):
    cases = [
        ({}, {}, True),
        ({}, {'yb_mode': 'active_2l'}, True),
        ({'state_reconciliation_mode': 'on_price_scale_detach', 'equalization_delay_s': 20},
         {'observed_state': 'state.jsonl'}, True),
        ({'actor_timing_mode': 'scheduled_actor'}, {}, False),
        ({}, {'actor_timeline': 'timeline.jsonl'}, False),
        ({}, {'yb_oracle': 'oracle.jsonl'}, False),
        ({}, {'yb_mode': 'active_2l', 'observed_state': 'state.jsonl'}, False),
    ]
    cases += [({'state_reconciliation_mode': mode}, {'observed_state': 'state.jsonl'}, False)
              for mode in ('on_external_skip', 'periodic', 'on_mismatch',
                           'on_mismatch_with_inputs', 'on_unexplained_skip')]
    cases += [({key: value}, {}, False) for key, value in (
        ('actor_hedge_mode', 'prehedged'), ('actor_hedge_delay_ns', 0),
        ('equalization_interval_s', 60), ('pool_ping_timestamps', []),
        ('actor_timeline_path', 'timeline.jsonl'), ('yb_oracle_path', 'oracle.jsonl'))]
    for session_patch, scenario_patch, accepted in cases:
        scenario = dict(id='case', cex_depth='book.npz', yb_mode='reference_2l') | scenario_patch
        session = dict(actor_timing_mode='minute_sequential', observation_interval_s=37) | session_patch
        path = tmp_path/'case.toml'
        text = '[run]\nid = "case"\nevaluator = "evaluator"\ntemplate = "template.json"\nbatch_size = 1\nworkers = 1\nmetric_fields = ["score"]\n'
        for name, fields in (('scenario', scenario), ('session', session)):
            text += f'[{name}]\n' + ''.join(f'{key} = {json.dumps(value)}\n' for key, value in fields.items())
        path.write_text(text + '[candidate.defaults]\npolicy_params = []\npool = {}\n')
        if not accepted:
            with pytest.raises(ConfigError):
                RunConfig.from_toml(path)
            continue
        config = RunConfig.from_toml(path)
        frame = OpenSessionFrame.model_validate(dict(open_session_request(config), request_id='r', session_id='s'))
        assert frame.event_mode == 'depth' and frame.observation_interval_s == 37
        assert frame.yb_mode == scenario['yb_mode'] and frame.market_path is None
