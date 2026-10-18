"""IB orchestration cannot mistake frozen Head loss for renderer learning."""

from tools.run_runc_v2_clean_stage_b import coarse_health_passed


def record(**after):
    return {"frozen_state_passed": True,
            "before_fixed_eval": {"total": 3, "raw_density": 1,
                                  "raw_occupancy": 2},
            "after_fixed_eval": {"total": 2, "raw_density": 0.5,
                                 "raw_occupancy": 1, **after}}


def test_all_three_actual_renderer_losses_must_drop():
    assert coarse_health_passed(record())[0]
    assert not coarse_health_passed(record(raw_density=1.1))[0]
    assert not coarse_health_passed(record(raw_occupancy=2.1))[0]


def test_learning_cannot_hide_a_frozen_parameter_change():
    value = record()
    value["frozen_state_passed"] = False
    assert not coarse_health_passed(value)[0]
