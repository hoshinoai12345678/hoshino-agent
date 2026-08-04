"""测试 EmotionEngine 的 PAD 情绪状态机

覆盖：
- update 后值在 [-1,1] 范围内（clamp 生效）
- _emotion_label 用 D 维度区分"不悦倔强"/"委屈"
- decay 向基线靠近
- 好感度分级
"""
from core.emotion import EmotionEngine


def test_update_and_clamp():
    """update 后值在 [-1,1] 范围内"""
    engine = EmotionEngine()
    # 传入超大 delta，验证 clamp 生效
    engine.update(
        pleasure_delta=10,
        arousal_delta=-10,
        dominance_delta=5,
        favorability_delta=200,
    )
    state = engine.state
    assert -1 <= state.pleasure <= 1
    assert -1 <= state.arousal <= 1
    assert -1 <= state.dominance <= 1
    assert -100 <= state.favorability <= 100


def test_emotion_label_uses_dominance():
    """高 D 低 P 应返回"不悦倔强"，低 D 低 P 应返回"委屈" """
    engine = EmotionEngine()

    # 低 P（<-0.3）、中 A（-0.2~0.3）、高 D（>0.2）→ 不悦倔强
    # 基线 P=0.3, A=0.5, D=0.2
    engine.reset()
    engine.update(pleasure_delta=-1, arousal_delta=-0.3, dominance_delta=0.5)
    # P=-0.7, A=0.2, D=0.7
    assert engine.get_state()["emotion_label"] == "不悦倔强"

    # 低 P（<-0.3）、中 A（-0.2~0.3）、低 D（<=0.2）→ 委屈
    engine.reset()
    engine.update(pleasure_delta=-1, arousal_delta=-0.3, dominance_delta=-0.5)
    # P=-0.7, A=0.2, D=-0.3
    assert engine.get_state()["emotion_label"] == "委屈"


def test_decay_towards_baseline():
    """decay 后状态向基线靠近"""
    engine = EmotionEngine()
    baseline_p = engine.state.pleasure
    baseline_a = engine.state.arousal
    baseline_d = engine.state.dominance

    # 把状态拉离基线
    engine.update(pleasure_delta=-1, arousal_delta=-1, dominance_delta=-1)
    moved_p = engine.state.pleasure
    moved_a = engine.state.arousal
    moved_d = engine.state.dominance

    # decay 后应向基线靠近（距离变小）
    engine.decay()
    decayed_p = engine.state.pleasure
    decayed_a = engine.state.arousal
    decayed_d = engine.state.dominance

    assert abs(decayed_p - baseline_p) < abs(moved_p - baseline_p)
    assert abs(decayed_a - baseline_a) < abs(moved_a - baseline_a)
    assert abs(decayed_d - baseline_d) < abs(moved_d - baseline_d)


def test_favorability_levels():
    """好感度分级正确（>=70挚爱，>=40亲密，>=10友好...）"""
    engine = EmotionEngine()

    # >=70 挚爱（初始 10 + 70 = 80）
    engine.reset()
    engine.update(favorability_delta=70)
    assert engine.get_state()["favorability_level"] == "挚爱"

    # >=40 亲密（10 + 40 = 50）
    engine.reset()
    engine.update(favorability_delta=40)
    assert engine.get_state()["favorability_level"] == "亲密"

    # >=10 友好（初始好感度即 10）
    engine.reset()
    assert engine.get_state()["favorability_level"] == "友好"

    # >=-10 普通（10 - 20 = -10）
    engine.reset()
    engine.update(favorability_delta=-20)
    assert engine.get_state()["favorability_level"] == "普通"

    # >=-40 疏远（10 - 50 = -40）
    engine.reset()
    engine.update(favorability_delta=-50)
    assert engine.get_state()["favorability_level"] == "疏远"

    # < -40 冷漠（10 - 60 = -50）
    engine.reset()
    engine.update(favorability_delta=-60)
    assert engine.get_state()["favorability_level"] == "冷漠"
