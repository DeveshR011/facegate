"""Authentication and enrollment state machines, driven with a fake engine."""

from __future__ import annotations

from conftest import make_face, sample, toward
from facegate.config import Settings
from facegate.core.auth import (ANCHOR, NO_MATCH_FRAMES, POSES, SPOOF_FRAMES, AuthSession,
                                AuthState, EnrollSession, learn_after_password)
from facegate.core.occlusion import VIEWS
from facegate.core.store import TemplateStore


def enrolled_store(tmp_path, owner):
    store = TemplateStore(tmp_path / "t.bin")
    store.add_appearance("Primary", [owner])
    return store


def run(session, frame, limit=200):
    for i in range(limit):
        update = session.feed(frame)
        if update.done:
            return update, i + 1
    raise AssertionError("session never finished")


def test_not_enrolled(engine, frame, tmp_path):
    update = AuthSession(engine, TemplateStore(tmp_path / "t.bin"), Settings()).feed(frame)
    assert update.state is AuthState.NOT_ENROLLED and update.done and not update.ok


def test_owner_unlocks_after_required_frames(engine, frame, tmp_path):
    s = Settings()
    owner = sample(1)
    engine.embedding = owner
    update, frames = run(AuthSession(engine, enrolled_store(tmp_path, owner), s), frame)
    assert update.ok and update.view == "full"
    assert frames == s.frames_required                 # no faster, no slower


def test_stranger_is_rejected_and_liveness_is_not_even_run(engine, frame, tmp_path):
    engine.embedding = sample(2)
    update, frames = run(AuthSession(engine, enrolled_store(tmp_path, sample(1)), Settings()), frame)
    assert update.state is AuthState.FAILED
    assert frames == NO_MATCH_FRAMES
    assert engine.liveness_calls == 0


def test_photo_of_owner_is_rejected_as_spoof(engine, frame, tmp_path):
    owner = sample(1)
    engine.embedding = owner
    engine.live = 0.05
    update, frames = run(AuthSession(engine, enrolled_store(tmp_path, owner), Settings()), frame)
    assert update.state is AuthState.SPOOF and not update.ok
    assert frames == SPOOF_FRAMES


def test_liveness_can_be_disabled(engine, frame, tmp_path):
    owner = sample(1)
    engine.embedding = owner
    engine.live = 0.05
    s = Settings()
    s.require_liveness = False
    update, _ = run(AuthSession(engine, enrolled_store(tmp_path, owner), s), frame)
    assert update.ok and engine.liveness_calls == 0


def test_one_bad_liveness_frame_does_not_block_the_owner(engine, frame, tmp_path):
    s = Settings()
    owner = sample(1)
    engine.embedding = owner
    session = AuthSession(engine, enrolled_store(tmp_path, owner), s)
    engine.live = 0.10
    assert not session.feed(frame).done
    engine.live = 0.95
    update, _ = run(session, frame)
    assert update.ok


def test_masked_owner_unlocks_through_the_mask_view(engine, frame, tmp_path):
    s = Settings()
    owner = sample(1)
    store = enrolled_store(tmp_path, owner)
    engine.embedding = {"full": toward(owner["full"], 0.34), "mask": toward(owner["mask"], 0.60)}
    update, _ = run(AuthSession(engine, store, s), frame)
    assert update.ok and update.view == "mask"

    s.occlusion_unlock = False
    update, _ = run(AuthSession(engine, store, s), frame)
    assert update.state is AuthState.FAILED


def test_no_face_times_out(engine, frame, tmp_path, clock):
    s = Settings()
    engine.faces = []
    session = AuthSession(engine, enrolled_store(tmp_path, sample(1)), s, clock=clock)
    assert session.feed(frame).state is AuthState.SEARCHING
    clock.advance(s.timeout_s + 0.1)
    assert session.feed(frame).state is AuthState.TIMEOUT


def test_small_face_asks_user_to_move_closer(engine, frame, tmp_path):
    engine.faces = [make_face(width=40)]
    update = AuthSession(engine, enrolled_store(tmp_path, sample(1)), Settings()).feed(frame)
    assert update.state is AuthState.SEARCHING and "closer" in update.message


def test_result_is_sticky(engine, frame, tmp_path):
    owner = sample(1)
    engine.embedding = owner
    session = AuthSession(engine, enrolled_store(tmp_path, owner), Settings())
    update, _ = run(session, frame)
    engine.embedding = sample(9)
    assert session.feed(frame) is update


def look(base, similarity, seed):
    """A changed look: every view `similarity` away from `base`. (Seeds clear of sample()'s.)"""
    return {v: toward(base[v], similarity, seed=1000 + 10 * seed + i) for i, v in enumerate(VIEWS)}


def test_adaptive_learning_keeps_new_looks_from_everyday_unlocks(engine, frame, tmp_path):
    owner = sample(1)
    store = enrolled_store(tmp_path, owner)

    # a look the gallery covers only loosely is learned...
    engine.embedding = look(owner, 0.55, seed=5)
    session = AuthSession(engine, store, Settings())
    assert run(session, frame)[0].ok
    assert store.learned_count == 1 and "learned this look" in session.summary()

    # ...one it already covers well adds nothing...
    engine.embedding = look(owner, 0.90, seed=6)
    assert run(AuthSession(engine, store, Settings()), frame)[0].ok
    assert store.learned_count == 1

    # ...and a match that only just clears the threshold unlocks but is not trusted to learn from
    engine.embedding = look(owner, 0.39, seed=7)
    assert run(AuthSession(engine, store, Settings()), frame)[0].ok
    assert store.learned_count == 1

    # never when the option is off
    s = Settings()
    s.adaptive_learning = False
    (tmp_path / "second").mkdir()
    fresh = enrolled_store(tmp_path / "second", owner)
    engine.embedding = look(owner, 0.55, seed=8)
    assert run(AuthSession(engine, fresh, s), frame)[0].ok
    assert fresh.learned_count == 0


def test_a_sudden_change_is_learned_after_the_password(engine, frame, tmp_path):
    """Haircut and new glasses: the scan misses, the password proves it was you, and from then
    on the new look is recognised."""
    owner = sample(1)
    store = enrolled_store(tmp_path, owner)
    engine.embedding = look(owner, 0.30, seed=11)          # resembles you, but below the threshold

    session = AuthSession(engine, store, Settings())
    update, _ = run(session, frame)
    assert update.state is AuthState.FAILED
    assert session.near_misses and store.learned_count == 0     # kept, not yet learned
    assert "failed" in session.summary() and "best score 0.30" in session.summary()

    assert learn_after_password(store, session.near_misses, Settings()) == 1
    assert run(AuthSession(engine, store, Settings()), frame)[0].ok


def test_near_misses_must_be_live_and_resemble_you(engine, frame, tmp_path):
    owner = sample(1)
    store = enrolled_store(tmp_path, owner)

    engine.embedding = sample(2)                            # a stranger
    session = AuthSession(engine, store, Settings())
    run(session, frame)
    assert session.near_misses == [] and engine.liveness_calls == 0

    engine.embedding = look(owner, 0.30, seed=12)           # a photo of a changed look
    engine.live = 0.20
    session = AuthSession(engine, store, Settings())
    run(session, frame)
    assert session.near_misses == []

    s = Settings()
    s.adaptive_learning = False                             # and nothing at all when learning is off
    engine.live = 0.95
    session = AuthSession(engine, store, s)
    run(session, frame)
    assert session.near_misses == []
    assert learn_after_password(store, [(0.3, look(owner, 0.3, seed=13))], s) == 0


def test_learned_looks_stay_anchored_to_enrollment(engine, frame, tmp_path):
    """A face matched only through a learned sample, no longer like the enrolled one, unlocks
    but is not learned from: learned looks cannot drift step by step to someone else."""
    owner = sample(1)
    store = enrolled_store(tmp_path, owner)
    learned = look(owner, 0.30, seed=14)
    store.add_adaptive(learned)
    drifted = look(learned, 0.60, seed=15)
    assert store.enrolled_score(drifted) < ANCHOR

    engine.embedding = drifted
    assert run(AuthSession(engine, store, Settings()), frame)[0].ok
    assert store.learned_count == 1


# ---- enrollment ----------------------------------------------------------

POSE_OFFSETS = {"center": (0, 0), "left": (0.25, 0), "right": (-0.25, 0), "up": (0, -0.15), "down": (0, 0.15)}


def test_enrollment_walks_through_every_pose(engine, frame, clock):
    session = EnrollSession(engine, Settings(), clock=clock)
    seen = []
    for _ in range(400):
        if session.done:
            break
        pose = session.poses[session._stage]
        yaw, pitch = POSE_OFFSETS[pose]
        engine.faces = [make_face(yaw=yaw, pitch=pitch)]
        clock.advance(0.3)
        update = session.feed(frame)
        if update.captured:
            seen.append(pose)
    assert session.done
    assert len(session.samples) == session.total == 3 + 2 * 4
    assert [p for i, p in enumerate(seen) if i == 0 or seen[i - 1] != p] == list(POSES)
    assert update.progress == 1.0


def test_enrollment_does_not_capture_the_wrong_pose(engine, frame, clock):
    session = EnrollSession(engine, Settings(), clock=clock)
    engine.faces = [make_face()]
    for _ in range(3):                       # centre pose
        clock.advance(0.3)
        session.feed(frame)
    assert session.poses[session._stage] == "left"
    for _ in range(5):                       # still looking straight ahead
        clock.advance(0.3)
        assert not session.feed(frame).captured
    assert len(session.samples) == 3


def test_enrollment_skips_an_unreachable_pose_instead_of_hanging(engine, frame, clock):
    session = EnrollSession(engine, Settings(), clock=clock)
    engine.faces = [make_face()]
    for _ in range(500):
        if session.done:
            break
        clock.advance(0.5)
        session.feed(frame)
    assert session.done and len(session.samples) == 3      # centre only


def test_enrollment_rejects_bad_conditions(engine, frame, clock):
    session = EnrollSession(engine, Settings(), clock=clock)
    clock.advance(1)

    engine.faces = []
    assert "Position" in session.feed(frame).message
    engine.faces = [make_face(), make_face(width=220)]
    assert "one person" in session.feed(frame).message.lower()
    engine.faces = [make_face(width=60)]
    assert "closer" in session.feed(frame).message
    engine.faces = [make_face()]
    engine.sharp = 2.0
    assert "still" in session.feed(frame).message.lower()
    engine.sharp = 120.0
    engine.live = 0.1
    assert "lighting" in session.feed(frame).message
    engine.live = 0.95
    assert session.feed(frame).captured
    assert len(session.samples) == 1


def test_enrollment_ignores_a_second_person(engine, frame, clock):
    session = EnrollSession(engine, Settings(), clock=clock)
    clock.advance(1)
    assert session.feed(frame).captured
    engine.embedding = sample(8)             # somebody else leans in
    clock.advance(1)
    update = session.feed(frame)
    assert not update.captured and "same person" in update.message
