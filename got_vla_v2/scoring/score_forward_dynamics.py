"""
score_forward_dynamics.py
-------------------------
Forward Dynamics Score: 시뮬레이터를 n_lookahead 스텝 미리 실행하여
reward + task progress로 행동 후보를 평가한다.

주의: 환경 상태 저장/복원(Save & Load)이 필요하므로
시뮬레이터 환경에서만 동작한다. 실제 로봇에는 적용 불가.
"""
from __future__ import annotations
import numpy as np


def get_env_state(env):
    """환경 상태를 numpy array로 저장한다."""
    if hasattr(env, "get_sim_state"):
        return env.get_sim_state()
    if hasattr(env, "sim") and hasattr(env.sim, "get_state"):
        return env.sim.get_state().flatten()
    raise AttributeError("[FD Score] 상태 저장 불가")


def set_env_state(env, state):
    """환경을 저장된 상태로 복원하고 done 플래그를 리셋한다."""
    if hasattr(env, "set_state"):
        env.set_state(state)
        # robosuite done 플래그 리셋 (terminated episode 오류 방지)
        if hasattr(env, "env") and hasattr(env.env, "done"):
            env.env.done = False
        if hasattr(env, "done"):
            env.done = False
        # 물리 시뮬레이터 상태 동기화
        if hasattr(env, "sim"):
            env.sim.forward()
        elif hasattr(env, "env") and hasattr(env.env, "sim"):
            env.env.sim.forward()
        return
    if hasattr(env, "sim") and hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        return
    raise AttributeError("[FD Score] 상태 복원 불가")


def compute_task_progress(obs: dict, task_description: str) -> float:
    """
    관측값에서 태스크 진행도를 추정한다.
    그리퍼 상태 + end-effector 높이로 물체를 집었는지 근사 측정.
    """
    try:
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        gripper_closed = float(np.mean(gripper_qpos) < 0.035)
        eef_height = float(eef_pos[2]) if len(eef_pos) > 2 else 0.0
        height_score = min(1.0, max(0.0, (eef_height - 0.82) / 0.25))
        return 0.5 * gripper_closed + 0.5 * height_score
    except Exception:
        return 0.5


def forward_dynamics_score(
    env,
    candidates: list,
    current_obs: dict,
    task_description: str,
    n_lookahead: int = 2,
    unnorm_fn=None,
    verbose: bool = False,
) -> list:
    """각 후보를 n_lookahead 스텝 시뮬레이션 후 score를 계산하고 환경을 복원한다."""
    if not candidates:
        return candidates

    try:
        saved_state = get_env_state(env)
        if verbose:
            print(f"  [FD Score] 상태 저장 완료, {len(candidates)}개 후보 평가 시작")

        for i, cand in enumerate(candidates):
            if not cand.actions:
                cand.score = -float("inf")
                continue

            reward = 0.0
            obs = current_obs

            try:
                score_sum = 0.0
                actual_lookahead = min(n_lookahead, len(cand.actions))

                for step_idx in range(actual_lookahead):
                    raw_action = cand.actions[step_idx]
                    action = unnorm_fn(raw_action) if unnorm_fn else raw_action
                    obs, reward, done, info = env.step(action.tolist())

                    step_weight = (step_idx + 1) / actual_lookahead
                    progress = compute_task_progress(obs, task_description)
                    score_sum += step_weight * (reward * 0.7 + progress * 0.3)

                    if done:
                        score_sum = 10.0  # 태스크 완료 보너스
                        set_env_state(env, saved_state)
                        break

                cand.score = score_sum / actual_lookahead
                cand.score_collision = float(reward)
                cand.score_consistency = compute_task_progress(obs, task_description)

                if verbose:
                    print(f"  [FD Score] 후보 {i+1}: score={cand.score:.4f} "
                          f"(reward={reward:.3f}, progress={cand.score_consistency:.3f})")

            except Exception as e:
                if verbose:
                    print(f"  [FD Score] 후보 {i+1} 실패: {e}")
                cand.score = -float("inf")
                try:
                    set_env_state(env, saved_state)
                except Exception:
                    pass

            finally:
                try:
                    set_env_state(env, saved_state)
                except Exception as e:
                    if verbose:
                        print(f"  [FD Score] 복원 실패: {e}")

        if verbose:
            scores = [f"{c.score:.4f}" for c in candidates]
            print(f"  [FD Score] 점수: {scores}")

        return candidates

    except Exception as e:
        print(f"[FD Score] 전체 실패: {e}")
        return candidates
