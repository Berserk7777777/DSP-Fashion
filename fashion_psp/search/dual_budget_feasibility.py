"""Exact denoising AND decoded-reward feasibility, with no candidate data.

Fixed112-plus-optional-view-cap contract. This is NOT a weighted total
compute planner. The latest research comparison fixes denoising work112
and reports decoder/reward costs separately. See PERSONAL_RESEARCH_PROTOCOL.

Fixed initial trajectories, monotone positive populations. Removing particles
requires scoring ALL current live particles at that same step. Terminal always
scores all survivors, including a single survivor. No pruning/ranking policy.
Cheap latent measurements, if used by a future policy, have separate cost.
"""
from functools import lru_cache


@lru_cache(None)
def reachable_budgets(steps_left, count, views_left):
    """Bit k is1 iff exactly k future particle-steps can finish within views_left.

    steps_left includes the next denoising step. An observation with no removal
    is unnecessary for reachability and is omitted here; callers can spend its
    views explicitly via feasible_actions, which reserves the remaining work.
    """
    if steps_left < 1 or count < 1 or views_left < count:
        return 0
    if steps_left == 1:
        return 1 << count
    continuations = reachable_budgets(steps_left-1,count,views_left)
    if views_left >= count+1:
        for keep in range(1,count):
            continuations |= reachable_budgets(steps_left-1,keep,views_left-count)
    return continuations << count


def can_finish(steps_left,count,particle_steps_left,views_left):
    if particle_steps_left < 0:
        return False
    return bool((reachable_budgets(steps_left,count,views_left) >> particle_steps_left) & 1)


def feasible_actions(step,spent,count,views_spent,view_cap=14,horizon=28,total_budget=112):
    """Called after denoising the current step, BEFORE seeing its reward.

    wait means no scoring and no pruning at this step.
    observe_keep lists population counts allowed after paying count views NOW.
    At terminal, selection always leaves1; all count views must still be paid.
    An actual policy must choose to observe before accessing any scores, then
    choose a subset whose size is in observe_keep. Budget feasibility does not
    certify reward reliability or give a guarantee about the winning seed.
    """
    if not 1 <= step <= horizon or count < 1 or views_spent < 0:
        raise ValueError("Invalid state")
    remaining=total_budget-spent
    views_left=view_cap-views_spent
    if step == horizon:
        return {"wait":False,"observe_keep":[1] if remaining==0 and views_left>=count else [],
                "terminal":True}
    h=horizon-step
    wait=can_finish(h,count,remaining,views_left)
    allowed=[m for m in range(1,count+1) if can_finish(h,m,remaining,views_left-count)]
    return {"wait":wait,"observe_keep":allowed,"terminal":False}
