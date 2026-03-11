from __future__ import annotations

from gpt_oss_interp.benchmarks.tasks import all_tasks


###############################################################################
#
# Canonical case pools
#
###############################################################################

# Original gpt-oss-supported clean pool:
# main_analysis_soft intersected with local_support, excluding the
# Harmony-tail pathology induction_002.
LEGACY_GPT_OSS_LOCAL_SUPPORT_CASE_IDS = (
    "caps_002",
    "caps_003",
    "coref_002",
    "coref_003",
    "coref_004",
    "induction_001",
    "induction_003",
    "induction_004",
)

# Historical 9-case soft-main set used by the original soft-main configs and
# run artifacts. This intentionally includes induction_002 because those
# configs reproduce the earlier benchmark thread rather than the newer bridge
# pool.
LEGACY_SOFT_MAIN_CASE_IDS = (
    "caps_002",
    "caps_003",
    "coref_002",
    "coref_003",
    "coref_004",
    "induction_001",
    "induction_002",
    "induction_003",
    "induction_004",
)

# Smaller-model screened additions accepted by the Gemma 3 1B fallback screen.
SCREENED_BRIDGE_ADDITION_CASE_IDS = (
    "caps_005",
    "caps_006",
    "caps_007",
    "coref_010",
    "induction_008",
    "induction_009",
)

# Cases that should stay out of the working bridge pool even if they are
# correct under a coarse benchmark filter.
BRIDGE_POOL_EXCLUDED_CASE_IDS = (
    "induction_002",
)

PROVISIONAL_BRIDGE_POOL_CASE_IDS = tuple(
    sorted(
        set(LEGACY_GPT_OSS_LOCAL_SUPPORT_CASE_IDS)
        | set(SCREENED_BRIDGE_ADDITION_CASE_IDS)
        - set(BRIDGE_POOL_EXCLUDED_CASE_IDS)
    )
)


###############################################################################
#
# Helpers
#
###############################################################################


def filter_tasks_by_case_ids(case_ids: set[str] | tuple[str, ...] | list[str]):
    selected = set(case_ids)
    tasks = []
    for task in all_tasks():
        filtered_cases = [case for case in task.cases if case.case_id in selected]
        if filtered_cases:
            task.cases = filtered_cases
            tasks.append(task)
    return tasks
