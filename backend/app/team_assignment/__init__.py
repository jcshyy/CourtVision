__all__ = ["NeedsTeamColorsError", "TeamAssigner", "normalize_jersey_color"]


def __getattr__(name):
    if name in __all__:
        from .team_assigner import (
            NeedsTeamColorsError,
            TeamAssigner,
            normalize_jersey_color,
        )

        return {
            "NeedsTeamColorsError": NeedsTeamColorsError,
            "TeamAssigner": TeamAssigner,
            "normalize_jersey_color": normalize_jersey_color,
        }[name]
    raise AttributeError(name)
