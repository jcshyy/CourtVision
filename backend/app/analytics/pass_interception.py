class PassInterceptionDetector:
    """Detects passes and interceptions using the reference repo's holder changes."""

    def __init__(
        self,
        max_holder_gap_frames=None,
        team_lookup_frames=30,
        transient_control_frames=8,
    ):
        self.max_holder_gap_frames = max_holder_gap_frames
        self.team_lookup_frames = max(1, int(team_lookup_frames))
        self.transient_control_frames = max(1, int(transient_control_frames))

    def clean_transient_control_chains(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
    ):
        """Remove implausibly brief control between a turnover and same-team catch."""
        cleaned = list(ball_acquisition)
        passes = self.detect_passes(cleaned, player_assignment)
        interceptions = self.detect_interceptions(
            cleaned,
            player_assignment,
            holder_states=holder_states,
        )
        for interception_frame, receiving_team in enumerate(interceptions):
            if receiving_team not in (1, 2):
                continue
            intermediate_holder = cleaned[interception_frame]
            end = min(
                len(cleaned),
                interception_frame + self.transient_control_frames + 1,
            )
            for catch_frame in range(interception_frame + 1, end):
                if passes[catch_frame] != receiving_team:
                    continue
                previous_holder, _ = _previous_holder_info(cleaned, catch_frame)
                if previous_holder != intermediate_holder:
                    continue
                for frame in range(interception_frame, catch_frame):
                    if cleaned[frame] == intermediate_holder:
                        cleaned[frame] = -1
                break
        return cleaned

    def detect_passes(self, ball_acquisition, player_assignment):
        passes = [-1] * len(ball_acquisition)
        previous_holder = -1
        previous_frame = -1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                previous_holder = ball_acquisition[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_acquisition[frame]

            if (
                previous_holder != -1
                and current_holder != -1
                and previous_holder != current_holder
                and self._gap_is_valid(frame, previous_frame)
            ):
                previous_team = self._holder_team(
                    player_assignment, previous_holder, previous_frame,
                )
                current_team = self._holder_team(
                    player_assignment, current_holder, frame,
                )

                if previous_team == current_team and previous_team != -1:
                    passes[frame] = previous_team

        return passes

    def detect_interceptions(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
    ):
        interceptions = [-1] * len(ball_acquisition)
        previous_holder = -1
        previous_frame = -1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                previous_holder = ball_acquisition[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_acquisition[frame]

            if (
                previous_holder != -1
                and current_holder != -1
                and previous_holder != current_holder
                and self._gap_is_valid(frame, previous_frame)
                and self._interception_evidence_is_valid(
                    holder_states,
                    previous_frame,
                    frame,
                )
            ):
                previous_team = self._holder_team(
                    player_assignment, previous_holder, previous_frame,
                )
                current_team = self._holder_team(
                    player_assignment, current_holder, frame,
                )

                if previous_team != current_team and previous_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions

    def _gap_is_valid(self, current_frame, previous_frame):
        return (
            self.max_holder_gap_frames is None
            or current_frame - previous_frame - 1 <= self.max_holder_gap_frames
        )

    def _holder_team(self, player_assignment, holder_id, frame):
        start = max(0, frame - self.team_lookup_frames + 1)
        for lookup_frame in range(frame, start - 1, -1):
            team_id = player_assignment[lookup_frame].get(holder_id)
            if team_id in (1, 2):
                return team_id
        return -1

    @staticmethod
    def _interception_evidence_is_valid(
        holder_states,
        previous_frame,
        current_frame,
    ):
        if holder_states is None:
            return True
        return any(
            state.get("state") == "loose"
            and state.get("ball_confidence") is not None
            for state in holder_states[previous_frame + 1 : current_frame]
        )

    def detect_events(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
    ):
        return events_from_arrays(
            self.detect_passes(ball_acquisition, player_assignment),
            self.detect_interceptions(
                ball_acquisition,
                player_assignment,
                holder_states=holder_states,
            ),
            ball_acquisition,
            player_assignment,
        )


def summarize_events(events):
    return {
        "passes": sum(event["type"] == "pass" for event in events),
        "interceptions": sum(event["type"] == "interception" for event in events),
    }


def events_from_arrays(
    passes,
    interceptions,
    ball_acquisition,
    player_assignment=None,
):
    events = []

    for frame_index, team_id in enumerate(passes):
        if team_id != -1:
            events.append(
                _array_event(
                    "pass",
                    frame_index,
                    team_id,
                    ball_acquisition,
                    player_assignment,
                )
            )

    for frame_index, team_id in enumerate(interceptions):
        if team_id != -1:
            events.append(
                _array_event(
                    "interception",
                    frame_index,
                    team_id,
                    ball_acquisition,
                    player_assignment,
                )
            )

    return sorted(events, key=lambda event: event["frame_index"])


def _array_event(
    event_type,
    frame_index,
    team_id,
    ball_acquisition,
    player_assignment=None,
):
    previous_holder, previous_frame = _previous_holder_info(
        ball_acquisition,
        frame_index,
    )
    previous_team = None
    if (
        player_assignment is not None
        and previous_frame is not None
        and previous_frame < len(player_assignment)
    ):
        previous_team = player_assignment[previous_frame].get(previous_holder)
    return {
        "type": event_type,
        "frame_index": frame_index,
        "from_player_id": previous_holder,
        "to_player_id": ball_acquisition[frame_index],
        "from_team_id": team_id if event_type == "pass" else previous_team,
        "to_team_id": team_id,
        "release_frame": previous_frame,
        "catch_frame": frame_index,
        "gap_frames": (
            frame_index - previous_frame - 1
            if previous_frame is not None
            else None
        ),
    }


def _previous_holder(ball_acquisition, frame_index):
    return _previous_holder_info(ball_acquisition, frame_index)[0]


def _previous_holder_info(ball_acquisition, frame_index):
    for index in range(frame_index - 1, -1, -1):
        player_id = ball_acquisition[index]
        if player_id is not None and player_id != -1:
            return player_id, index

    return None, None


PassAndInterceptionDetector = PassInterceptionDetector
