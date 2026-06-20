class PassInterceptionDetector:
    """Detects passes and interceptions using the reference repo's holder changes."""

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
            ):
                previous_team = player_assignment[previous_frame].get(
                    previous_holder,
                    -1,
                )
                current_team = player_assignment[frame].get(current_holder, -1)

                if previous_team == current_team and previous_team != -1:
                    passes[frame] = previous_team

        return passes

    def detect_interceptions(self, ball_acquisition, player_assignment):
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
            ):
                previous_team = player_assignment[previous_frame].get(
                    previous_holder,
                    -1,
                )
                current_team = player_assignment[frame].get(current_holder, -1)

                if previous_team != current_team and previous_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions

    def detect_events(self, ball_acquisition, player_assignment):
        return events_from_arrays(
            self.detect_passes(ball_acquisition, player_assignment),
            self.detect_interceptions(ball_acquisition, player_assignment),
            ball_acquisition,
        )


def summarize_events(events):
    return {
        "passes": sum(event["type"] == "pass" for event in events),
        "interceptions": sum(event["type"] == "interception" for event in events),
    }


def events_from_arrays(passes, interceptions, ball_acquisition):
    events = []

    for frame_index, team_id in enumerate(passes):
        if team_id != -1:
            events.append(
                _array_event("pass", frame_index, team_id, ball_acquisition)
            )

    for frame_index, team_id in enumerate(interceptions):
        if team_id != -1:
            events.append(
                _array_event("interception", frame_index, team_id, ball_acquisition)
            )

    return sorted(events, key=lambda event: event["frame_index"])


def _array_event(event_type, frame_index, team_id, ball_acquisition):
    return {
        "type": event_type,
        "frame_index": frame_index,
        "from_player_id": _previous_holder(ball_acquisition, frame_index),
        "to_player_id": ball_acquisition[frame_index],
        "from_team_id": team_id if event_type == "pass" else None,
        "to_team_id": team_id,
        "release_frame": None,
        "catch_frame": frame_index,
        "gap_frames": None,
    }


def _previous_holder(ball_acquisition, frame_index):
    for index in range(frame_index - 1, -1, -1):
        player_id = ball_acquisition[index]
        if player_id is not None and player_id != -1:
            return player_id

    return None


PassAndInterceptionDetector = PassInterceptionDetector
