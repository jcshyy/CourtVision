import colorsys
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path

import cv2

from backend.app.utils import load_cache, save_cache


logger = logging.getLogger(__name__)
ASSIGNMENT_ALGORITHM_VERSION = "v11"
MINIMUM_PROTOTYPE_MARGIN_RATIO = 0.1
MINIMUM_TORSO_AREA_FRACTION = 900 / (1280 * 720)
MINIMUM_TRACK_OBSERVATION_AGREEMENT = 0.8


class NeedsTeamColorsError(RuntimeError):
    """Actionable automatic-discovery failure for CLI/UI callers."""

    def __init__(self, reason, discovery_confidence=None):
        self.result = {
            "status": "needs_team_colors",
            "reason": reason,
            "discovery_confidence": discovery_confidence,
            "message": (
                "Automatic team discovery was not confident enough. Provide "
                "both teams' primary jersey colors with --team-1-color and "
                "--team-2-color (for example, #FFFFFF and #C8102E)."
            ),
        }
        super().__init__(self.result["message"])


class TeamAssigner:
    """Assign players using discovered jersey colors or configured CLIP labels."""

    def __init__(
        self,
        team_1_description=None,
        team_2_description=None,
        team_1_color=None,
        team_2_color=None,
        model_name="patrickjohncyh/fashion-clip",
        vote_window_size=5,
        initial_observations=3,
    ):
        if (team_1_description is None) != (team_2_description is None):
            raise ValueError("Both team descriptions must be provided together")
        if (team_1_color is None) != (team_2_color is None):
            raise ValueError("Both team jersey colors must be provided together")
        if team_1_description is not None and team_1_color is not None:
            raise ValueError("Use either team descriptions or jersey colors, not both")
        if vote_window_size < 1:
            raise ValueError("vote_window_size must be at least 1")
        if initial_observations < 1:
            raise ValueError("initial_observations must be at least 1")

        normalized_colors = None
        if team_1_color is not None:
            normalized_colors = (
                normalize_jersey_color(team_1_color),
                normalize_jersey_color(team_2_color),
            )
            prototypes = tuple(_color_prototype(color) for color in normalized_colors)
            unusable_colors = [
                color
                for color, prototype in zip(normalized_colors, prototypes)
                if prototype is None
            ]
            if unusable_colors:
                raise ValueError(
                    "Team jersey colors must be bright enough to produce a usable "
                    f"jersey prototype; rejected: {', '.join(unusable_colors)}"
                )
            if _color_distance(*prototypes) < 30:
                raise ValueError("Team jersey colors must be sufficiently distinct")
            self.team_colors = {1: prototypes[0], 2: prototypes[1]}
        else:
            self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_votes = {}
        self.vote_window_size = vote_window_size
        self.initial_observations = initial_observations
        self.team_1_class_name = team_1_description
        self.team_2_class_name = team_2_description
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.use_discovered_colors = team_1_description is None
        self.assignment_mode = (
            "user_colors" if normalized_colors is not None else "automatic"
        )
        self.normalized_team_colors = normalized_colors
        cache_identity = {
            "algorithm_version": ASSIGNMENT_ALGORITHM_VERSION,
            "assignment_mode": self.assignment_mode,
            "team_colors": list(normalized_colors) if normalized_colors else None,
        }
        self.assignment_metadata = dict(cache_identity)
        self.assignment_metadata["discovery_confidence"] = None
        self.discovery_result = None
        identity = json.dumps(
            cache_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        identity_digest = hashlib.sha256(identity).hexdigest()[:12]
        self.cache_filename = (
            f"player_assignment_{ASSIGNMENT_ALGORITHM_VERSION}_"
            f"{self.assignment_mode}_{identity_digest}.pkl"
        )
        self.metadata_filename = (
            f"team_assignment_{ASSIGNMENT_ALGORITHM_VERSION}_"
            f"{self.assignment_mode}_{identity_digest}.json"
        )

    def load_model(self):
        if self.model is not None and self.processor is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Configured text-label team assignment requires the "
                "'transformers' package. Install backend/requirements.txt or "
                "use automatic jersey-color discovery."
            ) from error

        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def get_player_color(self, frame, bbox):
        crop = _crop_player(frame, bbox)
        if crop is None:
            return self.team_1_class_name

        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(
            text=classes,
            images=crop,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)
        return classes[probabilities.argmax(dim=1)[0]]

    def get_player_jersey_color(self, frame, bbox):
        return _jersey_color(frame, bbox)

    def get_player_team(self, frame, player_bbox, player_id, refresh=False):
        if player_id in self.player_team_dict and not refresh:
            return self.player_team_dict[player_id]

        if self.use_discovered_colors:
            player_color = self.get_player_jersey_color(frame, player_bbox)
            observed_team = _confident_nearest_team(player_color, self.team_colors)
        else:
            player_color = self.get_player_color(frame, player_bbox)
            observed_team = 1 if player_color == self.team_1_class_name else 2

        votes = self.player_team_votes.setdefault(player_id, [])
        if observed_team is not None:
            votes.append(observed_team)
            del votes[:-self.vote_window_size]

        current_team = self.player_team_dict.get(player_id)
        team_id = _majority_team(votes, current_team)
        if team_id is None:
            team_id = -1
        if current_team is not None and team_id != current_team:
            logger.info(
                "Player %s team changed from %s to %s after votes %s",
                player_id,
                current_team,
                team_id,
                votes,
            )
        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(
        self,
        video_frames,
        player_tracks,
        read_from_cache=False,
        cache_path: str | Path | None = None,
    ):
        player_assignment = (
            load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        )
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            metadata_path = Path(cache_path).with_name(self.metadata_filename)
            if metadata_path.exists():
                try:
                    cached_metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    self.assignment_metadata["discovery_confidence"] = (
                        cached_metadata.get("discovery_confidence")
                    )
                except (OSError, json.JSONDecodeError):
                    logger.warning(
                        "Could not restore assignment metadata from %s",
                        metadata_path,
                    )
            return player_assignment

        if self.use_discovered_colors:
            if self.assignment_mode == "automatic":
                discovery = _discover_team_colors_result(video_frames, player_tracks)
                self.discovery_result = discovery
                self.assignment_metadata["discovery_confidence"] = discovery["confidence"]
                if discovery["status"] != "confident":
                    logger.warning(
                        "Automatic team discovery is uncertain (%s): %s",
                        discovery["reason"],
                        discovery["confidence"],
                    )
                    raise NeedsTeamColorsError(
                        discovery["reason"],
                        discovery["confidence"],
                    )
                discovered_colors = discovery["prototypes"]
                self.team_colors = {1: discovered_colors[0], 2: discovered_colors[1]}
                logger.info(
                    "Discovered team jersey colors: %s; confidence=%s",
                    self.team_colors,
                    discovery["confidence"],
                )
            else:
                logger.info(
                    "Using validated jersey-color guidance: %s",
                    self.normalized_team_colors,
                )
        else:
            self.load_model()

        self._bootstrap_player_teams(video_frames, player_tracks)
        player_assignment = []

        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num],
                    track["bbox"],
                    player_id,
                )
                player_assignment[frame_num][player_id] = team

        if cache_path:
            save_cache(cache_path, player_assignment)

        return player_assignment

    def _bootstrap_player_teams(self, video_frames, player_tracks):
        """Assign each offline track from all of its usable jersey observations."""
        if not self.use_discovered_colors:
            return

        observations = {}
        for frame, frame_tracks in zip(video_frames, player_tracks):
            for player_id, track in frame_tracks.items():
                player_observations = observations.setdefault(player_id, [])
                color = self.get_player_jersey_color(frame, track["bbox"])
                if _confident_nearest_team(color, self.team_colors) is not None:
                    player_observations.append(color)

        for player_id, colors in observations.items():
            if not colors:
                continue

            representative_color = _median_color(colors)
            team_id = _confident_nearest_team(representative_color, self.team_colors)
            if team_id is None:
                continue
            self.player_team_dict[player_id] = team_id
            self.player_team_votes[player_id] = [team_id]

    def assign_teams_across_frames(
        self,
        frames,
        player_tracks,
        read_from_cache=False,
        cache_path: str | Path | None = None,
        sample_every=None,
    ):
        return self.get_player_teams_across_frames(
            frames,
            player_tracks,
            read_from_cache=read_from_cache,
            cache_path=cache_path,
        )


def _crop_player(frame, bbox):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Configured text-label team assignment requires Pillow. Install "
            "backend/requirements.txt or use automatic jersey-color discovery."
        ) from error

    image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    if image.size == 0:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def _discover_team_colors(
    video_frames,
    player_tracks,
    max_samples=100,
    minimum_player_observations=2,
):
    result = _discover_team_colors_result(
        video_frames,
        player_tracks,
        max_samples=max_samples,
        minimum_player_observations=minimum_player_observations,
    )
    return result["prototypes"] if result["status"] == "confident" else None


def _discover_team_colors_result(
    video_frames,
    player_tracks,
    max_samples=100,
    minimum_player_observations=2,
):
    """Return prototypes plus evidence used to accept or reject discovery."""
    player_colors = {}
    candidate_observations = 0
    accepted_observations = 0
    frame_step = max(1, len(video_frames) // 20)

    for frame_num in range(0, min(len(video_frames), len(player_tracks)), frame_step):
        for player_id, track in player_tracks[frame_num].items():
            candidate_observations += 1
            color = _jersey_color(video_frames[frame_num], track["bbox"])
            if color is not None:
                player_colors.setdefault(player_id, []).append(color)
                accepted_observations += 1
            if accepted_observations >= max_samples:
                break
        if accepted_observations >= max_samples:
            break

    eligible_player_colors = {
        player_id: colors
        for player_id, colors in player_colors.items()
        if len(colors) >= minimum_player_observations
    }
    representatives = {
        player_id: _median_color(colors)
        for player_id, colors in eligible_player_colors.items()
    }
    prototypes = _cluster_team_colors(list(representatives.values()))
    confidence = {
        "candidate_observation_count": candidate_observations,
        "accepted_observation_count": accepted_observations,
        "accepted_observation_fraction": round(
            accepted_observations / max(1, candidate_observations),
            3,
        ),
        "observed_track_count": len(player_colors),
        "eligible_track_count": len(representatives),
        "prototype_separation": None,
        "cluster_support": [],
        "track_observation_agreement": None,
    }
    if prototypes is None:
        return {
            "status": "needs_team_colors",
            "reason": "insufficient_distinct_team_prototypes",
            "prototypes": None,
            "confidence": confidence,
        }

    prototype_map = {1: prototypes[0], 2: prototypes[1]}
    representative_teams = {
        player_id: _nearest_team(color, prototype_map)
        for player_id, color in representatives.items()
    }
    support = [
        sum(team_id == expected for team_id in representative_teams.values())
        for expected in (1, 2)
    ]
    agreement_votes = [
        _nearest_team(color, prototype_map) == representative_teams[player_id]
        for player_id, colors in eligible_player_colors.items()
        for color in colors
    ]
    confidence.update(
        {
            "prototype_separation": round(
                _color_distance(prototypes[0], prototypes[1]),
                3,
            ),
            "cluster_support": support,
            "track_observation_agreement": round(
                sum(agreement_votes) / max(1, len(agreement_votes)),
                3,
            ),
        }
    )
    if min(support) < 2:
        return {
            "status": "needs_team_colors",
            "reason": "insufficient_cluster_support",
            "prototypes": None,
            "confidence": confidence,
        }
    if confidence["track_observation_agreement"] < MINIMUM_TRACK_OBSERVATION_AGREEMENT:
        return {
            "status": "needs_team_colors",
            "reason": "unstable_track_observations",
            "prototypes": None,
            "confidence": confidence,
        }
    return {
        "status": "confident",
        "reason": None,
        "prototypes": prototypes,
        "confidence": confidence,
    }


def _jersey_color(frame, bbox):
    return _jersey_observation(frame, bbox)["feature"]


def _jersey_observation(frame, bbox):
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return {
            "feature": None,
            "accepted": False,
            "rejection_reason": "invalid_bbox",
            "bbox_width": width,
            "bbox_height": height,
        }

    frame_height, frame_width = frame.shape[:2]
    torso_x1 = max(0, x1 + width // 5)
    torso_x2 = min(frame_width, x2 - width // 5)
    torso_y1 = max(0, y1 + height // 5)
    torso_y2 = min(frame_height, y1 + height * 3 // 5)
    jersey = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if jersey.size == 0:
        return {
            "feature": None,
            "accepted": False,
            "rejection_reason": "empty_torso_crop",
            "bbox_width": width,
            "bbox_height": height,
            "torso_width": max(0, torso_x2 - torso_x1),
            "torso_height": max(0, torso_y2 - torso_y1),
        }

    feature_details = _jersey_feature_details(jersey.reshape(-1, 3).tolist())
    feature = feature_details["feature"]
    expected_torso_area = max(1, (width * 3 // 5) * (height * 2 // 5))
    actual_torso_area = jersey.shape[0] * jersey.shape[1]
    minimum_torso_area = max(
        1,
        math.ceil(frame_width * frame_height * MINIMUM_TORSO_AREA_FRACTION),
    )
    edge_clipped = x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height
    observation = {
        "feature": feature,
        "accepted": feature is not None,
        "rejection_reason": None if feature is not None else "no_visible_pixels",
        "bbox_width": width,
        "bbox_height": height,
        "torso_width": jersey.shape[1],
        "torso_height": jersey.shape[0],
        "torso_area": actual_torso_area,
        "minimum_torso_area": minimum_torso_area,
        "torso_frame_area_fraction": actual_torso_area / (frame_width * frame_height),
        "crop_fraction": actual_torso_area / expected_torso_area,
        "edge_clipped": edge_clipped,
        "blur_variance": _blur_variance(jersey),
    }
    observation.update(feature_details)
    if edge_clipped:
        observation.update(
            feature=None,
            accepted=False,
            rejection_reason="edge_clipped",
        )
    elif actual_torso_area < minimum_torso_area:
        observation.update(
            feature=None,
            accepted=False,
            rejection_reason="torso_too_small",
        )
    return observation


def _jersey_feature(pixels):
    return _jersey_feature_details(pixels)["feature"]


def _jersey_feature_details(pixels):
    hsv = _bgr_pixels_to_hsv(pixels)
    # Ignore deep shadows/background and common skin hues. Pure reds remain
    # eligible because their hue wraps around zero rather than the skin range.
    visible = [color for color in hsv if color[2] >= 0.15]
    filtered = [
        color
        for color in visible
        if not (
            5.0 <= color[0] <= 25.0
            and 0.15 <= color[1] <= 0.75
            and color[2] <= 0.95
        )
    ]
    minimum_pixels = max(8, len(hsv) // 10)
    skin_filtered_pixels = len(filtered)
    used_visible_fallback = len(filtered) < minimum_pixels
    if len(filtered) < minimum_pixels:
        filtered = visible
    if len(filtered) == 0:
        return {
            "feature": None,
            "total_pixels": len(hsv),
            "visible_pixels": len(visible),
            "filtered_pixels": 0,
            "skin_filtered_pixels": skin_filtered_pixels,
            "minimum_pixels": minimum_pixels,
            "used_visible_fallback": used_visible_fallback,
            "visible_fraction": len(visible) / max(1, len(hsv)),
        }

    hue_radians = [math.radians(color[0]) for color in filtered]
    saturation = [color[1] for color in filtered]
    feature = (
        statistics.median(
            math.cos(hue) * saturation[index]
            for index, hue in enumerate(hue_radians)
        ),
        statistics.median(
            math.sin(hue) * saturation[index]
            for index, hue in enumerate(hue_radians)
        ),
        statistics.median(saturation),
        statistics.median(color[2] for color in filtered),
    )
    return {
        "feature": tuple(int(round(value * 255)) for value in feature),
        "total_pixels": len(hsv),
        "visible_pixels": len(visible),
        "filtered_pixels": len(filtered),
        "skin_filtered_pixels": skin_filtered_pixels,
        "minimum_pixels": minimum_pixels,
        "used_visible_fallback": used_visible_fallback,
        "visible_fraction": len(visible) / max(1, len(hsv)),
        "filtered_fraction": len(filtered) / max(1, len(hsv)),
    }


def _bgr_pixels_to_hsv(pixels):
    hsv = []
    for blue, green, red in pixels:
        hue, saturation, value = colorsys.rgb_to_hsv(
            red / 255.0,
            green / 255.0,
            blue / 255.0,
        )
        hsv.append((hue * 360.0, saturation, value))
    return hsv


def normalize_jersey_color(value):
    """Normalize a user jersey color to uppercase #RRGGBB."""
    if not isinstance(value, str):
        raise ValueError("Jersey colors must use #RRGGBB format")
    normalized = value.strip().upper()
    if not normalized.startswith("#"):
        normalized = f"#{normalized}"
    if len(normalized) != 7 or any(
        character not in "0123456789ABCDEF" for character in normalized[1:]
    ):
        raise ValueError("Jersey colors must use #RRGGBB format")
    return normalized


def _color_prototype(normalized_color):
    red = int(normalized_color[1:3], 16)
    green = int(normalized_color[3:5], 16)
    blue = int(normalized_color[5:7], 16)
    return _jersey_feature([(blue, green, red)])


def _confident_nearest_team(
    color,
    team_colors,
    minimum_margin_ratio=MINIMUM_PROTOTYPE_MARGIN_RATIO,
):
    """Return a team only when the observation clearly favors one prototype."""
    distances = _team_distances(color, team_colors)
    if len(distances) != 2 or any(value is None for value in distances.values()):
        return None

    team_ids = list(distances)
    prototype_separation = _color_distance(
        team_colors[team_ids[0]],
        team_colors[team_ids[1]],
    )
    if prototype_separation <= 0:
        return None

    margin = abs(distances[team_ids[0]] - distances[team_ids[1]])
    if margin / prototype_separation < minimum_margin_ratio:
        return None
    return min(distances, key=distances.get)


def _blur_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _cluster_team_colors(colors, minimum_separation=30):
    if len(colors) < 4:
        return None

    first = colors[0]
    second = max(colors[1:], key=lambda color: _color_distance(first, color))
    if _color_distance(first, second) < minimum_separation:
        return None

    centers = [first, second]
    for _ in range(10):
        groups = [[], []]
        for color in colors:
            first_distance = _color_distance(color, centers[0])
            second_distance = _color_distance(color, centers[1])
            groups[0 if first_distance <= second_distance else 1].append(color)
        if not all(groups):
            return None

        new_centers = [_mean_color(group) for group in groups]
        if new_centers == centers:
            break
        centers = new_centers

    if _color_distance(centers[0], centers[1]) < minimum_separation:
        return None
    return tuple(centers)


def _nearest_team(color, team_colors):
    if color is None:
        return None
    return min(
        team_colors,
        key=lambda team_id: _color_distance(color, team_colors[team_id]),
    )


def _team_distances(color, team_colors):
    if color is None:
        return {team_id: None for team_id in team_colors}
    return {
        team_id: _color_distance(color, prototype)
        for team_id, prototype in team_colors.items()
    }


def _majority_team(votes, current_team=None):
    if not votes:
        return current_team
    counts = {team_id: votes.count(team_id) for team_id in set(votes)}
    highest_count = max(counts.values())
    leading_teams = {
        team_id for team_id, count in counts.items() if count == highest_count
    }
    if current_team in leading_teams:
        return current_team
    return min(leading_teams)


def _mean_color(colors):
    return tuple(
        int(round(sum(color[channel] for color in colors) / len(colors)))
        for channel in range(len(colors[0]))
    )


def _median_color(colors):
    return tuple(
        int(round(statistics.median(color[channel] for color in colors)))
        for channel in range(len(colors[0]))
    )


def _color_distance(first, second):
    return sum(
        (first[channel] - second[channel]) ** 2
        for channel in range(min(len(first), len(second)))
    ) ** 0.5
