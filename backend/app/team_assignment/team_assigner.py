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
ASSIGNMENT_ALGORITHM_VERSION = "v17_evidence_arbitration"
MINIMUM_PROTOTYPE_MARGIN_RATIO = 0.1
MAXIMUM_PROTOTYPE_DISTANCE_RATIO = 0.75
MINIMUM_TORSO_AREA_FRACTION = 900 / (1280 * 720)
MINIMUM_TRACK_OBSERVATION_AGREEMENT = 0.8
MINIMUM_TRACK_CONFIDENT_OBSERVATIONS = 3
MINIMUM_TRACK_TEAM_AGREEMENT = 0.8
MINIMUM_TRACK_WEIGHT_SHARE = 0.8
MAXIMUM_TRACK_EVIDENCE_SAMPLES = 24
MAXIMUM_UNKNOWN_OBSERVATION_FRACTION = 0.15
MINIMUM_COLOR_CONFLICT_EVIDENCE = 6
MINIMUM_COLOR_CONFLICT_WEIGHT_SHARE = 0.65


class NeedsTeamColorsError(RuntimeError):
    """Actionable automatic-discovery failure for CLI/UI callers."""

    def __init__(self, reason, discovery_confidence=None):
        self.result = {
            "status": "needs_team_colors",
            "reason": reason,
            "discovery_confidence": discovery_confidence,
            "message": (
                "Color analysis and FashionCLIP could not confidently separate "
                "the two teams. Add both teams' primary jersey colors for the "
                "most accurate team-level results, or continue with unresolved "
                "players marked unknown."
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
        tracking_algorithm_version=None,
        allow_uncertain_teams=False,
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
        self.model_device = None
        self.use_discovered_colors = team_1_description is None
        self.assignment_mode = (
            "user_colors" if normalized_colors is not None else "automatic"
        )
        self.allow_uncertain_teams = bool(allow_uncertain_teams)
        self.normalized_team_colors = normalized_colors
        cache_identity = {
            "algorithm_version": ASSIGNMENT_ALGORITHM_VERSION,
            "assignment_mode": self.assignment_mode,
            "team_colors": list(normalized_colors) if normalized_colors else None,
            "allow_uncertain_teams": self.allow_uncertain_teams,
        }
        if tracking_algorithm_version is not None:
            cache_identity["player_tracking_algorithm_version"] = str(
                tracking_algorithm_version
            )
        self.assignment_metadata = dict(cache_identity)
        self.assignment_metadata["discovery_confidence"] = None
        self.assignment_metadata["track_assignments"] = {}
        self.discovery_result = None
        self._last_jersey_observation = None
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
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Configured text-label team assignment requires the "
                "'transformers' package. Install backend/requirements.txt or "
                "use automatic jersey-color discovery."
            ) from error

        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            model_source = "local cache"
        except OSError:
            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model_source = "model registry"
        self.model = model.to(self.model_device)
        self.model.eval()
        self.processor = processor
        logger.info(
            "Loaded FashionCLIP from %s on %s",
            model_source,
            self.model_device,
        )

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

        inputs = {
            name: tensor.to(self.model_device)
            for name, tensor in inputs.items()
        }
        outputs = self.model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)
        return classes[probabilities.argmax(dim=1)[0]]

    def get_player_jersey_color(self, frame, bbox):
        self._last_jersey_observation = _jersey_observation(frame, bbox)
        return self._last_jersey_observation["feature"]

    def get_player_jersey_observation(self, frame, bbox):
        """Return detailed evidence while preserving the color-only public method."""
        self._last_jersey_observation = None
        feature = self.get_player_jersey_color(frame, bbox)
        observation = self._last_jersey_observation
        if observation is None or observation.get("feature") != feature:
            return {
                "feature": feature,
                "accepted": feature is not None,
                "rejection_reason": None if feature is not None else "no_feature",
                "quality_score": 1.0 if feature is not None else 0.0,
            }
        return observation

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
                    self.assignment_metadata["track_assignments"] = (
                        cached_metadata.get("track_assignments", {})
                    )
                    self.assignment_metadata["fashion_clip_fallback"] = (
                        cached_metadata.get("fashion_clip_fallback")
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
                    fashion_clip = self._discover_teams_with_fashion_clip(
                        video_frames,
                        player_tracks,
                    )
                    combined_confidence = {
                        **discovery["confidence"],
                        "fallback_used": "fashion_clip",
                        "fashion_clip": fashion_clip.get("confidence", {}),
                    }
                    self.assignment_metadata["discovery_confidence"] = combined_confidence
                    self.assignment_metadata["fashion_clip_fallback"] = fashion_clip
                    if fashion_clip["status"] != "confident":
                        logger.warning(
                            "FashionCLIP fallback is uncertain (%s): %s",
                            fashion_clip["reason"],
                            fashion_clip.get("confidence", {}),
                        )
                        if not self.allow_uncertain_teams:
                            raise NeedsTeamColorsError(
                                discovery["reason"],
                                combined_confidence,
                            )
                        self._record_uncertain_continuation(
                            discovery["reason"],
                            combined_confidence,
                        )
                        self.assignment_metadata["track_assignments"] = {
                            str(int(player_id)): {
                                "team_id": -1,
                                "reason": "uncertain_team_continuation",
                            }
                            for frame_tracks in player_tracks
                            for player_id in frame_tracks
                        }
                        player_assignment = [
                            {player_id: -1 for player_id in frame_tracks}
                            for frame_tracks in player_tracks
                        ]
                        self.assignment_metadata["unknown_observation_fraction"] = (
                            _unknown_assignment_fraction(player_tracks, {})
                        )
                        if cache_path:
                            save_cache(cache_path, player_assignment)
                        return player_assignment
                    for player_id, team_id in fashion_clip["track_assignments"].items():
                        self.player_team_dict[player_id] = team_id
                        self.player_team_votes[player_id] = [team_id]
                    self.assignment_metadata["track_assignments"] = {
                        str(int(player_id)): {
                            "team_id": team_id,
                            "reason": "fashion_clip_cluster",
                        }
                        for player_id, team_id in fashion_clip["track_assignments"].items()
                    }
                    logger.info(
                        "FashionCLIP resolved uncertain jersey colors on %s: %s",
                        fashion_clip.get("device", self.model_device),
                        fashion_clip["confidence"],
                    )
                    player_assignment = [
                        {
                            player_id: self.player_team_dict.get(player_id, -1)
                            for player_id in frame_tracks
                        }
                        for frame_tracks in player_tracks
                    ]
                    unknown_fraction = _unknown_assignment_fraction(
                        player_tracks,
                        self.player_team_dict,
                    )
                    self.assignment_metadata["unknown_observation_fraction"] = (
                        unknown_fraction
                    )
                    if unknown_fraction > MAXIMUM_UNKNOWN_OBSERVATION_FRACTION:
                        combined_confidence["unknown_observation_fraction"] = (
                            unknown_fraction
                        )
                        if not self.allow_uncertain_teams:
                            raise NeedsTeamColorsError(
                                "too_many_unknown_team_observations",
                                combined_confidence,
                            )
                        self._record_uncertain_continuation(
                            "too_many_unknown_team_observations",
                            combined_confidence,
                        )
                    if cache_path:
                        save_cache(cache_path, player_assignment)
                    return player_assignment
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
        if self.assignment_mode == "automatic":
            self._resolve_unknown_tracks_with_fashion_clip(
                video_frames,
                player_tracks,
            )
            unknown_fraction = _unknown_assignment_fraction(
                player_tracks,
                self.player_team_dict,
            )
            self.assignment_metadata["unknown_observation_fraction"] = unknown_fraction
            if unknown_fraction > MAXIMUM_UNKNOWN_OBSERVATION_FRACTION:
                confidence = {
                    **(self.assignment_metadata.get("discovery_confidence") or {}),
                    "unknown_observation_fraction": unknown_fraction,
                    "individual_fashion_clip": self.assignment_metadata.get(
                        "individual_fashion_clip_fallback",
                        {},
                    ),
                }
                if not self.allow_uncertain_teams:
                    raise NeedsTeamColorsError(
                        "too_many_unknown_team_observations",
                        confidence,
                    )
                self._record_uncertain_continuation(
                    "too_many_unknown_team_observations",
                    confidence,
                )
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

    def _record_uncertain_continuation(self, reason, confidence):
        self.assignment_metadata.update(
            {
                "proceeded_with_uncertain_teams": True,
                "uncertainty_reason": reason,
                "uncertainty_confidence": confidence,
                "uncertainty_warning": (
                    "Team colors were skipped after uncertain automatic assignment. "
                    "Unknown players remain unclassified and team possession, pass, "
                    "and interception totals may be inaccurate."
                ),
            }
        )
        logger.warning(
            "Continuing with uncertain team assignments (%s); team-level events may "
            "be inaccurate",
            reason,
        )

    def _discover_teams_with_fashion_clip(
        self,
        video_frames,
        player_tracks,
        max_samples_per_track=4,
        max_total_samples=128,
    ):
        """Cluster persistent player tracks using FashionCLIP jersey embeddings."""
        try:
            import torch

            self.load_model()
            samples = []
            sample_track_ids = []
            per_track_counts = {}
            frame_step = max(1, len(video_frames) // 24)
            for frame_index in range(
                0,
                min(len(video_frames), len(player_tracks)),
                frame_step,
            ):
                for player_id, track in player_tracks[frame_index].items():
                    if per_track_counts.get(player_id, 0) >= max_samples_per_track:
                        continue
                    crop = _crop_player_jersey(
                        video_frames[frame_index],
                        track["bbox"],
                    )
                    if crop is None:
                        continue
                    samples.append(crop)
                    sample_track_ids.append(player_id)
                    per_track_counts[player_id] = per_track_counts.get(player_id, 0) + 1
                    if len(samples) >= max_total_samples:
                        break
                if len(samples) >= max_total_samples:
                    break

            eligible_tracks = {
                player_id
                for player_id, count in per_track_counts.items()
                if count >= 2
            }
            if len(eligible_tracks) < 4:
                return {
                    "status": "needs_team_colors",
                    "reason": "insufficient_fashion_clip_tracks",
                    "device": self.model_device,
                    "track_assignments": {},
                    "confidence": {
                        "sample_count": len(samples),
                        "eligible_track_count": len(eligible_tracks),
                    },
                }

            embeddings = []
            batch_size = 16
            for start in range(0, len(samples), batch_size):
                inputs = self.processor(
                    images=samples[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                )
                pixel_values = inputs["pixel_values"].to(self.model_device)
                with torch.inference_mode():
                    batch_embeddings = self.model.get_image_features(
                        pixel_values=pixel_values,
                    )
                    if hasattr(batch_embeddings, "pooler_output"):
                        batch_embeddings = batch_embeddings.pooler_output
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings,
                        dim=1,
                    )
                embeddings.extend(batch_embeddings.detach().cpu().tolist())

            filtered_embeddings = [
                embedding
                for embedding, player_id in zip(embeddings, sample_track_ids)
                if player_id in eligible_tracks
            ]
            filtered_track_ids = [
                player_id
                for player_id in sample_track_ids
                if player_id in eligible_tracks
            ]
            result = _cluster_fashion_clip_embeddings(
                filtered_embeddings,
                filtered_track_ids,
            )
            result["device"] = self.model_device
            return result
        except Exception as error:
            logger.exception("FashionCLIP fallback could not run")
            return {
                "status": "needs_team_colors",
                "reason": "fashion_clip_unavailable",
                "device": self.model_device,
                "track_assignments": {},
                "confidence": {"error": str(error)},
            }

    def _resolve_unknown_tracks_with_fashion_clip(self, video_frames, player_tracks):
        unknown_track_ids = {
            player_id
            for player_id, team_id in self.player_team_dict.items()
            if team_id not in (1, 2)
        }
        if not unknown_track_ids:
            return

        fashion_clip = self._discover_teams_with_fashion_clip(
            video_frames,
            player_tracks,
        )
        fallback_metadata = {
            "attempted_track_ids": sorted(int(value) for value in unknown_track_ids),
            "fashion_clip": fashion_clip,
            "resolved_track_ids": [],
            "color_evidence_overrides": [],
            "unresolved_conflicts": [],
        }
        self.assignment_metadata["individual_fashion_clip_fallback"] = fallback_metadata
        if fashion_clip["status"] != "confident":
            return

        mapping = _map_fashion_clip_clusters_to_teams(
            fashion_clip["track_assignments"],
            self.player_team_dict,
        )
        fallback_metadata["mapping"] = mapping
        if mapping["status"] != "confident":
            return

        track_metadata = self.assignment_metadata.get("track_assignments", {})
        for player_id in unknown_track_ids:
            cluster_id = fashion_clip["track_assignments"].get(player_id)
            fashion_team_id = mapping["cluster_to_team"].get(cluster_id)
            existing = track_metadata.get(str(int(player_id)), {})
            arbitration = _arbitrate_track_assignment(
                existing,
                fashion_team_id,
            )
            team_id = arbitration["team_id"]
            if team_id not in (1, 2):
                track_metadata[str(int(player_id))] = {
                    **existing,
                    "status": "unknown",
                    "team_id": None,
                    "reason": arbitration["reason"],
                    "original_reason": existing.get("reason"),
                    "fashion_clip_team_id": fashion_team_id,
                }
                fallback_metadata["unresolved_conflicts"].append(int(player_id))
                continue
            self.player_team_dict[player_id] = team_id
            self.player_team_votes[player_id] = [team_id]
            track_metadata[str(int(player_id))] = {
                **existing,
                "status": "assigned",
                "team_id": team_id,
                "reason": arbitration["reason"],
                "original_reason": existing.get("reason"),
                "fashion_clip_team_id": fashion_team_id,
            }
            fallback_metadata["resolved_track_ids"].append(int(player_id))
            if arbitration["reason"] == "weighted_color_over_fashion_clip_conflict":
                fallback_metadata["color_evidence_overrides"].append(int(player_id))

        self.assignment_metadata["track_assignments"] = track_metadata
        logger.info(
            "FashionCLIP resolved %d/%d individually uncertain tracks",
            len(fallback_metadata["resolved_track_ids"]),
            len(unknown_track_ids),
        )

    def _bootstrap_player_teams(self, video_frames, player_tracks):
        """Assign each offline track from all of its usable jersey observations."""
        if not self.use_discovered_colors:
            return

        observations = {}
        for frame_index, (frame, frame_tracks) in enumerate(
            zip(video_frames, player_tracks)
        ):
            for player_id, track in frame_tracks.items():
                player_observations = observations.setdefault(player_id, [])
                observation = self.get_player_jersey_observation(
                    frame,
                    track["bbox"],
                )
                player_observations.append(
                    {
                        **observation,
                        "frame": frame_index,
                    }
                )

        track_metadata = {}
        for player_id, player_observations in observations.items():
            decision = _track_team_decision(
                player_observations,
                self.team_colors,
                allow_guided_fallback=self.assignment_mode == "user_colors",
            )
            team_id = decision["team_id"]
            self.player_team_dict[player_id] = team_id if team_id is not None else -1
            self.player_team_votes[player_id] = [team_id] if team_id is not None else []
            track_metadata[str(int(player_id))] = decision
            if team_id is None:
                logger.info(
                    "Player %s remains unknown: %s",
                    player_id,
                    decision["reason"],
                )

        self.assignment_metadata["track_assignments"] = track_metadata

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


def _crop_player_jersey(frame, bbox):
    """Return the torso region used for FashionCLIP fallback evidence."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "FashionCLIP team fallback requires Pillow. Install "
            "backend/requirements.txt."
        ) from error

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    torso_x1 = max(0, x1 + width // 5)
    torso_x2 = min(frame_width, x2 - width // 5)
    torso_y1 = max(0, y1 + height // 5)
    torso_y2 = min(frame_height, y1 + height * 3 // 5)
    image = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if image.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _cluster_fashion_clip_embeddings(embeddings, sample_track_ids):
    """Cluster normalized FashionCLIP samples and reject weak two-team splits."""
    by_track = {}
    for embedding, player_id in zip(embeddings, sample_track_ids):
        by_track.setdefault(player_id, []).append([float(value) for value in embedding])
    track_ids = sorted(by_track)
    if len(track_ids) < 4:
        return {
            "status": "needs_team_colors",
            "reason": "insufficient_fashion_clip_tracks",
            "track_assignments": {},
            "confidence": {"eligible_track_count": len(track_ids)},
        }

    representatives = [
        _normalize_vector(_mean_vectors(by_track[player_id]))
        for player_id in track_ids
    ]
    first_index, second_index = max(
        (
            (first, second)
            for first in range(len(track_ids))
            for second in range(first + 1, len(track_ids))
        ),
        key=lambda pair: 1 - _dot(
            representatives[pair[0]],
            representatives[pair[1]],
        ),
    )
    if first_index == second_index:
        return {
            "status": "needs_team_colors",
            "reason": "indistinct_fashion_clip_embeddings",
            "track_assignments": {},
            "confidence": {"eligible_track_count": len(track_ids)},
        }
    if track_ids[first_index] > track_ids[second_index]:
        first_index, second_index = second_index, first_index
    centroids = [representatives[first_index], representatives[second_index]]

    labels = None
    for _ in range(12):
        similarities = [
            [_dot(vector, centroid) for centroid in centroids]
            for vector in representatives
        ]
        next_labels = [
            0 if row[0] >= row[1] else 1
            for row in similarities
        ]
        if any(next_labels.count(cluster) == 0 for cluster in (0, 1)):
            break
        next_centroids = [
            _normalize_vector(
                _mean_vectors(
                    [
                        vector
                        for vector, label in zip(representatives, next_labels)
                        if label == cluster
                    ]
                )
            )
            for cluster in (0, 1)
        ]
        labels = next_labels
        if all(
            max(abs(left - right) for left, right in zip(new, old)) <= 1e-5
            for new, old in zip(next_centroids, centroids)
        ):
            centroids = next_centroids
            break
        centroids = next_centroids

    if labels is None:
        return {
            "status": "needs_team_colors",
            "reason": "indistinct_fashion_clip_embeddings",
            "track_assignments": {},
            "confidence": {"eligible_track_count": len(track_ids)},
        }

    similarities = [
        [_dot(vector, centroid) for centroid in centroids]
        for vector in representatives
    ]
    labels = [0 if row[0] >= row[1] else 1 for row in similarities]
    margins = [abs(row[0] - row[1]) for row in similarities]
    support = [labels.count(cluster) for cluster in (0, 1)]
    separation = 1 - _dot(centroids[0], centroids[1])
    track_labels = {
        player_id: labels[index]
        for index, player_id in enumerate(track_ids)
    }
    sample_agreement = []
    for embedding, player_id in zip(embeddings, sample_track_ids):
        normalized_embedding = _normalize_vector(embedding)
        sample_similarities = [
            _dot(normalized_embedding, centroid)
            for centroid in centroids
        ]
        sample_label = 0 if sample_similarities[0] >= sample_similarities[1] else 1
        sample_agreement.append(sample_label == track_labels[player_id])
    agreement = sum(sample_agreement) / max(1, len(sample_agreement))
    mean_margin = sum(margins) / len(margins)
    confidence = {
        "sample_count": len(embeddings),
        "eligible_track_count": len(track_ids),
        "cluster_support": support,
        "centroid_separation": round(separation, 4),
        "mean_assignment_margin": round(mean_margin, 4),
        "sample_track_agreement": round(agreement, 4),
    }
    reason = None
    if min(support) < 2:
        reason = "insufficient_fashion_clip_cluster_support"
    elif separation < 0.04:
        reason = "indistinct_fashion_clip_embeddings"
    elif mean_margin < 0.03:
        reason = "weak_fashion_clip_assignment_margin"
    elif agreement < 0.75:
        reason = "unstable_fashion_clip_track_evidence"
    if reason:
        return {
            "status": "needs_team_colors",
            "reason": reason,
            "track_assignments": {},
            "confidence": confidence,
        }
    return {
        "status": "confident",
        "reason": None,
        "track_assignments": {
            int(player_id): track_labels[player_id] + 1
            for player_id in track_ids
        },
        "confidence": confidence,
    }


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _normalize_vector(vector):
    magnitude = math.sqrt(max(0.0, _dot(vector, vector)))
    if magnitude <= 1e-12:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def _mean_vectors(vectors):
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]


def _map_fashion_clip_clusters_to_teams(
    fashion_clip_assignments,
    color_assignments,
):
    """Align arbitrary FashionCLIP cluster ids with established display teams."""
    anchors = [
        (fashion_clip_assignments[player_id], team_id)
        for player_id, team_id in color_assignments.items()
        if team_id in (1, 2)
        and fashion_clip_assignments.get(player_id) in (1, 2)
    ]
    cluster_support = {
        cluster_id: sum(cluster == cluster_id for cluster, _ in anchors)
        for cluster_id in (1, 2)
    }
    if len(anchors) < 4 or min(cluster_support.values()) < 1:
        return {
            "status": "needs_team_colors",
            "reason": "insufficient_fashion_clip_mapping_anchors",
            "cluster_to_team": {},
            "confidence": {
                "anchor_count": len(anchors),
                "cluster_anchor_support": cluster_support,
            },
        }
    candidates = [
        {1: 1, 2: 2},
        {1: 2, 2: 1},
    ]
    scored = [
        sum(mapping[cluster] == team for cluster, team in anchors)
        for mapping in candidates
    ]
    best_index = 0 if scored[0] >= scored[1] else 1
    agreement = scored[best_index] / len(anchors)
    confidence = {
        "anchor_count": len(anchors),
        "cluster_anchor_support": cluster_support,
        "mapping_agreement": round(agreement, 4),
    }
    if agreement < 0.75:
        return {
            "status": "needs_team_colors",
            "reason": "unstable_fashion_clip_team_mapping",
            "cluster_to_team": {},
            "confidence": confidence,
        }
    return {
        "status": "confident",
        "reason": None,
        "cluster_to_team": candidates[best_index],
        "confidence": confidence,
    }


def _arbitrate_track_assignment(color_decision, fashion_team_id):
    """Resolve per-track color/FashionCLIP conflicts conservatively."""
    if fashion_team_id not in (1, 2):
        return {"team_id": None, "reason": "missing_fashion_clip_track_assignment"}

    weights = {
        int(team_id): float(weight)
        for team_id, weight in color_decision.get("team_vote_weights", {}).items()
        if int(team_id) in (1, 2)
    }
    confident_count = int(color_decision.get("confident_observation_count") or 0)
    color_team_id = None
    if len(weights) == 2 and weights[1] != weights[2]:
        color_team_id = max(weights, key=weights.get)
    weight_share = float(color_decision.get("weight_share") or 0.0)

    if color_team_id is None or confident_count < MINIMUM_COLOR_CONFLICT_EVIDENCE:
        return {"team_id": fashion_team_id, "reason": "fashion_clip_track_fallback"}
    if color_team_id == fashion_team_id:
        return {
            "team_id": fashion_team_id,
            "reason": "fashion_clip_confirms_weighted_color",
        }
    if weight_share >= MINIMUM_COLOR_CONFLICT_WEIGHT_SHARE:
        return {
            "team_id": color_team_id,
            "reason": "weighted_color_over_fashion_clip_conflict",
        }
    return {"team_id": None, "reason": "color_fashion_clip_conflict"}


def _unknown_assignment_fraction(player_tracks, assignments):
    total_observations = sum(len(frame_tracks) for frame_tracks in player_tracks)
    if total_observations == 0:
        return 0.0
    unknown_observations = sum(
        1
        for frame_tracks in player_tracks
        for player_id in frame_tracks
        if assignments.get(player_id) not in (1, 2)
    )
    return round(unknown_observations / total_observations, 4)


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
    observation["quality_score"] = _observation_quality_score(observation)
    if edge_clipped:
        observation.update(
            feature=None,
            accepted=False,
            rejection_reason="edge_clipped",
            quality_score=0.0,
        )
    elif actual_torso_area < minimum_torso_area:
        observation.update(
            feature=None,
            accepted=False,
            rejection_reason="torso_too_small",
            quality_score=0.0,
        )
    elif observation.get("used_visible_fallback"):
        observation.update(
            feature=None,
            accepted=False,
            rejection_reason="insufficient_jersey_pixels",
            quality_score=0.0,
        )
    return observation


def _observation_quality_score(observation):
    """Score usable crop evidence without treating texture as jersey identity."""
    if observation.get("feature") is None:
        return 0.0
    torso_area = observation.get("torso_area", 0)
    minimum_area = max(1, observation.get("minimum_torso_area", 1))
    area_score = min(1.0, torso_area / (minimum_area * 2))
    visible_score = min(1.0, observation.get("visible_fraction", 0.0) / 0.5)
    filtered_score = min(1.0, observation.get("filtered_fraction", 0.0) / 0.5)
    blur_score = min(1.0, observation.get("blur_variance", 0.0) / 100.0)
    return round(
        0.45 * area_score
        + 0.25 * filtered_score
        + 0.20 * visible_score
        + 0.10 * blur_score,
        4,
    )


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
    maximum_distance_ratio=MAXIMUM_PROTOTYPE_DISTANCE_RATIO,
):
    """Return a team only when evidence is both close and clearly separated."""
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

    nearest_distance = min(distances.values())
    if nearest_distance / prototype_separation > maximum_distance_ratio:
        return None

    margin = abs(distances[team_ids[0]] - distances[team_ids[1]])
    if margin / prototype_separation < minimum_margin_ratio:
        return None
    return min(distances, key=distances.get)


def _track_team_decision(
    observations,
    team_colors,
    allow_guided_fallback=False,
):
    decision = _strict_track_team_decision(observations, team_colors)
    if decision["team_id"] is not None or not allow_guided_fallback:
        return decision
    guided = _guided_track_team_decision(observations, team_colors)
    if guided is None:
        return decision
    guided["automatic_rejection_reason"] = decision["reason"]
    guided["guidance"] = "user_colors_relaxed_nearest"
    return guided


def _strict_track_team_decision(observations, team_colors):
    """Aggregate temporally distributed, confidence-weighted track evidence."""
    if len(team_colors) != 2:
        return _empty_track_decision(
            observations,
            "missing_team_prototypes",
        )

    team_ids = list(team_colors)
    prototype_separation = _color_distance(
        team_colors[team_ids[0]],
        team_colors[team_ids[1]],
    )
    evidence = []
    for observation in observations:
        feature = observation.get("feature")
        team_id = _confident_nearest_team(feature, team_colors)
        distances = _team_distances(feature, team_colors)
        if team_id is None or prototype_separation <= 0:
            continue
        margin_ratio = (
            abs(distances[team_ids[0]] - distances[team_ids[1]])
            / prototype_separation
        )
        nearest_distance_ratio = min(distances.values()) / prototype_separation
        quality_score = float(observation.get("quality_score", 1.0))
        weight = quality_score * min(1.0, margin_ratio)
        evidence.append(
            {
                "frame": int(observation.get("frame", len(evidence))),
                "team_id": int(team_id),
                "quality_score": round(quality_score, 4),
                "margin_ratio": round(margin_ratio, 4),
                "nearest_distance_ratio": round(nearest_distance_ratio, 4),
                "weight": round(weight, 4),
            }
        )

    selected = _temporally_diverse_evidence(
        evidence,
        MAXIMUM_TRACK_EVIDENCE_SAMPLES,
    )
    if len(selected) < MINIMUM_TRACK_CONFIDENT_OBSERVATIONS:
        return _track_decision_result(
            observations,
            selected,
            None,
            "insufficient_confident_observations",
        )

    counts = {
        team_id: sum(item["team_id"] == team_id for item in selected)
        for team_id in team_ids
    }
    weights = {
        team_id: sum(
            item["weight"] for item in selected if item["team_id"] == team_id
        )
        for team_id in team_ids
    }
    highest_weight = max(weights.values())
    leaders = [team_id for team_id, weight in weights.items() if weight == highest_weight]
    if len(leaders) != 1:
        return _track_decision_result(
            observations,
            selected,
            None,
            "tied_track_evidence",
        )

    team_id = leaders[0]
    agreement = counts[team_id] / len(selected)
    total_weight = sum(weights.values())
    weight_share = weights[team_id] / total_weight if total_weight > 0 else 0.0
    if counts[team_id] < MINIMUM_TRACK_CONFIDENT_OBSERVATIONS:
        reason = "insufficient_winning_observations"
        team_id = None
    elif agreement < MINIMUM_TRACK_TEAM_AGREEMENT:
        reason = "inconsistent_track_observations"
        team_id = None
    elif weight_share < MINIMUM_TRACK_WEIGHT_SHARE:
        reason = "insufficient_weighted_support"
        team_id = None
    else:
        reason = None

    return _track_decision_result(
        observations,
        selected,
        team_id,
        reason,
        counts=counts,
        weights=weights,
        agreement=agreement,
        weight_share=weight_share,
    )


def _guided_track_team_decision(observations, team_colors):
    """Use explicit user colors to classify any track with usable jersey evidence."""
    if len(team_colors) != 2:
        return None
    team_ids = list(team_colors)
    prototype_separation = _color_distance(
        team_colors[team_ids[0]],
        team_colors[team_ids[1]],
    )
    if prototype_separation <= 0:
        return None

    evidence = []
    for observation in observations:
        feature = observation.get("feature")
        if feature is None:
            continue
        distances = _team_distances(feature, team_colors)
        if len(distances) != 2 or any(value is None for value in distances.values()):
            continue
        team_id = min(distances, key=distances.get)
        quality_score = max(0.05, float(observation.get("quality_score", 1.0)))
        distance_ratio = distances[team_id] / prototype_separation
        weight = quality_score / (1.0 + distance_ratio)
        evidence.append(
            {
                "frame": int(observation.get("frame", len(evidence))),
                "team_id": int(team_id),
                "quality_score": round(quality_score, 4),
                "margin_ratio": round(
                    abs(distances[team_ids[0]] - distances[team_ids[1]])
                    / prototype_separation,
                    4,
                ),
                "nearest_distance_ratio": round(distance_ratio, 4),
                "weight": round(weight, 4),
            }
        )
    selected = _temporally_diverse_evidence(
        evidence,
        MAXIMUM_TRACK_EVIDENCE_SAMPLES,
    )
    if not selected:
        return None
    counts = {
        team_id: sum(item["team_id"] == team_id for item in selected)
        for team_id in team_ids
    }
    weights = {
        team_id: sum(
            item["weight"] for item in selected if item["team_id"] == team_id
        )
        for team_id in team_ids
    }
    highest_weight = max(weights.values())
    leaders = [team_id for team_id, value in weights.items() if value == highest_weight]
    if len(leaders) != 1:
        highest_count = max(counts.values())
        leaders = [team_id for team_id, value in counts.items() if value == highest_count]
    if len(leaders) != 1:
        return None
    team_id = leaders[0]
    agreement = counts[team_id] / len(selected)
    total_weight = sum(weights.values())
    weight_share = weights[team_id] / total_weight if total_weight > 0 else 0.0
    return _track_decision_result(
        observations,
        selected,
        team_id,
        None,
        counts=counts,
        weights=weights,
        agreement=agreement,
        weight_share=weight_share,
    )


def _temporally_diverse_evidence(evidence, maximum_samples):
    if len(evidence) <= maximum_samples:
        return list(evidence)

    selected = []
    for sample_index in range(maximum_samples):
        start = sample_index * len(evidence) // maximum_samples
        end = (sample_index + 1) * len(evidence) // maximum_samples
        selected.append(
            max(
                evidence[start:end],
                key=lambda item: (item["weight"], -item["frame"]),
            )
        )
    return selected


def _empty_track_decision(observations, reason):
    return _track_decision_result(observations, [], None, reason)


def _track_decision_result(
    observations,
    evidence,
    team_id,
    reason,
    *,
    counts=None,
    weights=None,
    agreement=None,
    weight_share=None,
):
    rejection_counts = {}
    for observation in observations:
        if observation.get("accepted", observation.get("feature") is not None):
            continue
        rejection_reason = observation.get("rejection_reason") or "unspecified"
        rejection_counts[rejection_reason] = (
            rejection_counts.get(rejection_reason, 0) + 1
        )
    return {
        "team_id": int(team_id) if team_id is not None else None,
        "status": "assigned" if team_id is not None else "unknown",
        "reason": reason,
        "observation_count": len(observations),
        "accepted_observation_count": sum(
            observation.get("accepted", observation.get("feature") is not None)
            for observation in observations
        ),
        "confident_observation_count": len(evidence),
        "selected_evidence_frames": [item["frame"] for item in evidence],
        "team_vote_counts": {
            str(int(key)): int(value) for key, value in (counts or {}).items()
        },
        "team_vote_weights": {
            str(int(key)): round(value, 4) for key, value in (weights or {}).items()
        },
        "agreement": round(agreement, 4) if agreement is not None else None,
        "weight_share": round(weight_share, 4) if weight_share is not None else None,
        "rejection_counts": rejection_counts,
    }


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
