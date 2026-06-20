from pathlib import Path

import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.app.utils import load_cache, save_cache


class TeamAssigner:
    """Assigns players to teams using the reference repo's FashionCLIP flow."""

    def __init__(
        self,
        team_1_description="white jersey",
        team_2_description="dark jersey",
        team_1_color=(255, 245, 238),
        team_2_color=(128, 0, 0),
        model_name="patrickjohncyh/fashion-clip",

        
    ):
        self.team_colors = {
            1: team_1_color,
            2: team_2_color,
        }
        self.player_team_dict = {}
        self.team_1_class_name = team_1_description
        self.team_2_class_name = team_2_description
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None and self.processor is not None:
            return

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

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

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
            return player_assignment

        self.load_model()
        player_assignment = []

        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            if frame_num % 50 == 0:
                self.player_team_dict = {}

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
    image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
