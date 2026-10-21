"""Browsable catalog of compatible trained policies for Watch AI."""

import re
from dataclasses import dataclass
from pathlib import Path

import pygame

from racing.render.style import PALETTE as P
from racing.render.style import font

_STEP = re.compile(r"step_(\d+)_update_\d+")


@dataclass(frozen=True, slots=True)
class ModelEntry:
    path: Path
    label: str


class ModelChooser:
    def __init__(self, root):
        root = Path(root)
        candidates = sorted(
            path for path in root.rglob("*.zip")
            if path.with_suffix(".metadata.json").is_file()
        ) if root.is_dir() else []
        maxima = {}
        for path in candidates:
            match = _STEP.fullmatch(path.parent.name)
            if match and path.name == "model.zip":
                run = path.parent.parent.parent
                maxima[run] = max(maxima.get(run, 0), int(match.group(1)))
        self.entries = tuple(sorted(
            (self._entry(path, maxima) for path in candidates),
            key=lambda entry: self._sort_key(entry.path),
        ))
        self.index = 0

    @staticmethod
    def _sort_key(path):
        if path.name == "best_model.zip":
            return (0, str(path))
        if path.name == "final_model.zip":
            return (1, str(path))
        return (2, str(path))

    @staticmethod
    def _entry(path, maxima):
        match = _STEP.fullmatch(path.parent.name)
        if match and path.name == "model.zip":
            step = int(match.group(1))
            run = path.parent.parent.parent
            ratio = step / maxima[run]
            stage = "Early" if ratio <= 1 / 3 else "Mid" if ratio <= 2 / 3 else "Late"
            label = f"{stage} training · {step:,} steps · {run.name}"
        elif path.name == "best_model.zip":
            label = f"Best model · {path.parent.name}"
        elif path.name == "final_model.zip":
            label = f"Final model · {path.parent.name}"
        else:
            label = f"Saved model · {path.parent.name}"
        return ModelEntry(path, label)

    @property
    def selected(self):
        return self.entries[self.index]

    def move(self, delta):
        self.index = (self.index + delta) % len(self.entries)

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill(P["grass_dark"])
        margin, gap = 18, 10
        surface.blit(font(28, True).render("WATCH A TRAINED AI", True, P["accent"]), (margin, 12))
        count = len(self.entries)
        subtitle = f"{self.index + 1 if count else 0} / {count} · Best, final, and learning snapshots"
        surface.blit(font(14).render(subtitle, True, P["ivory"]), (margin, 48))
        buttons = {}
        if count:
            rows = 5 if height >= 560 else 4
            page = self.index // rows
            top = 82
            row_height = (height - top - 90 - gap * (rows - 1)) // rows
            for offset, entry in enumerate(self.entries[page * rows:(page + 1) * rows]):
                index = page * rows + offset
                rect = pygame.Rect(margin, top + offset * (row_height + gap), width - margin * 2, row_height)
                pygame.draw.rect(surface, P["panel"], rect, border_radius=8)
                pygame.draw.rect(surface, P["accent"] if index == self.index else P["asphalt_light"], rect, 2, border_radius=8)
                surface.blit(font(17, True).render(entry.label, True, P["ivory"]), (rect.x + 12, rect.y + 12))
                path_text = str(entry.path)
                while font(13).size(path_text)[0] > rect.width - 24:
                    path_text = path_text[8:]
                surface.blit(font(13).render(path_text, True, P["secondary"]), (rect.x + 12, rect.y + 40))
                buttons[f"model:{index}"] = rect
        else:
            surface.blit(font(16).render("No compatible trained models were found.", True, P["accent"]), (margin, 90))
        labels = (("model_prev", "<"), ("model_next", ">"), ("select_model", "Watch [Enter]"), ("model_back", "Back [Esc]"))
        widths = (42, 42, 160, 140)
        x = margin
        for (action, label), button_width in zip(labels, widths):
            rect = pygame.Rect(x, height - 58, button_width, 36)
            pygame.draw.rect(surface, P["panel"], rect, border_radius=6)
            surface.blit(font(14, True).render(label, True, P["ivory"]), (rect.x + 9, rect.y + 9))
            buttons[action] = rect
            x += button_width + 8
        return buttons
