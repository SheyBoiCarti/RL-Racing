"""Public track catalog and responsive, keyboard-accessible preview screen."""

import pygame

from racing.render.style import PALETTE as P
from racing.render.style import font
from racing.tracks import Track

DEFAULT_MANIFEST = "experiments/splits/v1.json"


class TrackChooser:
    def __init__(self, manifest_path=DEFAULT_MANIFEST, catalog_loader=None):
        self.manifest_path = manifest_path
        self.tracks = {"demo": Track.from_json("assets/tracks/demo.json")}
        self.error = None
        try:
            if catalog_loader is None:
                from racing.tracks.splits import load_split_catalog

                catalog_loader = load_split_catalog
            generated = {}
            for split in ("train", "validation"):
                generated.update({key: track for key, track in catalog_loader(manifest_path, split).items()
                                  if key.startswith(split + "_") and track.split == split})
            self.tracks.update(generated)
        except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
            self.error = f"{manifest_path}: {error}"
        self.ids = sorted(self.tracks, key=lambda key: (key != "demo", key))
        self.index = 0

    @property
    def selected_id(self):
        return self.ids[self.index]

    def move(self, delta):
        self.index = (self.index + delta) % len(self.ids)

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill(P["grass_dark"])
        margin, gap = 18, 12
        surface.blit(font(28, True).render("CHOOSE YOUR CIRCUIT", True, P["accent"]), (margin, 12))
        subtitle = f"{self.index + 1} / {len(self.ids)} · Demo, training and validation"
        surface.blit(font(14).render(subtitle, True, P["ivory"]), (margin, 48))
        columns = 4 if width >= 1000 else 2
        rows = 2 if height >= 560 else 1
        count = columns * rows
        page = self.index // count
        card_w = (width - 2 * margin - gap * (columns - 1)) // columns
        card_h = (height - 166 - gap * (rows - 1)) // rows
        buttons = {}
        for offset, key in enumerate(self.ids[page * count:(page + 1) * count]):
            track = self.tracks[key]
            rect = pygame.Rect(margin + (offset % columns) * (card_w + gap),
                               76 + (offset // columns) * (card_h + gap), card_w, card_h)
            pygame.draw.rect(surface, P["panel"], rect, border_radius=8)
            pygame.draw.rect(surface, P["accent"] if key == self.selected_id else P["asphalt_light"],
                             rect, 2, border_radius=8)
            low, high = track.outer.min(axis=0), track.outer.max(axis=0)
            scale = min((rect.width - 28) / (high[0] - low[0]),
                        (rect.height - 66) / (high[1] - low[1]))
            center = (low + high) / 2

            def screen(point):
                return (rect.centerx + (point[0] - center[0]) * scale,
                        rect.y + (rect.height - 48) / 2 + (point[1] - center[1]) * scale)

            pygame.draw.polygon(surface, P["asphalt"], [screen(p) for p in track.outer])
            pygame.draw.polygon(surface, P["panel"], [screen(p) for p in track.inner])
            for boundary in (track.outer, track.inner):
                pygame.draw.polygon(surface, P["ivory"], [screen(p) for p in boundary], 1)
            _, _, a, b = track.gates[0]
            pygame.draw.line(surface, P["accent"], screen(a), screen(b), 3)
            clip = surface.get_clip()
            surface.set_clip(rect.inflate(-12, 0))
            surface.blit(font(14, True).render(track.name, True, P["ivory"]), (rect.x + 10, rect.bottom - 44))
            caption = f"{key} · {track.length_m:.0f} m"
            surface.blit(font(14).render(caption, True, P["secondary"]), (rect.x + 10, rect.bottom - 24))
            surface.set_clip(clip)
            buttons[f"track:{key}"] = rect
        if self.error:
            panel = pygame.Rect(margin, 76, width - margin * 2, height - 166)
            pygame.draw.rect(surface, P["panel"], panel, border_radius=8)
            words = ("Cannot load selection. You can return safely to the menu. " + self.error).split()
            lines, line = [], ""
            for word in words:
                # Split long absolute filenames too, preserving every character.
                for part in [word[i:i + max(12, (panel.width - 24) // 8)]
                             for i in range(0, len(word), max(12, (panel.width - 24) // 8))]:
                    if font(14).size(line + " " + part)[0] > panel.width - 24:
                        lines.append(line)
                        line = part
                    else:
                        line = (line + " " + part).strip()
            lines.append(line)
            for row, line in enumerate(lines[:max(1, (panel.height - 16) // 20)]):
                surface.blit(font(14).render(line, True, P["accent"]),
                             (panel.x + 12, panel.y + 10 + row * 20))
            # The error panel covers cards, so don't leave hidden hit targets.
            buttons = {}
        labels = (("chooser_prev", "<"), ("chooser_next", ">"),
                  ("select_track", "Select [Enter]"), ("chooser_back", "Back [Esc]"))
        widths = (42, 42, 160, 140)
        x = margin
        for (action, label), button_w in zip(labels, widths):
            rect = pygame.Rect(x, height - 68, button_w, 36)
            pygame.draw.rect(surface, P["panel"], rect, border_radius=6)
            surface.blit(font(14, True).render(label, True, P["ivory"]), (rect.x + 9, rect.y + 9))
            buttons[action] = rect
            x += button_w + 8
        message = ("Select keeps this circuit · Back returns to menu" if self.error else
                   "Arrows browse · Click a preview, then Select · C opens this chooser from the menu")
        while font(13).size(message)[0] > width - 2 * margin and len(message) > 8:
            message = message[:len(message) - 5] + "…"
        surface.blit(font(13).render(message, True, P["accent"] if self.error else P["secondary"]),
                     (margin, height - 25))
        return buttons
