"""A complete physics-driven multi-lap race verifies without controller inference."""

from dataclasses import replace

from racing.agents.scripted import ScriptedAgent
from racing.race.config import RaceConfig
from racing.race.replay import ReplayRecorder, load_replay, verification_state
from racing.race.session import RaceSession
from racing.tracks import Track


class PerCarScripted:
    def __init__(self, car_id):
        self.car_id = car_id
        self.agent = ScriptedAgent()

    def reset(self, seed):
        self.agent.reset(seed)

    def act(self, observation, snapshot, track):
        car = next(car for car in snapshot.cars if car.car_id == self.car_id)
        return self.agent.act(observation, replace(snapshot, cars=(car,)), track)


def test_complete_driven_three_lap_replay(tmp_path):
    track = Track.from_json("assets/tracks/demo.json")
    controllers = {car_id: PerCarScripted(car_id) for car_id in ("human", "ai")}
    session = RaceSession(track, config=RaceConfig(countdown_seconds=0), controllers=controllers)
    path = tmp_path / "complete.jsonl"
    recorder = ReplayRecorder(path, session, model_label="Two scripted verification controllers")
    session.start()
    while session.state == "RUNNING":
        session.step()
    assert recorder.closed
    assert session.winner in ("ai", "human", "draw")
    assert max(car.completed_laps for car in session.snapshot().cars) == 3
    replay = load_replay(path)
    assert verification_state(replay.session) == verification_state(session)
    assert replay.session.results == session.results
