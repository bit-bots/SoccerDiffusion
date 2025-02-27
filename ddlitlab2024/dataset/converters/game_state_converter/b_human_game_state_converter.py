from enum import Enum, auto

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.models import GameState, Recording, RobotState, TeamColor
from ddlitlab2024.dataset.resampling.original_rate_resampler import OriginalRateResampler


class State(Enum):  # Adapted from b-human's Src/Representations/Infrastructure/GameState.h
    beforeHalf = 0
    standby = auto()
    afterHalf = auto()
    timeout = auto()
    playing = auto()
    setupOwnKickOff = auto()
    setupOpponentKickOff = auto()
    waitForOwnKickOff = auto()
    waitForOpponentKickOff = auto()
    ownKickOff = auto()
    opponentKickOff = auto()
    setupOwnPenaltyKick = auto()
    setupOpponentPenaltyKick = auto()
    waitForOwnPenaltyKick = auto()
    waitForOpponentPenaltyKick = auto()
    ownPenaltyKick = auto()
    opponentPenaltyKick = auto()
    ownPushingFreeKick = auto()
    opponentPushingFreeKick = auto()
    ownKickIn = auto()
    opponentKickIn = auto()
    ownGoalKick = auto()
    opponentGoalKick = auto()
    ownCornerKick = auto()
    opponentCornerKick = auto()
    beforePenaltyShootout = auto()
    waitForOwnPenaltyShot = auto()
    waitForOpponentPenaltyShot = auto()
    ownPenaltyShot = auto()
    opponentPenaltyShot = auto()
    afterOwnPenaltyShot = auto()
    afterOpponentPenaltyShot = auto()

    @classmethod
    def is_playing(cls, state: int) -> bool:
        return state in (
            cls.playing.value,
            cls.ownKickOff.value,
            cls.opponentKickOff.value,
            cls.ownPenaltyKick.value,
            cls.opponentPenaltyKick.value,
            cls.ownPushingFreeKick.value,
            cls.opponentPushingFreeKick.value,
            cls.ownKickIn.value,
            cls.opponentKickIn.value,
            cls.ownGoalKick.value,
            cls.opponentGoalKick.value,
            cls.ownCornerKick.value,
            cls.opponentCornerKick.value,
            cls.ownPenaltyShot.value,
            cls.opponentPenaltyShot.value,
        )

    @classmethod
    def is_stopped(cls, state: int) -> bool:
        return state in (
            cls.beforeHalf.value,
            cls.standby.value,
            cls.afterHalf.value,
            cls.timeout.value,
            cls.setupOwnKickOff.value,
            cls.setupOpponentKickOff.value,
            cls.waitForOwnKickOff.value,
            cls.waitForOpponentKickOff.value,
            cls.ownKickOff.value,
            cls.opponentKickOff.value,
        )

    @classmethod
    def is_positioning(cls, state: int) -> bool:
        return state in (
            cls.setupOwnKickOff.value,
            cls.setupOpponentKickOff.value,
            cls.setupOwnPenaltyKick.value,
            cls.setupOpponentPenaltyKick.value,
        )


class PlayerState(Enum):  # Adapted from b-human's Src/Representations/Infrastructure/GameState.h
    unstiff = 0
    calibration = auto()
    penalizedManual = auto()
    penalizedIllegalBallContact = auto()
    penalizedPlayerPushing = auto()
    penalizedIllegalMotionInSet = auto()
    penalizedInactivePlayer = auto()
    penalizedIllegalPosition = auto()
    penalizedLeavingTheField = auto()
    penalizedRequestForPickup = auto()
    penalizedLocalGameStuck = auto()
    penalizedIllegalPositionInSet = auto()
    penalizedPlayerStance = auto()
    penalizedIllegalMotionInStandby = auto()
    substitute = auto()
    active = auto()

    @classmethod
    def is_penalized(cls, state: int) -> bool:
        return state in (
            cls.penalizedManual.value,
            cls.penalizedIllegalBallContact.value,
            cls.penalizedPlayerPushing.value,
            cls.penalizedIllegalMotionInSet.value,
            cls.penalizedInactivePlayer.value,
            cls.penalizedIllegalPosition.value,
            cls.penalizedLeavingTheField.value,
            cls.penalizedRequestForPickup.value,
            cls.penalizedLocalGameStuck.value,
            cls.penalizedIllegalPositionInSet.value,
            cls.penalizedIllegalMotionInStandby.value,
            cls.penalizedPlayerStance.value,
            cls.substitute.value,
        )


class BHumanGameStateConverter(Converter):
    def __init__(self, resampler: OriginalRateResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data, recording: Recording):
        # B-Human uses an TeamColor-Enum with the same ordering as we do.
        # They use an int-Enum, but we have a str-enum and expect a string.
        # That's why we use the int as an index to the TeamColor str-enum.
        team_color = list(TeamColor)[data.game_state["ownTeam"]["fieldPlayerColor"]].value

        if recording.team_color is None:
            recording.team_color = team_color

        team_color_changed = recording.team_color != team_color

        if team_color_changed:
            logger.warning("The team color changed, during one recording! This will be ignored.")

    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        models = ModelData()

        for sample in self.resampler.resample(data, relative_timestamp):
            models.game_states.append(self._create_game_state(sample.data.game_state, sample.timestamp, recording))

        return models

    def _create_game_state(self, msg, sampling_timestamp: float, recording: Recording) -> GameState:
        return GameState(stamp=sampling_timestamp, recording=recording, state=self._get_state(msg))

    def _get_state(self, data) -> RobotState:
        if State.is_positioning(data["state"]):
            return RobotState.POSITIONING

        if PlayerState.is_penalized(data["playerState"]) or State.is_stopped(data["state"]):
            return RobotState.STOPPED

        if State.is_playing(data["state"]):
            return RobotState.PLAYING

        return RobotState.UNKNOWN
