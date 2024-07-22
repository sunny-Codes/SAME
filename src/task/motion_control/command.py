class Command:
    def __init__(self) -> None:
        self.input_mode = None

    def step(frame):
        pass


class AutoCommand(Command):
    def __init__(self, controller) -> None:
        super(AutoCommand, self).__init__()
        self.input_mode = "auto"
        self.target_begin = 0
        self.target_interval = 50
        self.controller = controller

    def step(self, frame):
        if frame < self.controller.max_load_frame:
            self.controller.updateGoal_fromRecord(frame)
            return

        if frame - self.target_begin > self.target_interval:
            self.controller.updateGoalRandom(frame)
            self.target_begin = frame

    def reset(self):
        self.target_begin = 0
