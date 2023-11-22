from torch.optim.lr_scheduler import _LRScheduler


class MultistepWarmUpRestargets(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.5, T_up=1, last_epoch=-1):
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_up = T_up
        self.T_step = 0
        self.c = 0
        self.multiply = 1
        self.T_cur = -1
        self.milestones = milestones
        self.gamma = gamma
        super(MultistepWarmUpRestargets, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_step < self.T_up:
            return [(self.T_step / self.T_up) * i * self.multiply for i in self.base_lrs]
        else:
            return [i * self.multiply for i in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur += 1
        self.T_step = self.T_step + 1
        try:
            if self.T_cur >= self.milestones[self.c]:
                self.c += 1
                self.T_step = 0
                self.multiply *= self.gamma
        except:
            pass
