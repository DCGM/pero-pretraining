class WarmupSchleduler:
    def __init__(self, optimizer, base_lr, warm_up_iterations, warm_up_polynomial_order):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warm_up_iterations = warm_up_iterations
        self.warm_up_polynomial_order = warm_up_polynomial_order

        self._last_lr = None

    @property
    def current_lr(self):
        return self._last_lr

    def update_learning_rate(self, iteration_count):
        if self.warm_up_iterations is None or self.warm_up_polynomial_order is None:
            self._last = self.base_lr

        if iteration_count <= self.warm_up_iterations and self.warm_up_iterations > 0:
            lr = ((iteration_count / self.warm_up_iterations) ** self.warm_up_polynomial_order) * self.base_lr
        else:
            lr = self.base_lr

        self._last_lr = lr
        self._set_current_lr()

    def _set_current_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
