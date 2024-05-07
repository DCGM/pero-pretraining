import torch

class VICRegLoss(torch.nn.Module):
    def __init__(self, variance_weight=1.0, invariance_weight=1.0, covariance_weight=1.0, variance_threshold=1.0):
        super().__init__()
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight
        self.variance_threshold = variance_threshold

        self.eps = 1e-5

    def forward(self, x, y, image_masks1, image_masks2, shift_masks1, shift_masks2):
        inv_x = x[shift_masks1 == 1]
        inv_y = y[shift_masks2 == 1]
        invariance_loss = torch.nn.functional.mse_loss(inv_x, inv_y)

        image_x = x[image_masks1 == 1]
        image_y = y[image_masks2 == 1]
        image_xy = torch.cat([image_x, image_y], dim=0)
        variance_loss = self._variance(image_xy)
        covariance_loss = self._covariance(image_xy)

        loss = (self.variance_weight * variance_loss +
                self.invariance_weight * invariance_loss +
                self.covariance_weight * covariance_loss)

        result = {
            'loss': loss,
            'loss.variance': variance_loss,
            'loss.invariance': invariance_loss,
            'loss.covariance': covariance_loss,
        }

        return result

    def _variance(self, z: torch.Tensor):
        return torch.mean(torch.nn.functional.relu(self.variance_threshold - torch.sqrt(torch.var(z, dim=0) + self.eps)))

    def _covariance(self, z: torch.Tensor):
        mean_z = torch.mean(z, dim=0)
        cov_z = ((z - mean_z).T @ (z - mean_z))/(z.shape[0] - 1)
        return torch.sum(self._off_diagonal(cov_z) ** 2) / z.shape[1]

    def _off_diagonal(self, z: torch.Tensor):
        n, m = z.shape
        return z.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, y, image_masks1, image_masks2, shift_masks1, shift_masks2):
        loss = torch.zeros(x.shape[0])
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)

        for i, (line_x, line_y, shift_mask1, shift_mask2, image_mask1, image_mask2) \
                in enumerate(zip(x, y, shift_masks1, shift_masks2, image_masks1, image_masks2)):
            loss[i] = self._line_contrastive_loss(line_x, line_y, shift_mask1, shift_mask2, image_mask1, image_mask2)

        loss = loss.mean()

        result = {
            'loss': loss,
        }

        return result

    def _line_contrastive_loss(self, x, y, shift_mask_x, shift_mask_y, image_mask_x, image_mask_y):
        x = x[shift_mask_x == 1]
        y = y[shift_mask_y == 1]

        similarities = torch.mm(x, y.t()) / self.temperature
        similarities = similarities[image_mask_x == 1, :][:, image_mask_y == 1]

        loss = -torch.log(torch.diag(torch.exp(similarities)) / torch.sum(torch.exp(similarities), dim=0))
        loss = loss.mean()

        return loss
