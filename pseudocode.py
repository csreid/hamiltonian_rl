z_mid = 0.5 * (z_curr + z_next)
z_mid = z_mid.detach().requires_grad_(True)

H_mid = H_net(z_mid)
grad_H = torch.autograd.grad(H_mid.sum(), z_mid, create_graph=True)[0]

dH_predicted = -dt * (grad_H @ R @ grad_H.T).diagonal() + dt * (grad_H * Bu).sum(-1)

dH_actual = H(z_next.detach()) - H_net(z_curr.detach())

consistency_loss = (dH_actual - dH_predicted).pow(2).mean()
