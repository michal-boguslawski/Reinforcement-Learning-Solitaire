import torch as T

from .base import OnPolicy
from models.models import OnPolicyMinibatch


class SarsaPolicy(OnPolicy):
    def _calculate_loss(self, batch: OnPolicyMinibatch) -> T.Tensor:
        states, results, actions = (
            batch.states,
            batch.returns,
            batch.actions
        )
        
        output = self.network(states)
        
        q_values = output.logits.gather(dim=-1, index=actions).squeeze(-1)
        loss = self.loss_fn(q_values, results)
        return loss
