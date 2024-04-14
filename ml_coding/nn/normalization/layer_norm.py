import torch

"""
In transformer, why using layerNorm instead of batchNorm?

https://zhuanlan.zhihu.com/p/456863215 
https://github.com/DA-southampton/NLP_ability/blob/master/

Summary:
BN assumes that it is doing scaling under a certain feature across different samples(batches).
mean_bn = x.mean(axis=0);
mean_layer = x.mean(axis=-1) # assume -1 is the feature dim


But for NLP task the inputs are (batch, length, hidden). BN assumes for each postion, the token is the same, which
is incorrect. But for LN, it is doing reduce_norm at hidden dim (output size is [batch, length]), which has correct
assumption.


When do forwarding, since the mean and var can be exactly caculated from the tensor, there is no need to store the 
moving avg of the mean and var.



code: reference
https://mp.weixin.qq.com/s/XFniIyQcrxambld5KmXr6Q
"""


class LayerNorm(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features_ = n_features
        # create two trainable parameters to scale the
        self.gamma = torch.nn.Parameter( torch.ones(self.n_features_) )
        self.beta = torch.nn.Parameter( torch.zeros(self.n_features_))
        self.eps = 1e-12

    def forward(self, x):
        """
        x: with shape (batch, channel1, channel2,..., feature)

        return turn tensor shape will be the same as x
        """
        # The layer norm conduct reduce_mean and reduce_var at the last（feature） dimenstion

        # mean_shape: (batch, channel1, chanel2,...)
        mean = x.mean(dim=-1, keepdim=True)


        # when calculating the var, use unbiased=False otherwide it will use (n - 1) as Denominator
        # var shape: (batch, channel1, chanel2,...)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # self.gamma with broadcast to (batch, channel1, channel2, ..., feature
        x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return x

