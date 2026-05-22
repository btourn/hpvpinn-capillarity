
import equinox as eqx
import jax
import jax.numpy as jnp


class FeedForwardNN(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, width: int, depth: int, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)):
        #key_net, key_init = jax.random.split(key)
        self.net = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=activation,
            key=key,
        )

        # Apply Xavier initialization
        #self.net = self._xavier_init(self.net, key_init)

    def _xavier_init(self, net, key):

        keys = jax.random.split(key, len(net.layers))

        new_layers = []

        for layer, k in zip(net.layers, keys):

            in_dim = layer.weight.shape[1]
            out_dim = layer.weight.shape[0]

            limit = 1.5 * jnp.sqrt(6.0 / (in_dim + out_dim))  # Scaled Xavier limit by 1.5

            w_key, b_key = jax.random.split(k)

            new_weight = jax.random.uniform(
                w_key,
                (out_dim, in_dim),
                minval=-limit,
                maxval=limit,
            )

            new_bias = jnp.zeros((out_dim,))

            new_layer = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                layer,
                (new_weight, new_bias),
            )

            new_layers.append(new_layer)

        return eqx.tree_at(lambda m: m.layers, net, new_layers)

    def __call__(self, x):
        raw = self.net(jnp.atleast_1d(x))[0]
        return 2.0 * jax.nn.sigmoid(-x * jax.nn.softplus(raw))