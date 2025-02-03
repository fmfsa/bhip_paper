import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import plate, sample
from numpyro import infer, handlers, deterministic

def hierarchical_model(N, D, E, e, X, y):
    """
    Hierarchical Bayesian model.

    Parameters:
    - N: Number of observations
    - D: Number of covariates
    - E: Number of environments
    - e: Array of environment indices for each observation (shape: [N])
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (shape: [N])
    """
    # Priors for mu and tau
    mu = sample("mu", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    tau = sample("tau", dist.HalfCauchy(jnp.ones(D)))

    # Sample beta for each environment
    with plate("E_plate", E):
        beta = sample("beta", dist.Normal(mu, tau).to_event(1))

    # Compute the mean for each observation
    mean = jnp.sum(X * beta[e, :], axis=1)
    sample("obs", dist.Normal(mean, 1.0), obs=y)

def hierarchical_logistic_model(N, D, E, e, X, y):
    """
    Hierarchical Bayesian logistic regression model.

    Parameters:
    - N: Number of observations
    - D: Number of covariates
    - E: Number of environments
    - e: Array of environment indices for each observation (shape: [N])
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (binary, shape: [N])
    """
    mu = sample("mu", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    tau = sample("tau", dist.HalfCauchy(jnp.ones(D)))

    with plate("E_plate", E):
        beta = sample("beta", dist.Normal(mu, tau).to_event(1))

    linear_pred = jnp.sum(X * beta[e, :], axis=1)
    sample("obs", dist.Bernoulli(logits=linear_pred), obs=y)

def hierarchical_multinomial_model(N, D, E, e, X, y):
    """
    Hierarchical multinomial logistic regression model for categorical outcomes.

    Parameters:
    - N: Number of observations
    - D: Number of covariates
    - E: Number of environments
    - e: Array of environment indices for each observation (shape: [N])
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (categorical, shape: [N])
    - num_classes: Number of unique categories in the outcome
    """
    mu = sample("mu", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    tau = sample("tau", dist.HalfCauchy(jnp.ones(D)))

    with plate("E_plate", E):
        beta = sample("beta", dist.Normal(mu, tau).to_event(1))

    logits = jnp.sum(X * beta[e, :], axis=1)

    sample("obs", dist.Categorical(logits=logits), obs=y)

def spike_slab_hierarchical_model(N, D, E, e, X, y):
    """
    Spike-and-Slab with partial pooling across E environments, discrete z in {0,1}.
    We'll use DiscreteHMCGibbs to handle z, and HMC for the rest.

    - N: number of data points
    - D: number of predictors
    - E: number of environments
    - e: (N,) environment indices
    - X: (N, D)
    - y: (N,) outcome
    """

    sigma_obs = sample("sigma_obs", dist.HalfCauchy(1.0))

    # Plate over D (predictors)
    with plate("predictors", D):
        p_slab = sample("p_slab", dist.Beta(1., 1.))
        # z[d] is discrete
        z_ = sample("z", dist.Bernoulli(p_slab))

        mu_ = sample("mu", dist.Normal(0., 1.))
        tau_ = sample("tau", dist.HalfCauchy(1.))
        spike_scale_ = sample("spike_scale", dist.HalfCauchy(0.3))

    # Plate over E (environments)
    with plate("envs", E):
        slabBeta = sample("slabBeta", dist.Normal(mu_, tau_).to_event(1))     # (E, D)
        spikeBeta = sample("spikeBeta", dist.Normal(0., spike_scale_).to_event(1))

    z_2d = jnp.broadcast_to(z_[None, :], (E, D))
    beta_env = z_2d * slabBeta + (1.0 - z_2d) * spikeBeta  # shape (E, D)
    deterministic("beta", beta_env)

    # Construct linear predictor
    mean = jnp.sum(X * beta_env[e, :], axis=1)

    # Plate for data dimension (avoid shape errors)
    with plate("data", N):
        sample("obs", dist.Normal(mean, sigma_obs), obs=y)

def horseshoe_hierarchical_model(N, D, E, e, X, y):
    """
    Horseshoe prior for D predictors. Each environment e has its own coefficient for each predictor d.
    The coefficient ~ Normal(0, tau_global * lambda_d), ignoring environment-level partial pooling for now.

    - N: number of observations
    - D: number of predictors
    - E: number of environments
    - e: (N,) environment index for each data row
    - X: (N, D) feature matrix
    - y: (N,) target values
    """

    # 1) Observation noise
    sigma_obs = sample("sigma_obs", dist.HalfCauchy(1.0))

    # 2) Global scale for horseshoe
    tau_global = sample("tau_global", dist.HalfCauchy(1.0))  # ~ HalfCauchy(1.0) e.g.

    # 3) Local scales for each predictor
    with plate("predictors", D):
        lambda_local = sample("lambda_local", dist.HalfCauchy(1.0))

    # 4) Environment-level coefficients
    with plate("envs", E):

        beta_env = sample(
            "beta",
            dist.Normal(0., tau_global * lambda_local).to_event(1)
        )

    # 5) Linear predictor
    mean = jnp.sum(X * beta_env[e, :], axis=1)

    # 6) Observations
    with plate("data", N):
        sample("obs", dist.Normal(mean, sigma_obs), obs=y)


def heteroskedastic_hierarchical_model(N, D, E, e, X, y):
    """
    Hierarchical heteroskedastic Bayesian model.

    Parameters:
    - N: Number of observations
    - D: Number of covariates
    - E: Number of environments
    - e: Array of environment indices for each observation (shape: [N])
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (shape: [N])
    """
    # Priors for mu and tau
    mu = sample("mu", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    tau = sample("tau", dist.HalfCauchy(jnp.ones(D)))
    sigma = sample("sigma", dist.HalfCauchy(1.0))

    # Sample beta for each environment
    with plate("E_plate", E):
        beta = sample("beta", dist.Normal(mu, tau).to_event(1))

    # Compute the mean for each observation
    mean = jnp.sum(X * beta[e, :], axis=1)
    sample("obs", dist.Normal(mean, sigma), obs=y)

def heteroskedastic_logistic_hierarchical_model(N, D, E, e, X, y):
    """
    Hierarchical heteroskedastic Bayesian logistics model.

    Parameters:
    - N: Number of observations
    - D: Number of covariates
    - E: Number of environments
    - e: Array of environment indices for each observation (shape: [N])
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (shape: [N])
    """
    mu = sample("mu", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    tau = sample("tau", dist.HalfCauchy(jnp.ones(D)))

    log_sigma = sample("log_sigma", dist.Normal(0.0, 1.0))
    sigma = jnp.exp(log_sigma)

    with plate("E_plate", E):
        beta = sample("beta", dist.Normal(mu, tau).to_event(1))

    linear_pred = jnp.sum(X * beta[e, :], axis=1)
    scaled_logits = linear_pred / sigma

    sample("obs", dist.Bernoulli(logits=scaled_logits), obs=y)

def nc_hierarchical_model_general(N, D, E, e, X, y, model_func=hierarchical_model, centered=0.0):
    """
    Generalized hierarchical model with non-centered parameterization.
    """
    with handlers.reparam(config={"beta": infer.reparam.LocScaleReparam(centered=centered)}):
        model_func(N, D, E, e, X, y)